"""
Main entry point for EOPD training with verl.

Usage:
    python -m eopd.main_eopd \
        --config-path ./config --config-name eopd_trainer \
        actor_rollout_ref.model.path=/path/to/model \
        data.train_files=/path/to/train.parquet
"""

import logging
import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="eopd_trainer", version_base=None)
def main(config):
    run_eopd(config)


def run_eopd(config):
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(EOPDTaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class EOPDTaskRunner:
    """Ray remote class for EOPD training."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from eopd.eopd_worker import EOPDWorker

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(EOPDWorker)
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(EOPDWorker)
        self.mapping[Role.ActorRollout] = "global_pool"
        self.mapping[Role.ActorRolloutRef] = "global_pool"
        return RayWorkerGroup

    def init_resource_pool_mgr(self, config):
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        from verl.experimental.reward_loop import migrate_legacy_reward_impl
        from verl.utils.fs import copy_to_local

        logger.info("EOPDTaskRunner on %s, PID %d", socket.gethostname(), os.getpid())

        config = migrate_legacy_reward_impl(config)

        try:
            OmegaConf.resolve(config)
            logger.info("Config:\n%s", OmegaConf.to_yaml(config))
        except Exception:
            logger.info("Config (unresolved):\n%s", OmegaConf.to_yaml(config))

        ray_worker_group_cls = self.add_worker(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

        from verl.trainer.main_ppo import create_rl_dataset
        from verl.utils.dataset.rl_dataset import collate_fn as rl_collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, is_train=True
        )

        val_dataset = None
        if config.data.get("val_files"):
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False
            )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from eopd.eopd_trainer import EOPDTrainer

        trainer = EOPDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=rl_collate_fn,
        )

        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
