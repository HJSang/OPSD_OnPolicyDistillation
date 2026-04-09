"""
Main entry point for OPD training with verl.

Usage:
    python -m opd.main_opd \
        --config-path ./config --config-name opd_trainer \
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


@hydra.main(config_path="config", config_name="opd_trainer", version_base=None)
def main(config):
    run_opd(config)


def run_opd(config):
    if not ray.is_initialized():
        from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_class = ray.remote(num_cpus=1)(OPDTaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))


class OPDTaskRunner:
    """Ray remote class for OPD training."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role
        from opd.opd_worker import OPDWorker

        # Map both ActorRollout and ActorRolloutRef to OPDWorker.
        # RayPPOTrainer.init_workers() looks up Role.ActorRollout for
        # the hybrid engine path, while OPDTrainer needs ActorRolloutRef
        # for the ref model (frozen teacher).
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(OPDWorker)
        self.role_worker_mapping[Role.ActorRolloutRef] = ray.remote(OPDWorker)
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

        logger.info("OPDTaskRunner on %s, PID %d", socket.gethostname(), os.getpid())

        config = migrate_legacy_reward_impl(config)

        try:
            OmegaConf.resolve(config)
            logger.info("Config:\n%s", OmegaConf.to_yaml(config))
        except Exception:
            logger.info("Config (unresolved):\n%s", OmegaConf.to_yaml(config))

        ray_worker_group_cls = self.add_worker(config)

        # Load tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

        # Create RL dataset (uses verl's standard format with question/ground_truth)
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

        from opd.opd_trainer import OPDTrainer

        trainer = OPDTrainer(
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
