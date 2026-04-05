"""
EOPD trainer built on top of verl's PPO driver infrastructure.

Reuses PPO's data contract, validation flow, generation dumping, and checkpoint
management. Swaps in the EOPD training step which combines PPO-style clipped
reverse KL with entropy-gated top-k forward KL.
"""

import logging
import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_response_mask
from verl.utils.metric import reduce_metrics
from verl.utils.reward_score.math_dapo import verify

from opsd.batch_builder import build_opsd_batch_from_verl_batch

py_logger = logging.getLogger(__name__)


class EOPDTrainer(RayPPOTrainer):
    """EOPD trainer: entropy-aware on-policy distillation."""

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, type],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
    ):
        if Role.ActorRollout not in role_worker_mapping and Role.ActorRolloutRef in role_worker_mapping:
            role_worker_mapping = dict(role_worker_mapping)
            role_worker_mapping[Role.ActorRollout] = role_worker_mapping[Role.ActorRolloutRef]

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
        )

        eopd_cfg = config.get("eopd", {})
        self.entropy_threshold = eopd_cfg.get("entropy_threshold", 0.8)
        self.topk = eopd_cfg.get("topk", 16)
        self.clip_epsilon = eopd_cfg.get("clip_epsilon", 0.2)
        self.n_ppo_epochs = eopd_cfg.get("n_ppo_epochs", 4)
        self.chunk_size = eopd_cfg.get("chunk_size", 512)
        self.eopd_max_length = eopd_cfg.get("max_length", 16384)
        self.test_freq = config.trainer.test_freq

        self.teacher_system_prompt = eopd_cfg.get("teacher_system_prompt", None)
        teacher_model_path = config.actor_rollout_ref.ref.get("model", {}).get("path", None)
        actor_model_path = config.actor_rollout_ref.model.path
        self.use_teacher_context = teacher_model_path is None or teacher_model_path == actor_model_path

        apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        if OmegaConf.is_config(apply_chat_template_kwargs):
            apply_chat_template_kwargs = OmegaConf.to_container(apply_chat_template_kwargs, resolve=True)
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

    def _decode_response_texts(self, batch: DataProto) -> list[str]:
        if "response_mask" not in batch.batch:
            batch.batch["response_mask"] = compute_response_mask(batch)

        decoded = []
        for response_ids, response_mask in zip(
            batch.batch["responses"],
            batch.batch["response_mask"],
            strict=True,
        ):
            decoded.append(self.tokenizer.decode(response_ids[response_mask.bool()], skip_special_tokens=True))
        return decoded

    def _pad_eopd_batch_for_dispatch(self, eopd_batch: DataProto) -> DataProto:
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        n_samples = eopd_batch.batch["student_input_ids"].shape[0]
        if n_samples == 0 or n_samples % n_dp == 0:
            return eopd_batch

        pad_to = ((n_samples // n_dp) + 1) * n_dp
        pad_count = pad_to - n_samples
        padded = {}
        for key in list(eopd_batch.batch.keys()):
            tensor = eopd_batch.batch[key]
            if key == "valid_row_mask":
                padded_rows = torch.zeros(pad_count, dtype=tensor.dtype, device=tensor.device)
            else:
                padded_rows = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
            padded[key] = torch.cat([tensor, padded_rows])
        return DataProto.from_single_dict(padded)

    def _validate(self, merged: bool = False):
        """EOPD validation: generate responses and verify against ground truth."""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            val_n = self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1)
            test_batch = test_batch.repeat(repeat_times=val_n, interleave=True)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
            self.checkpoint_manager.sleep_replicas()
            test_output = unpad_dataproto(test_output_padded, pad_size=pad_size)

            test_batch = test_batch.union(test_output)
            test_batch.batch["response_mask"] = compute_response_mask(test_batch)
            responses = self._decode_response_texts(test_batch)

            for response, gt in zip(responses, ground_truths, strict=True):
                if gt is not None:
                    correct, pred = verify(response, gt)
                    acc = float(correct)
                else:
                    acc = 0.0
                    pred = ""
                sample_scores.append(acc)
                reward_extra_infos_dict["reward"].append(acc)
                reward_extra_infos_dict["acc"].append(acc)
                reward_extra_infos_dict["pred"].append(pred)

            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(responses)

            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch))
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, [])

        py_logger.info("Validation results: %s", metrics)
        self.checkpoint_manager.update_weights()
        return metrics

    def fit(self):
        """Main EOPD training loop."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()
        progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="EOPD Training")

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            if val_metrics:
                logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                progress.close()
                return

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if self.global_steps >= self.total_training_steps:
                    progress.close()
                    py_logger.info("EOPD training complete at step %d", self.global_steps)
                    return

                self.global_steps += 1
                step_t0 = time.time()
                metrics = {}

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                saved_raw_prompt = batch.non_tensor_batch.get("raw_prompt")

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": True,
                    "global_steps": self.global_steps,
                }

                gen_t0 = time.time()
                gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
                self.checkpoint_manager.sleep_replicas()
                metrics["timing/generate_s"] = time.time() - gen_t0

                batch = batch.union(gen_output)

                if saved_raw_prompt is not None and "raw_prompt" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["raw_prompt"] = saved_raw_prompt
                batch.batch["response_mask"] = compute_response_mask(batch)

                responses = self._decode_response_texts(batch)
                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
                ]
                batch_size = len(responses)

                n_correct = sum(
                    1
                    for response, ground_truth in zip(responses, ground_truths, strict=True)
                    if ground_truth is not None and verify(response, ground_truth)[0]
                )
                metrics["eopd/accuracy"] = n_correct / max(1, batch_size)
                metrics["eopd/batch_size"] = batch_size

                train_t0 = time.time()
                eopd_batch = build_opsd_batch_from_verl_batch(
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_length=self.eopd_max_length,
                    teacher_system_prompt=self.teacher_system_prompt if self.use_teacher_context else None,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                    use_teacher_context=self.use_teacher_context,
                )

                if eopd_batch is not None:
                    eopd_batch = self._pad_eopd_batch_for_dispatch(eopd_batch)
                    eopd_batch.meta_info["entropy_threshold"] = self.entropy_threshold
                    eopd_batch.meta_info["topk"] = self.topk
                    eopd_batch.meta_info["clip_epsilon"] = self.clip_epsilon
                    eopd_batch.meta_info["n_ppo_epochs"] = self.n_ppo_epochs
                    eopd_batch.meta_info["chunk_size"] = self.chunk_size

                    eopd_output = self.actor_rollout_wg.update_eopd(eopd_batch)
                    eopd_metrics = reduce_metrics(eopd_output.meta_info["metrics"])
                    metrics.update(eopd_metrics)
                else:
                    metrics["eopd/skipped"] = 1.0

                self.checkpoint_manager.update_weights()
                metrics["timing/train_s"] = time.time() - train_t0

                is_last = self.global_steps >= self.total_training_steps
                is_val = self.test_freq > 0 and self.global_steps % self.test_freq == 0
                if is_val or is_last:
                    metrics.update(self._validate())

                save_freq = self.config.trainer.save_freq
                if save_freq > 0 and (is_last or self.global_steps % save_freq == 0):
                    self._save_checkpoint()

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["timing/step_s"] = time.time() - step_t0

                logger.log(data=metrics, step=self.global_steps)
                progress.update(1)

                if is_last:
                    progress.close()
                    py_logger.info("EOPD training complete at step %d", self.global_steps)
                    return

        progress.close()
        py_logger.info("EOPD training complete!")
