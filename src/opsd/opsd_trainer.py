"""
OPSD trainer built on top of verl's PPO driver infrastructure.

The PPO trainer already owns the standard verl data contract, validation flow,
generation dumping, and checkpoint management. OPSD reuses that outer shell and
only swaps in a custom train step that converts a rollout batch into paired
teacher/student sequences for divergence minimization.
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
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.metric import reduce_metrics

from .batch_builder import build_opsd_batch_from_verl_batch

py_logger = logging.getLogger(__name__)


class OPSDTrainer(RayPPOTrainer):
    """OPSD trainer that reuses PPO's dataloading, validation, and checkpointing."""

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

        opsd_cfg = config.get("opsd", {})
        self.loss_type = opsd_cfg.get("loss_type", "reverse_kl")
        self.beta = opsd_cfg.get("beta", 0.5)
        self.chunk_size = opsd_cfg.get("chunk_size", 512)
        self.opsd_max_length = opsd_cfg.get("max_length", 16384)
        self.test_freq = config.trainer.test_freq
        self.teacher_system_prompt = opsd_cfg.get("teacher_system_prompt", None)
        # When a separate teacher model is used (ref.model.path != actor model path),
        # skip ground truth injection — the bigger teacher model is inherently better.
        teacher_model_path = config.actor_rollout_ref.ref.get("model", {}).get("path", None)
        actor_model_path = config.actor_rollout_ref.model.path
        self.use_teacher_context = teacher_model_path is None or teacher_model_path == actor_model_path
        self.reward_beta = opsd_cfg.get("reward_beta", None)
        if self.reward_beta is not None and self.reward_beta <= 0:
            self.reward_beta = None
        apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        if OmegaConf.is_config(apply_chat_template_kwargs):
            apply_chat_template_kwargs = OmegaConf.to_container(apply_chat_template_kwargs, resolve=True)
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})

        # Load custom reward function from config (same mechanism as GRPO)
        self.reward_fn = get_custom_reward_fn(config)
        if self.reward_fn is None:
            # Fallback to verl's built-in math_dapo verify for backward compatibility
            from verl.utils.reward_score.math_dapo import verify
            self.reward_fn = lambda solution_str, ground_truth, **kwargs: {
                "score": float(verify(solution_str, ground_truth)[0]),
                "acc": float(verify(solution_str, ground_truth)[0]),
                "pred": verify(solution_str, ground_truth)[1],
            }

    # init_workers() is inherited from RayPPOTrainer -- it handles
    # resource pools, worker group creation, AgentLoopManager, and
    # mode transitions (sleep/wake_up) correctly.

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

    def _pad_opsd_batch_for_dispatch(self, opsd_batch: DataProto) -> DataProto:
        n_dp = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        n_samples = opsd_batch.batch["student_input_ids"].shape[0]
        if n_samples == 0 or n_samples % n_dp == 0:
            return opsd_batch

        pad_to = ((n_samples // n_dp) + 1) * n_dp
        pad_count = pad_to - n_samples
        padded = {}
        for key in list(opsd_batch.batch.keys()):
            tensor = opsd_batch.batch[key]
            if key == "valid_row_mask" or key == "sample_weights":
                padded_rows = torch.zeros(pad_count, dtype=tensor.dtype, device=tensor.device)
            else:
                padded_rows = tensor[-1:].expand(pad_count, *tensor.shape[1:]).clone()
            padded[key] = torch.cat([tensor, padded_rows])
        return DataProto.from_single_dict(padded)

    def _validate(self, merged: bool = False):
        """OPSD validation: generate responses and verify against ground truth.

        Overrides the parent's _validate() which depends on the full reward loop
        infrastructure. OPSD runs the math verifier directly and delegates metric
        aggregation (pass@k, maj@k, mean, std) to verl's process_validation_metrics
        so that val.n > 1 produces proper per-prompt statistics.
        """
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

            # Score each response with the reward function
            for response, gt in zip(responses, ground_truths, strict=True):
                if gt is not None:
                    result = self.reward_fn(solution_str=response, ground_truth=gt)
                    if isinstance(result, dict):
                        acc = float(result.get("acc", result.get("score", 0.0)))
                        pred = result.get("pred", "")
                    else:
                        acc = float(result)
                        pred = ""
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

        # Dump generations if configured
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

        # Use verl's metric aggregation (pass@k, maj@k, mean, std grouped by UID)
        data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])
        metrics = self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, [])

        py_logger.info("Validation results: %s", metrics)
        # Wake SGLang back up for the next rollout
        self.checkpoint_manager.update_weights()
        return metrics

    def fit(self):
        """Main OPSD training loop using PPO's driver-side infra."""
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
        progress = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPSD Training")

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
                    py_logger.info("OPSD training complete at step %d", self.global_steps)
                    return

                self.global_steps += 1
                step_t0 = time.time()
                metrics = {}

                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Save raw_prompt before _get_gen_batch pops it from non_tensor_batch
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

                # Restore raw_prompt (popped by _get_gen_batch since it's not in reward_model_keys)
                if saved_raw_prompt is not None and "raw_prompt" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["raw_prompt"] = saved_raw_prompt
                batch.batch["response_mask"] = compute_response_mask(batch)

                responses = self._decode_response_texts(batch)
                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch
                ]
                batch_size = len(responses)

                rewards = []
                for response, ground_truth in zip(responses, ground_truths, strict=True):
                    if ground_truth is not None:
                        result = self.reward_fn(solution_str=response, ground_truth=ground_truth)
                        acc = float(result.get("acc", result.get("score", 0.0))) if isinstance(result, dict) else float(result)
                        rewards.append(1.0 if acc > 0 else 0.0)
                    else:
                        rewards.append(0.0)
                n_correct = sum(rewards)
                metrics["opsd/accuracy"] = n_correct / max(1, batch_size)
                metrics["opsd/batch_size"] = batch_size

                # Compute per-sample weights from rewards
                sample_weights = None
                if self.reward_beta is not None:
                    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
                    sample_weights = torch.softmax(reward_tensor / self.reward_beta, dim=0) * len(rewards)
                    metrics["opsd/reward_weight_max"] = sample_weights.max().item()
                    metrics["opsd/reward_weight_min"] = sample_weights.min().item()
                    metrics["opsd/n_correct"] = int(n_correct)

                train_t0 = time.time()
                opsd_batch = build_opsd_batch_from_verl_batch(
                    batch=batch,
                    tokenizer=self.tokenizer,
                    max_length=self.opsd_max_length,
                    teacher_system_prompt=self.teacher_system_prompt if self.use_teacher_context else None,
                    apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                    use_teacher_context=self.use_teacher_context,
                    sample_weights=sample_weights,
                )

                if opsd_batch is not None:
                    opsd_batch = self._pad_opsd_batch_for_dispatch(opsd_batch)
                    opsd_batch.meta_info["opsd_loss_type"] = self.loss_type
                    opsd_batch.meta_info["opsd_beta"] = self.beta
                    opsd_batch.meta_info["opsd_chunk_size"] = self.chunk_size
                    if self.reward_beta is not None:
                        opsd_batch.meta_info["opsd_reward_beta"] = self.reward_beta

                    opsd_output = self.actor_rollout_wg.update_opsd(opsd_batch)
                    opsd_metrics = reduce_metrics(opsd_output.meta_info["metrics"])
                    metrics.update(opsd_metrics)
                else:
                    metrics["opsd/skipped"] = 1.0

                # Reload SGLang with updated actor weights for next rollout
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
                    py_logger.info("OPSD training complete at step %d", self.global_steps)
                    return

        progress.close()
        py_logger.info("OPSD training complete!")
