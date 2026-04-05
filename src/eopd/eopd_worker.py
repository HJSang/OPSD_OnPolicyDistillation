"""
EOPD Worker: Entropy-Aware On-Policy Distillation training.

Extends verl's ActorRolloutRefWorker with a two-phase training step:
  Phase 1 (Teacher Query): Run teacher forward once, extract per-token info:
    - log pi_te(x_t | c_t) for the sampled token
    - teacher entropy H_t at each position
    - top-k token indices and renormalized probs
    Also run student in no_grad to get old log-probs (behavior policy).

  Phase 2 (PPO Training): Multiple gradient steps using stored teacher info:
    - Clipped reverse KL with importance sampling (PPO-style)
    - Forward KL over teacher's top-k tokens at high-entropy positions
"""

import logging

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.protocol import DataProto
from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.attention_utils import index_first_axis, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.torch_functional import entropy_from_logits_with_chunking
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .losses import compute_eopd_loss

logger = logging.getLogger(__name__)


def _shift_loss_mask_right_per_sequence(loss_mask_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Shift loss-mask labels within each unpadded sequence."""
    shifted = torch.zeros_like(loss_mask_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        start = int(seq_start.item())
        end = int(seq_end.item())
        if end - start > 1:
            shifted[start : end - 1] = loss_mask_rmpad[start + 1 : end]
    return shifted


def _shift_targets_per_sequence(input_ids_rmpad: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Build next-token targets within each unpadded sequence."""
    targets = torch.zeros_like(input_ids_rmpad)
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        s, e = int(seq_start.item()), int(seq_end.item())
        if e - s > 1:
            targets[s : e - 1] = input_ids_rmpad[s + 1 : e]
    return targets


class EOPDWorker(AsyncActorRolloutRefWorker):
    """Worker with entropy-aware on-policy distillation training."""

    def __init__(self, config, role: str, **kwargs):
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_eopd(self, data: DataProto) -> DataProto:
        """One EOPD training step: teacher query + PPO-style training.

        Input DataProto batch keys:
          teacher_input_ids, teacher_attention_mask, teacher_position_ids, teacher_loss_mask
          student_input_ids, student_attention_mask, student_position_ids, student_loss_mask

        Meta info:
          entropy_threshold, topk, clip_epsilon, n_ppo_epochs, chunk_size
        """
        assert self._is_actor, "update_eopd requires actor role"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        ref_needs_offload = self.config.ref.fsdp_config.get("param_offload", False)
        ref_offloaded = False
        if hasattr(self, "ref_module_fsdp") and self.ref_module_fsdp is not None:
            if ref_needs_offload:
                load_fsdp_model_to_gpu(self.ref_module_fsdp)
                ref_offloaded = True

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            metrics = self._eopd_training_step(data)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["eopd/lr"] = lr.item() if torch.is_tensor(lr) else lr
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if ref_offloaded:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    def _eopd_training_step(self, data: DataProto) -> dict:
        """Two-phase EOPD training: teacher query once, then PPO-style iteration."""
        self.actor_module_fsdp.train()
        self.ref_module_fsdp.eval()

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        micro_batch_size = self.config.actor.get(
            "ppo_micro_batch_size_per_gpu",
            self.config.actor.get("micro_batch_size_per_gpu", 2),
        )

        entropy_threshold = data.meta_info.get("entropy_threshold", 0.8)
        topk = data.meta_info.get("topk", 16)
        clip_epsilon = data.meta_info.get("clip_epsilon", 0.2)
        n_ppo_epochs = data.meta_info.get("n_ppo_epochs", 4)
        chunk_size = data.meta_info.get("chunk_size", 512)

        batch_size = data.batch["student_input_ids"].shape[0]
        if batch_size == 0:
            return {"eopd/total_loss": 0.0, "eopd/num_tokens": 0}

        micro_batches = data.split(micro_batch_size)
        device = get_device_id()

        # ================================================================
        # Phase 1: Teacher Query (once, no grad)
        # For each micro-batch, run teacher and old-student forward passes,
        # extract per-token info, then store on CPU to free GPU memory.
        # ================================================================
        mb_teacher_data = []  # list of dicts per micro-batch

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(device)

            valid_row_mask = micro_batch.batch.get("valid_row_mask")
            if valid_row_mask is not None:
                valid_row_mask = valid_row_mask.bool()
                if not valid_row_mask.any():
                    mb_teacher_data.append(None)
                    continue

            t_input_ids = micro_batch.batch["teacher_input_ids"]
            t_attention_mask = micro_batch.batch["teacher_attention_mask"]
            t_position_ids = micro_batch.batch["teacher_position_ids"]
            t_loss_mask = micro_batch.batch["teacher_loss_mask"]

            s_input_ids = micro_batch.batch["student_input_ids"]
            s_attention_mask = micro_batch.batch["student_attention_mask"]
            s_position_ids = micro_batch.batch["student_position_ids"]
            s_loss_mask = micro_batch.batch["student_loss_mask"]

            if valid_row_mask is not None:
                t_input_ids = t_input_ids[valid_row_mask]
                t_attention_mask = t_attention_mask[valid_row_mask]
                t_position_ids = t_position_ids[valid_row_mask]
                t_loss_mask = t_loss_mask[valid_row_mask]
                s_input_ids = s_input_ids[valid_row_mask]
                s_attention_mask = s_attention_mask[valid_row_mask]
                s_position_ids = s_position_ids[valid_row_mask]
                s_loss_mask = s_loss_mask[valid_row_mask]

            if use_remove_padding:
                teacher_fwd = self._forward_logits_unpadded
                student_fwd = self._forward_logits_targets_unpadded
            else:
                teacher_fwd = self._forward_logits_padded
                student_fwd = self._forward_logits_targets_padded

            with torch.no_grad():
                # Teacher forward -> full logits at response positions
                teacher_logits = teacher_fwd(
                    self.ref_module_fsdp, t_input_ids, t_attention_mask, t_position_ids, t_loss_mask
                )

                # Student forward (behavior policy) -> logits + target IDs
                old_student_logits, target_ids = student_fwd(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
                )

                if teacher_logits.shape[0] != old_student_logits.shape[0]:
                    raise RuntimeError(
                        "Teacher and student response token counts diverged. "
                        "This usually means prompt truncation dropped response tokens."
                    )

                n_resp = teacher_logits.shape[0]
                if n_resp == 0:
                    mb_teacher_data.append(None)
                    continue

                # Extract teacher info from full logits
                teacher_logits_f = teacher_logits.float()
                teacher_log_softmax = F.log_softmax(teacher_logits_f, dim=-1)
                teacher_probs = teacher_log_softmax.exp()

                # 1. Teacher log-prob of sampled token
                teacher_log_probs = teacher_log_softmax.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # 2. Teacher entropy
                teacher_entropy = -(teacher_probs * teacher_log_softmax).sum(dim=-1)

                # 3. Top-k tokens and renormalized probs
                topk_probs_raw, topk_indices = teacher_probs.topk(topk, dim=-1)
                topk_probs = topk_probs_raw / topk_probs_raw.sum(dim=-1, keepdim=True)

                del teacher_logits_f, teacher_log_softmax, teacher_probs, topk_probs_raw, teacher_logits

                # 4. Old student log-prob of sampled token
                old_log_softmax = F.log_softmax(old_student_logits.float(), dim=-1)
                old_log_probs = old_log_softmax.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                # 5. Student entropy (for diagnostics)
                student_entropy_sum = entropy_from_logits_with_chunking(
                    old_student_logits.float(), chunk_size=chunk_size
                ).sum().item()

                del old_log_softmax, old_student_logits

                # 6. Entropy mask
                entropy_mask = teacher_entropy > entropy_threshold

            # Store on CPU to free GPU for training phase
            mb_teacher_data.append({
                "teacher_log_probs": teacher_log_probs.cpu(),
                "old_log_probs": old_log_probs.cpu(),
                "teacher_topk_indices": topk_indices.cpu(),
                "teacher_topk_probs": topk_probs.cpu(),
                "entropy_mask": entropy_mask.cpu(),
                "target_ids": target_ids.cpu(),
                "n_tokens": n_resp,
                "student_entropy_sum": student_entropy_sum,
            })

        # Count active micro-batches
        active_mbs = [i for i, d in enumerate(mb_teacher_data) if d is not None]
        if not active_mbs:
            return {
                "eopd/total_loss": 0.0, "eopd/rkl_loss": 0.0, "eopd/fkl_loss": 0.0,
                "eopd/entropy": 0.0, "eopd/grad_norm": 0.0, "eopd/num_tokens": 0,
                "eopd/batch_size": int(batch_size), "eopd/valid_rows": 0,
            }

        grad_accum = len(active_mbs)
        total_tokens = sum(mb_teacher_data[i]["n_tokens"] for i in active_mbs)
        total_entropy_sum = sum(mb_teacher_data[i]["student_entropy_sum"] for i in active_mbs)

        # ================================================================
        # Phase 2: PPO-style Training (n_ppo_epochs optimizer steps)
        # Re-run student forward each epoch, use stored teacher info.
        # ================================================================
        cumulative_metrics = {
            "eopd/total_loss": 0.0, "eopd/rkl_loss": 0.0, "eopd/fkl_loss": 0.0,
            "eopd/clip_fraction": 0.0, "eopd/high_entropy_fraction": 0.0,
        }
        total_grad_norm = 0.0
        lr_stepped = False

        for ppo_epoch in range(n_ppo_epochs):
            self.actor_optimizer.zero_grad()

            for mb_idx in active_mbs:
                mb = micro_batches[mb_idx].to(device)
                td = mb_teacher_data[mb_idx]

                valid_row_mask = mb.batch.get("valid_row_mask")
                s_input_ids = mb.batch["student_input_ids"]
                s_attention_mask = mb.batch["student_attention_mask"]
                s_position_ids = mb.batch["student_position_ids"]
                s_loss_mask = mb.batch["student_loss_mask"]

                if valid_row_mask is not None:
                    vmask = valid_row_mask.bool()
                    s_input_ids = s_input_ids[vmask]
                    s_attention_mask = s_attention_mask[vmask]
                    s_position_ids = s_position_ids[vmask]
                    s_loss_mask = s_loss_mask[vmask]

                if use_remove_padding:
                    student_fwd = self._forward_logits_targets_unpadded
                else:
                    student_fwd = self._forward_logits_targets_padded

                # Student forward (with grad) -> current logits + target IDs
                student_logits, target_ids = student_fwd(
                    self.actor_module_fsdp, s_input_ids, s_attention_mask, s_position_ids, s_loss_mask
                )

                # Current student log-prob of sampled token
                current_log_probs = F.log_softmax(student_logits.float(), dim=-1).gather(
                    -1, target_ids.unsqueeze(-1)
                ).squeeze(-1)

                # Move stored teacher data to GPU
                t_log_probs = td["teacher_log_probs"].to(device)
                old_lps = td["old_log_probs"].to(device)
                t_topk_idx = td["teacher_topk_indices"].to(device)
                t_topk_probs = td["teacher_topk_probs"].to(device)
                e_mask = td["entropy_mask"].to(device)

                loss, epoch_metrics = compute_eopd_loss(
                    teacher_log_probs=t_log_probs,
                    old_log_probs=old_lps,
                    current_log_probs=current_log_probs,
                    teacher_topk_indices=t_topk_idx,
                    teacher_topk_probs=t_topk_probs,
                    student_logits=student_logits,
                    entropy_mask=e_mask,
                    clip_epsilon=clip_epsilon,
                    chunk_size=chunk_size,
                )

                scaled_loss = loss / grad_accum
                scaled_loss.backward()

                # Accumulate metrics
                for k, v in epoch_metrics.items():
                    if k in cumulative_metrics:
                        cumulative_metrics[k] += v

                del student_logits, current_log_probs, t_log_probs, old_lps, t_topk_idx, t_topk_probs, e_mask

            # Gradient clipping and optimizer step
            grad_clip = self.config.actor.get("grad_clip", 1.0)
            if isinstance(self.actor_module_fsdp, FSDP):
                grad_norm = self.actor_module_fsdp.clip_grad_norm_(max_norm=grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module_fsdp.parameters(), max_norm=grad_clip
                )

            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()

            if torch.isfinite(grad_norm):
                self.actor_optimizer.step()
                self.actor_lr_scheduler.step()
                lr_stepped = True
            else:
                logger.warning("Non-finite grad_norm (%.4f) at PPO epoch %d, skipping step", grad_norm.item(), ppo_epoch)
                self.actor_optimizer.zero_grad()

            total_grad_norm += grad_norm.detach().item()

        if not lr_stepped:
            cumulative_metrics["eopd/skipped_all_steps"] = 1.0

        # Average metrics over PPO epochs * micro-batches
        n_updates = n_ppo_epochs * grad_accum
        return {
            "eopd/total_loss": cumulative_metrics["eopd/total_loss"] / max(1, n_updates),
            "eopd/rkl_loss": cumulative_metrics["eopd/rkl_loss"] / max(1, n_updates),
            "eopd/fkl_loss": cumulative_metrics["eopd/fkl_loss"] / max(1, n_updates),
            "eopd/clip_fraction": cumulative_metrics["eopd/clip_fraction"] / max(1, n_updates),
            "eopd/high_entropy_fraction": cumulative_metrics["eopd/high_entropy_fraction"] / max(1, n_updates),
            "eopd/entropy": total_entropy_sum / max(1, total_tokens),
            "eopd/grad_norm": total_grad_norm / max(1, n_ppo_epochs),
            "eopd/num_tokens": int(total_tokens),
            "eopd/batch_size": int(batch_size),
            "eopd/valid_rows": sum(mb_teacher_data[i]["n_tokens"] > 0 for i in active_mbs),
            "eopd/n_ppo_epochs": n_ppo_epochs,
        }

    # ------------------------------------------------------------------
    # Forward pass helpers (from OPSD/RLSD patterns)
    # ------------------------------------------------------------------

    def _forward_logits_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (padded path)."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False,
            )
            logits = outputs.logits
            del outputs

        shift_logits = logits[:, :-1, :]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices]

    def _forward_logits_targets_padded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits AND target IDs (padded path)."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False,
            )
            logits = outputs.logits
            del outputs

        shift_logits = logits[:, :-1, :]
        shift_targets = input_ids[:, 1:]
        shift_loss_mask = loss_mask[:, 1:]
        del logits

        B, S, V = shift_logits.shape
        flat_logits = shift_logits.reshape(B * S, V)
        flat_targets = shift_targets.reshape(B * S)
        flat_mask = shift_loss_mask.reshape(B * S)
        del shift_logits

        response_indices = flat_mask.nonzero(as_tuple=True)[0]
        return flat_logits[response_indices], flat_targets[response_indices]

    def _forward_logits_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits (unpadded/flash_attn path)."""
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad, attention_mask=None,
                position_ids=position_ids_rmpad, use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]
        return logits_rmpad[response_indices]

    def _forward_logits_targets_unpadded(self, model, input_ids, attention_mask, position_ids, loss_mask):
        """Forward pass returning response-position logits AND target IDs (unpadded path)."""
        input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)

        input_ids_flat_rmpad = input_ids.reshape(-1)[indices]

        use_ulysses = self.ulysses_sequence_parallel_size > 1
        if use_ulysses:
            input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                input_ids_rmpad, position_ids_rmpad=position_ids_rmpad,
                sp_size=self.ulysses_sequence_parallel_size,
            )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids_rmpad, attention_mask=None,
                position_ids=position_ids_rmpad, use_cache=False,
            )
            logits_rmpad = outputs.logits.squeeze(0)
            del outputs

        if use_ulysses:
            logits_rmpad = gather_outputs_and_unpad(
                logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size,
            )

        loss_mask_flat = loss_mask.reshape(-1)
        loss_mask_rmpad = loss_mask_flat[indices]

        shifted_loss_mask = _shift_loss_mask_right_per_sequence(loss_mask_rmpad, cu_seqlens)
        response_indices = shifted_loss_mask.nonzero(as_tuple=True)[0]

        targets_rmpad = _shift_targets_per_sequence(input_ids_flat_rmpad, cu_seqlens)

        return logits_rmpad[response_indices], targets_rmpad[response_indices]
