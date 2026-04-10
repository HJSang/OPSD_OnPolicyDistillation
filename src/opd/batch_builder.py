"""
Build paired teacher/student tokenized batches for on-policy distillation.

Both teacher and student see the exact same input (same prompt + same response).
The teacher is a separate (typically bigger/stronger) model that produces better
distributions naturally. No privileged information (ground truth) is injected.
"""

import logging
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from verl.protocol import DataProto

logger = logging.getLogger(__name__)


def _build_sequence_from_token_ids(
    prompt_ids: list[int],
    response_ids: torch.Tensor,
    max_length: int,
    pad_token_id: int,
) -> Optional[dict]:
    """Build a padded sequence: [prompt | response | padding].

    If the response is longer than max_length, it is truncated to fit.
    The prompt is left-truncated to make room for the response.
    """
    response_ids_list = response_ids.tolist()
    if not response_ids_list:
        return None

    # Truncate response if it alone exceeds max_length (leave room for at least 1 prompt token)
    if len(response_ids_list) >= max_length:
        response_ids_list = response_ids_list[: max_length - 1]
        logger.debug("Truncated response from %d to %d tokens to fit max_length=%d",
                      len(response_ids.tolist()), len(response_ids_list), max_length)

    max_prompt_len = max_length - len(response_ids_list)
    prompt_ids = prompt_ids[-max_prompt_len:]
    full_ids = prompt_ids + response_ids_list
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids_list)

    seq_len = len(full_ids)
    if seq_len < 2:
        return None

    pad_len = max_length - seq_len
    return {
        "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
        "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
        "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
    }


def _get_response_mask(batch: DataProto) -> torch.Tensor:
    if "response_mask" in batch.batch:
        return batch.batch["response_mask"]

    if "attention_mask" not in batch.batch or "responses" not in batch.batch:
        raise KeyError("OPD batch construction requires response_mask or attention_mask + responses")

    response_length = batch.batch["responses"].shape[1]
    return batch.batch["attention_mask"][:, -response_length:]


def _build_opd_batch_from_prompts(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """Build OPD batch from pre-tokenized ``prompts`` + ``response_mask``.

    Matches single-turn and agent-loop rollouts: uses the same prompt tokenization
    the model saw during generation. ``response_mask`` selects LLM vs tool tokens
    for the loss (trainer must set it, e.g. ``compute_response_mask`` for single-turn).
    ``apply_chat_template_kwargs`` is unused (API compatibility).
    """
    if len(batch) == 0:
        return None
    if "prompts" not in batch.batch:
        raise KeyError("OPD prompts path requires batch['prompts']")
    if "responses" not in batch.batch:
        raise KeyError("OPD requires rollout responses in the batch")
    if "response_mask" not in batch.batch:
        raise KeyError("OPD prompts path requires batch['response_mask']")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    response_mask = _get_response_mask(batch)
    response_length = batch.batch["responses"].shape[1]
    response_attention = batch.batch["attention_mask"][:, -response_length:]
    prompt_length = batch.batch["prompts"].shape[1]
    prompt_attention = batch.batch["attention_mask"][:, :prompt_length]

    seqs = []
    kept_indices = []
    skipped = 0

    prompts = batch.batch["prompts"]
    responses = batch.batch["responses"]

    for sample_idx, (prompt_ids_padded, prompt_attn, response_ids, sample_resp_mask, sample_resp_attn) in enumerate(zip(
        prompts, prompt_attention, responses, response_mask, response_attention, strict=True
    )):
        # Extract real (non-padding) prompt token IDs — these include tool schemas
        real_prompt_ids = prompt_ids_padded[prompt_attn.bool()].tolist()

        # Extract real (non-padding) response tokens and their LLM/tool mask
        real_resp_mask = sample_resp_attn.bool()
        valid_response_ids = response_ids[real_resp_mask]
        valid_token_mask = sample_resp_mask[real_resp_mask]  # 1=LLM, 0=tool

        if len(real_prompt_ids) == 0 or valid_response_ids.numel() == 0:
            skipped += 1
            continue

        if valid_token_mask.sum() == 0:
            logger.warning("Skipping sample: no LLM-generated tokens (all tool response)")
            skipped += 1
            continue

        # Build sequence with per-token mask
        response_ids_list = valid_response_ids.tolist()
        mask_list = valid_token_mask.tolist()

        if len(response_ids_list) >= max_length:
            response_ids_list = response_ids_list[: max_length - 1]
            mask_list = mask_list[: max_length - 1]

        max_prompt_len = max_length - len(response_ids_list)
        prompt_ids = real_prompt_ids[-max_prompt_len:]  # left-truncate prompt if needed
        full_ids = prompt_ids + response_ids_list
        loss_mask = [0] * len(prompt_ids) + [int(m) for m in mask_list]

        seq_len = len(full_ids)
        if seq_len < 2:
            skipped += 1
            continue

        pad_len = max_length - seq_len
        seq = {
            "input_ids": torch.tensor(full_ids + [pad_token_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
            "position_ids": torch.tensor(list(range(seq_len)) + [0] * pad_len, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask + [0] * pad_len, dtype=torch.float32),
        }
        seqs.append(seq)
        kept_indices.append(sample_idx)

    if skipped:
        logger.warning("Skipped %d samples during OPD batch construction (prompts path)", skipped)

    if not seqs:
        return None

    # Teacher and student see exact same input — stack once, share tensors
    batch_dict = {}
    for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
        shared = torch.stack([s[key] for s in seqs])
        batch_dict[f"teacher_{key}"] = shared
        batch_dict[f"student_{key}"] = shared
    batch_dict["valid_row_mask"] = torch.ones(len(seqs), dtype=torch.bool)

    if sample_weights is not None:
        batch_dict["sample_weights"] = sample_weights[kept_indices]

    return DataProto.from_single_dict(batch_dict)


def _build_opd_batch_from_raw_prompt(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """Build OPD batch by re-tokenizing ``raw_prompt`` with the chat template.

    Use only when ``batch["prompts"]`` is absent. Prefer :func:`build_opd_batch`
    so rollout inputs match stored prompt IDs when available.
    """
    if len(batch) == 0:
        return None
    if "raw_prompt" not in batch.non_tensor_batch:
        raise KeyError("OPD requires data.return_raw_chat=True so raw_prompt is available")
    if "responses" not in batch.batch:
        raise KeyError("OPD requires rollout responses in the batch")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    response_mask = _get_response_mask(batch)
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    seqs = []
    kept_indices = []
    skipped = 0

    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    responses = batch.batch["responses"]

    for sample_idx, (raw_prompt, response_ids, sample_response_mask) in enumerate(zip(
        raw_prompts, responses, response_mask, strict=True
    )):
        student_messages = list(raw_prompt)
        valid_response_ids = response_ids[sample_response_mask.bool()]

        if len(student_messages) == 0 or valid_response_ids.numel() == 0:
            logger.warning("Skipping sample: empty prompt (%d messages) or empty response (%d tokens)",
                           len(student_messages), valid_response_ids.numel())
            skipped += 1
            continue

        prompt_ids = tokenizer.apply_chat_template(
            student_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
        )
        seq = _build_sequence_from_token_ids(prompt_ids, valid_response_ids, max_length, pad_token_id)
        if seq is None:
            skipped += 1
            continue

        seqs.append(seq)
        kept_indices.append(sample_idx)

    if skipped:
        logger.warning("Skipped %d samples during OPD batch construction (raw_prompt path)", skipped)

    if not seqs:
        return None

    # Teacher and student see the exact same input — stack once, share tensors
    batch_dict = {}
    for key in ("input_ids", "attention_mask", "position_ids", "loss_mask"):
        shared = torch.stack([s[key] for s in seqs])
        batch_dict[f"teacher_{key}"] = shared
        batch_dict[f"student_{key}"] = shared
    batch_dict["valid_row_mask"] = torch.ones(len(seqs), dtype=torch.bool)

    if sample_weights is not None:
        batch_dict["sample_weights"] = sample_weights[kept_indices]

    return DataProto.from_single_dict(batch_dict)


def build_opd_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """Single entry point for OPD batch construction after rollout.

    If ``batch["prompts"]`` is present, uses those token IDs plus ``response_mask``
    (same inputs the model saw during generation). Otherwise re-tokenizes
    ``non_tensor_batch["raw_prompt"]`` — requires ``data.return_raw_chat=True``.

    The trainer should set ``response_mask`` before calling (e.g. via
    ``compute_response_mask`` for single-turn, or preserve the agent loop mask).
    """
    if len(batch) == 0:
        return None
    if "responses" not in batch.batch:
        raise KeyError("OPD requires rollout responses in the batch")

    if "prompts" in batch.batch:
        if "response_mask" not in batch.batch:
            raise KeyError(
                "OPD requires batch['response_mask'] when using pre-tokenized prompts; "
                "set it in the trainer (e.g. compute_response_mask) before build_opd_batch."
            )
        return _build_opd_batch_from_prompts(
            batch=batch,
            tokenizer=tokenizer,
            max_length=max_length,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            sample_weights=sample_weights,
        )

    if "raw_prompt" in batch.non_tensor_batch:
        return _build_opd_batch_from_raw_prompt(
            batch=batch,
            tokenizer=tokenizer,
            max_length=max_length,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            sample_weights=sample_weights,
        )

    raise KeyError(
        "OPD batch construction needs batch['prompts'] (preferred) or "
        "non_tensor_batch['raw_prompt'] with data.return_raw_chat=True."
    )


def build_opd_batch_multiturn(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """Pre-tokenized prompt path only (requires ``prompts`` + ``response_mask``). Same as ``build_opd_batch`` when prompts exist."""
    return _build_opd_batch_from_prompts(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        sample_weights=sample_weights,
    )


def build_opd_batch_from_verl_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    apply_chat_template_kwargs: Optional[dict] = None,
    sample_weights: Optional[torch.Tensor] = None,
) -> Optional[DataProto]:
    """Tokenize from ``raw_prompt`` only; prefer :func:`build_opd_batch` for training."""
    return _build_opd_batch_from_raw_prompt(
        batch=batch,
        tokenizer=tokenizer,
        max_length=max_length,
        apply_chat_template_kwargs=apply_chat_template_kwargs,
        sample_weights=sample_weights,
    )
