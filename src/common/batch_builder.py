"""
Build paired teacher/student tokenized batches for self-distillation training.

The input is a standard verl rollout batch:
  - prompt messages come from ``non_tensor_batch["raw_prompt"]``
  - privileged context comes from ``non_tensor_batch["reward_model"]["ground_truth"]``
  - the sampled continuation comes from ``batch["responses"]``

Teacher and student always consume the exact same sampled response token IDs.
Only the prompt context differs.

Optionally carries per-sample tensors (e.g. reward advantages) through the
filtering process so they stay aligned with the surviving samples.
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
    """Build a padded sequence, truncating the response if it exceeds max_length."""
    response_ids_list = response_ids.tolist()
    if not response_ids_list:
        return None

    # Truncate response to leave room for at least 1 prompt token
    if len(response_ids_list) >= max_length:
        response_ids_list = response_ids_list[: max_length - 1]

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
        raise KeyError("Batch construction requires response_mask or attention_mask + responses")

    response_length = batch.batch["responses"].shape[1]
    return batch.batch["attention_mask"][:, -response_length:]


# DAPO-style instruction suffix
_INSTRUCTION_SUFFIX = (
    'The last line of your response should be of the form '
    'Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
    'Remember to put your answer on its own line after "Answer:".'
)


def _build_teacher_messages(
    student_messages: list[dict],
    ground_truth: str,
    teacher_system_prompt: Optional[str],
) -> list[dict]:
    """Build teacher prompt messages with privileged ground truth context."""
    if teacher_system_prompt:
        return [
            {"role": "system", "content": f"{teacher_system_prompt}\n\nCorrect answer: {ground_truth}"},
            *student_messages,
        ]

    student_content = ""
    for msg in reversed(student_messages):
        if msg.get("role") == "user":
            student_content = msg["content"]
            break

    teacher_content = (
        f"{student_content}\n\n"
        f"The ground truth answer to this problem is:\n"
        f"Answer: {ground_truth}\n\n"
        f"After reading the ground truth answer above, make sure you truly understand "
        f"the reasoning behind each step. Now, using your own words and independent "
        f"reasoning, derive the same final answer to the problem above. "
        f"Think step by step."
    )

    teacher_messages = []
    user_replaced = False
    for msg in student_messages:
        if msg.get("role") == "user" and not user_replaced:
            teacher_messages.append({"role": "user", "content": teacher_content})
            user_replaced = True
        else:
            teacher_messages.append(dict(msg))

    if not user_replaced:
        teacher_messages.append({"role": "user", "content": teacher_content})

    return teacher_messages


def build_teacher_student_batch(
    batch: DataProto,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 16384,
    teacher_system_prompt: Optional[str] = None,
    apply_chat_template_kwargs: Optional[dict] = None,
    per_sample_data: Optional[dict[str, torch.Tensor]] = None,
    use_teacher_context: bool = True,
) -> Optional[DataProto]:
    """Build paired teacher/student sequences from a standard verl rollout batch.

    When use_teacher_context=True (same model): teacher gets ground truth context,
    student gets question only. Separate teacher/student sequences.

    When use_teacher_context=False (bigger teacher model): both teacher and student
    see the exact same input_ids.

    Args:
        batch: Standard verl rollout batch with prompts, responses, raw_prompt, reward_model.
        tokenizer: Tokenizer for encoding prompts.
        max_length: Max total sequence length (prompt + response).
        teacher_system_prompt: Optional system prompt for teacher (only when use_teacher_context=True).
        apply_chat_template_kwargs: Kwargs for apply_chat_template (same for both).
        per_sample_data: Optional dict of {name: tensor[N]} to carry through filtering.
        use_teacher_context: If True, teacher sees ground truth. If False, same prompt for both.
    """
    if len(batch) == 0:
        return None
    if "raw_prompt" not in batch.non_tensor_batch:
        raise KeyError("Requires data.return_raw_chat=True so raw_prompt is available")
    if "responses" not in batch.batch:
        raise KeyError("Requires rollout responses in the batch")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    response_mask = _get_response_mask(batch)
    chat_kwargs = dict(apply_chat_template_kwargs or {})
    student_seqs = []
    teacher_seqs = []
    kept_indices = []
    skipped = 0

    raw_prompts = batch.non_tensor_batch["raw_prompt"]
    reward_models = batch.non_tensor_batch.get("reward_model", [None] * len(raw_prompts))
    responses = batch.batch["responses"]

    for i, (raw_prompt, reward_model, response_ids, sample_response_mask) in enumerate(
        zip(raw_prompts, reward_models, responses, response_mask, strict=True)
    ):
        student_messages = list(raw_prompt)
        valid_response_ids = response_ids[sample_response_mask.bool()]

        if len(student_messages) == 0 or valid_response_ids.numel() == 0:
            skipped += 1
            continue

        # Tokenize student prompt
        student_prompt_ids = tokenizer.apply_chat_template(
            student_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
        )
        s_seq = _build_sequence_from_token_ids(student_prompt_ids, valid_response_ids, max_length, pad_token_id)
        if s_seq is None:
            skipped += 1
            continue

        if use_teacher_context:
            gt = reward_model.get("ground_truth") if reward_model else None
            if gt is None:
                skipped += 1
                continue
            teacher_messages = _build_teacher_messages(student_messages, gt, teacher_system_prompt)
            teacher_prompt_ids = tokenizer.apply_chat_template(
                teacher_messages, add_generation_prompt=True, tokenize=True, **chat_kwargs,
            )
            t_seq = _build_sequence_from_token_ids(teacher_prompt_ids, valid_response_ids, max_length, pad_token_id)
            if t_seq is None:
                skipped += 1
                continue
        else:
            # Teacher model mode: teacher sees exact same input as student
            t_seq = s_seq

        student_seqs.append(s_seq)
        teacher_seqs.append(t_seq)
        kept_indices.append(i)

    if skipped:
        logger.warning("Skipped %d samples during batch construction", skipped)

    if not teacher_seqs:
        return None

    batch_dict = {}
    for prefix, seqs in [("teacher_", teacher_seqs), ("student_", student_seqs)]:
        for key in ["input_ids", "attention_mask", "position_ids", "loss_mask"]:
            batch_dict[f"{prefix}{key}"] = torch.stack([s[key] for s in seqs])
    batch_dict["valid_row_mask"] = torch.ones(len(student_seqs), dtype=torch.bool)

    if per_sample_data and kept_indices:
        kept = torch.tensor(kept_indices, dtype=torch.long)
        for name, tensor in per_sample_data.items():
            batch_dict[name] = tensor[kept]

    return DataProto.from_single_dict(batch_dict)
