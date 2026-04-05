"""
Chunk-wise memory-efficient divergence losses.

All loss functions process tokens in chunks to avoid OOM from materializing
full (N, V) float32 probability tensors. With V=152K (Qwen3), a single
(N, V) float32 tensor at N=4096 would be ~2.3 GB.

Supports three divergence types:
  - reverse_kl: KL(p_student || p_teacher) -- mode-seeking
  - forward_kl: KL(p_teacher || p_student) -- mean-seeking
  - jsd: JSD_beta(p_teacher || p_student) -- interpolation of both
"""

import math

import torch
import torch.nn.functional as F


def compute_reverse_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise reverse KL: KL(p_student || p_teacher).

    Mode-seeking: student concentrates mass on teacher's high-probability
    tokens, adopting the teacher's concise reasoning style.

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk to bound peak memory.

    Returns:
        (loss, n_tokens) -- scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        kl_chunk = F.kl_div(t_lp, s_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp, s_lp

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_forward_kl_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise forward KL: KL(p_teacher || p_student).

    Mean-seeking: student spreads probability to cover all modes of the
    teacher distribution, avoiding zero probability on teacher-likely tokens.

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        chunk_size: Tokens per chunk.

    Returns:
        (loss, n_tokens) -- scalar mean loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    kl_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        kl_chunk = F.kl_div(s_lp, t_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp, s_lp

        kl_sum = kl_sum + kl_chunk.sum()
        del kl_chunk

    return kl_sum / n_tokens, n_tokens


def compute_jsd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    beta: float = 0.5,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, int]:
    """Chunk-wise Jensen-Shannon divergence with logsumexp mixture.

    JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1-beta) * KL(p_S || m)
    where m = beta * p_T + (1-beta) * p_S

    Args:
        teacher_logits: (N, V) from frozen teacher (no grad).
        student_logits: (N, V) from trainable student (with grad).
        beta: Interpolation weight (0.5 = symmetric JSD).
        chunk_size: Tokens per chunk.

    Returns:
        (loss, n_tokens) -- scalar mean JSD loss and token count.
    """
    n_tokens = teacher_logits.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True), 0

    log_beta = math.log(beta) if beta > 0 else float("-inf")
    log_1m_beta = math.log(1.0 - beta) if beta < 1 else float("-inf")

    teacher_chunks = [c.clone() for c in teacher_logits.split(chunk_size, dim=0)]
    del teacher_logits

    jsd_sum = torch.tensor(0.0, device=student_logits.device)

    for i, t_chunk in enumerate(teacher_chunks):
        start = i * chunk_size
        end = start + t_chunk.shape[0]

        t_lp = F.log_softmax(t_chunk.float(), dim=-1)
        s_lp = F.log_softmax(student_logits[start:end].float(), dim=-1)
        del t_chunk
        teacher_chunks[i] = None

        log_m = torch.logsumexp(
            torch.stack([t_lp + log_beta, s_lp + log_1m_beta], dim=0),
            dim=0,
        )

        kl_t = F.kl_div(log_m, t_lp, reduction="none", log_target=True).sum(dim=-1)
        del t_lp
        kl_s = F.kl_div(log_m, s_lp, reduction="none", log_target=True).sum(dim=-1)
        del s_lp, log_m

        jsd_sum = jsd_sum + (beta * kl_t + (1.0 - beta) * kl_s).sum()
        del kl_t, kl_s

    return jsd_sum / n_tokens, n_tokens


DIVERGENCE_FN_MAP = {
    "reverse_kl": compute_reverse_kl_loss,
    "forward_kl": compute_forward_kl_loss,
    "jsd": compute_jsd_loss,
}
