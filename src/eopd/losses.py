"""
EOPD loss functions: PPO-style clipped reverse KL + entropy-gated top-k forward KL.

Unlike OPSD's raw full-vocabulary divergence, EOPD uses:
  1. Clipped reverse KL with importance sampling (single-token log-probs, PPO-style)
  2. Forward KL over teacher's top-k tokens, only at high-entropy positions

Reference: "Entropy-Aware On-Policy Distillation of Language Models" (Jin et al., 2026)
  Eq. 8: Clipped reverse KL surrogate
  Eq. 9: Combined EOPD objective
  Eq. 10: Top-k forward KL approximation
"""

import torch
import torch.nn.functional as F


def compute_clipped_rkl_loss(
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, int, float]:
    """PPO-style clipped reverse KL loss (Eq. 7-8).

    Uses importance sampling with the behavior policy pi_old and clipping
    to stabilize training. The "advantage" is the teacher-student log-prob
    difference, not a reward-based advantage.

    Args:
        teacher_log_probs: (N,) log pi_te(x_t | c_t), no grad.
        old_log_probs: (N,) log pi_old(x_t | c_t), no grad.
        current_log_probs: (N,) log pi_theta(x_t | c_t), has grad.
        clip_epsilon: PPO clip range (default 0.2).

    Returns:
        (loss, n_tokens, clip_fraction)
    """
    n_tokens = teacher_log_probs.shape[0]
    if n_tokens == 0:
        return torch.tensor(0.0, device=current_log_probs.device, requires_grad=True), 0, 0.0

    # Advantage: how much the teacher prefers this token over the old student (Eq. 6)
    advantage = (teacher_log_probs - old_log_probs).detach()

    # Importance ratio: pi_theta / pi_old
    log_ratio = current_log_probs - old_log_probs.detach()
    ratio = torch.exp(log_ratio)

    # Clipped PPO surrogate (Eq. 8) — note the negation for minimization
    surr1 = -ratio * advantage
    surr2 = -torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    loss_per_token = torch.max(surr1, surr2)

    # Track clipping fraction for diagnostics
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

    return loss_per_token.mean(), n_tokens, clip_fraction


def compute_topk_fkl_loss(
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    student_logits: torch.Tensor,
    entropy_mask: torch.Tensor,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, int]:
    """Forward KL over teacher's top-k tokens at high-entropy positions (Eq. 10).

    Only computed at positions where entropy_mask is True (teacher entropy > tau).
    The teacher distribution is renormalized over top-k tokens.

    Args:
        teacher_topk_indices: (N, k) int64, teacher's top-k token indices.
        teacher_topk_probs: (N, k) float32, renormalized teacher probs over top-k.
        student_logits: (N, V) with grad, current student logits.
        entropy_mask: (N,) bool, True where teacher entropy > tau.
        chunk_size: Tokens per chunk for memory efficiency.

    Returns:
        (loss_mean_over_masked, n_masked_tokens)
    """
    device = student_logits.device

    masked_indices = entropy_mask.nonzero(as_tuple=True)[0]
    n_masked = masked_indices.shape[0]

    if n_masked == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    fkl_sum = torch.tensor(0.0, device=device)

    for chunk_start in range(0, n_masked, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_masked)
        idx = masked_indices[chunk_start:chunk_end]

        s_logits = student_logits[idx]  # (C, V)
        s_log_probs = F.log_softmax(s_logits.float(), dim=-1)  # (C, V)

        t_indices = teacher_topk_indices[idx]  # (C, k)
        t_probs = teacher_topk_probs[idx].float()  # (C, k)

        # Gather student log-probs at teacher's top-k positions
        s_log_probs_topk = s_log_probs.gather(-1, t_indices)  # (C, k)
        del s_log_probs

        # FKL = sum_k p_te_tilde(x) * [log p_te_tilde(x) - log p_theta(x)]
        t_log_probs = t_probs.clamp(min=1e-10).log()
        fkl_per_token = (t_probs * (t_log_probs - s_log_probs_topk)).sum(dim=-1)  # (C,)
        del t_log_probs, s_log_probs_topk

        fkl_sum = fkl_sum + fkl_per_token.sum()
        del fkl_per_token

    return fkl_sum / n_masked, n_masked


def compute_eopd_loss(
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    student_logits: torch.Tensor,
    entropy_mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Combined EOPD loss (Eq. 9): clipped reverse KL + entropy-gated top-k forward KL.

    L_EOPD = L_OPD (clipped RKL at all positions) + I[H_t > tau] * L_FKL (at high-entropy positions)

    Args:
        teacher_log_probs: (N,) teacher log-prob of sampled token.
        old_log_probs: (N,) behavior policy log-prob of sampled token.
        current_log_probs: (N,) current student log-prob of sampled token (has grad).
        teacher_topk_indices: (N, k) teacher's top-k token indices.
        teacher_topk_probs: (N, k) renormalized teacher probs over top-k.
        student_logits: (N, V) current student logits (has grad).
        entropy_mask: (N,) bool, True where teacher entropy > tau.
        clip_epsilon: PPO clip range.
        chunk_size: Chunk size for top-k FKL computation.

    Returns:
        (total_loss, metrics_dict)
    """
    n_tokens = teacher_log_probs.shape[0]

    rkl_loss, _, clip_frac = compute_clipped_rkl_loss(
        teacher_log_probs, old_log_probs, current_log_probs, clip_epsilon
    )

    fkl_loss_per_masked, n_masked = compute_topk_fkl_loss(
        teacher_topk_indices, teacher_topk_probs, student_logits, entropy_mask, chunk_size
    )

    # Scale FKL to match paper's token-mean normalization (Algorithm 1, line 13):
    # L = (1/N) * Σ_t L_OPD_t + (1/N) * Σ_{t: H>τ} L_FKL_t
    # fkl_loss_per_masked is averaged over n_masked; we need it averaged over N.
    fkl_weight = n_masked / max(1, n_tokens)
    fkl_loss = fkl_loss_per_masked * fkl_weight

    total_loss = rkl_loss + fkl_loss

    metrics = {
        "eopd/rkl_loss": rkl_loss.detach().item(),
        "eopd/fkl_loss": fkl_loss.detach().item(),
        "eopd/fkl_loss_per_masked": fkl_loss_per_masked.detach().item(),
        "eopd/total_loss": total_loss.detach().item(),
        "eopd/clip_fraction": clip_frac,
        "eopd/high_entropy_tokens": n_masked,
        "eopd/high_entropy_fraction": n_masked / max(1, n_tokens),
    }

    return total_loss, metrics
