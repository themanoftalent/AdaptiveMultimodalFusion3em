from __future__ import annotations
import torch


def compute_pseudo_optimal_alpha(
    p_r: torch.Tensor,
    p_h: torch.Tensor,
    labels: torch.Tensor,
    default_alpha: float = 0.25,
    step: float = 0.05,
) -> torch.Tensor:
    """Computes pseudo-optimal alpha targets for gate training.

    Vectorized implementation using PyTorch broadcasting.
    """
    if p_r.shape != p_h.shape:
        raise ValueError("p_r and p_h must have the same shape.")
    if p_r.ndim != 2 or p_r.shape[1] != 3:
        raise ValueError("p_r and p_h must have shape [N, 3].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [N].")

    device = p_r.device
    n = p_r.shape[0]

    # Create candidate alphas tensor: shape (K,)
    candidate_alphas = torch.arange(0.0, 1.0 + step / 2.0, step, device=device)
    k = candidate_alphas.shape[0]

    # Reshape for broadcasting:
    # p_r and p_h: (N, 1, 3)
    # alphas: (1, K, 1)
    alphas = candidate_alphas.view(1, k, 1)
    p_r_expanded = p_r.unsqueeze(1)
    p_h_expanded = p_h.unsqueeze(1)

    # Compute fused probabilities for all candidates: shape (N, K, 3)
    fused_probs = alphas * p_r_expanded + (1.0 - alphas) * p_h_expanded

    # Find predictions for all candidates: shape (N, K)
    preds = torch.argmax(fused_probs, dim=2)

    # Check which predictions are correct: shape (N, K)
    correct = (preds == labels.unsqueeze(1))

    # Compute differences from default_alpha for each candidate: shape (K,)
    diffs = torch.abs(candidate_alphas - default_alpha)

    # Set differences for incorrect candidates to a large value: shape (N, K)
    large_value = torch.tensor(1e9, device=device)
    masked_diffs = torch.where(correct, diffs.unsqueeze(0), large_value)

    # Find candidate index with minimum difference for each sample: shape (N,)
    min_diffs, min_indices = torch.min(masked_diffs, dim=1)

    # Gather the best alpha for each sample: shape (N, 1)
    best_alphas = candidate_alphas[min_indices].unsqueeze(1)

    # If no candidate predicted the target class correctly, fall back to default_alpha
    has_no_correct = (min_diffs >= 1e8).unsqueeze(1)
    best_alphas = torch.where(has_no_correct, torch.tensor(default_alpha, device=device), best_alphas)

    return best_alphas
