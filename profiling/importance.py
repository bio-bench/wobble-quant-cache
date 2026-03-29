"""Per-dimension importance ranking for tier assignment.

Ranks dimensions by importance for each (layer, head) pair. High-importance
dimensions are "first codon positions" (most bits), low-importance are
"wobble positions" (fewest bits).

Tier boundaries are determined by gap analysis on sorted importance scores,
finding natural breakpoints rather than arbitrary percentile cutoffs.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_importance_scores(
    variance: np.ndarray,
    kurtosis: np.ndarray,
    kurtosis_weight: float = 0.1,
) -> np.ndarray:
    """Compute per-dimension importance scores.

    importance = variance * (1 + kurtosis_weight * max(0, kurtosis))

    Heavy-tailed dimensions (high kurtosis) carry more information.

    Args:
        variance: Shape [n_layers, n_kv_heads, head_dim]. Non-negative.
        kurtosis: Shape [n_layers, n_kv_heads, head_dim]. Excess kurtosis.
        kurtosis_weight: Weight for kurtosis contribution. Default 0.1.

    Returns:
        importance: Shape [n_layers, n_kv_heads, head_dim]. Non-negative.
    """
    if variance.shape != kurtosis.shape:
        raise ValueError(f"Shape mismatch: variance {variance.shape} != kurtosis {kurtosis.shape}")
    if variance.ndim != 3:
        raise ValueError(f"Expected 3D arrays, got {variance.ndim}D")

    kurtosis_boost = np.maximum(0.0, kurtosis)
    importance = variance * (1.0 + kurtosis_weight * kurtosis_boost)
    return importance


def find_tier_boundaries(
    importance: np.ndarray,
    n_tiers: int = 3,
) -> np.ndarray:
    """Find natural breakpoints in sorted importance scores via gap analysis.

    Args:
        importance: Shape [head_dim] for one (layer, head) pair.
        n_tiers: Number of tiers (default 3).

    Returns:
        tier_assignments: Shape [head_dim], values in {1, 2, ..., n_tiers}.
            Tier 1 = highest importance, Tier n_tiers = lowest (wobble).
    """
    if importance.ndim != 1:
        raise ValueError(f"Expected 1D array, got {importance.ndim}D")

    head_dim = importance.shape[0]
    if n_tiers == 1:
        return np.ones(head_dim, dtype=np.int32)

    sorted_indices = np.argsort(-importance)
    sorted_values = importance[sorted_indices]
    gaps = sorted_values[:-1] - sorted_values[1:]

    n_boundaries = n_tiers - 1
    if n_boundaries >= len(gaps):
        boundary_positions = np.arange(len(gaps))
    else:
        top_gap_indices = np.argpartition(-gaps, n_boundaries)[:n_boundaries]
        boundary_positions = np.sort(top_gap_indices)

    tier_in_sorted = np.empty(head_dim, dtype=np.int32)
    prev = 0
    for tier_idx, bp in enumerate(boundary_positions):
        # Unavoidable loop: iterates exactly (n_tiers - 1) times, typically 2.
        tier_in_sorted[prev : bp + 1] = tier_idx + 1
        prev = bp + 1
    tier_in_sorted[prev:] = n_tiers

    tier_assignments = np.empty(head_dim, dtype=np.int32)
    tier_assignments[sorted_indices] = tier_in_sorted

    return tier_assignments


def rank_all_dimensions(
    stats: dict[str, np.ndarray],
    n_tiers: int = 3,
    kurtosis_weight: float = 0.1,
) -> np.ndarray:
    """Compute tier assignments for all (layer, head) pairs.

    Args:
        stats: Must contain k_variance, k_kurtosis, v_variance, v_kurtosis,
            each of shape [n_layers, n_kv_heads, head_dim].
        n_tiers: Number of tiers.
        kurtosis_weight: Weight for kurtosis contribution.

    Returns:
        tier_assignments: Shape [n_layers, n_kv_heads, head_dim].
    """
    k_importance = compute_importance_scores(
        stats["k_variance"], stats["k_kurtosis"], kurtosis_weight=kurtosis_weight)
    v_importance = compute_importance_scores(
        stats["v_variance"], stats["v_kurtosis"], kurtosis_weight=kurtosis_weight)
    combined = (k_importance + v_importance) / 2.0

    n_layers, n_kv_heads, head_dim = combined.shape
    flat = combined.reshape(-1, head_dim)

    flat_tiers = np.empty_like(flat, dtype=np.int32)
    for i in range(flat.shape[0]):
        flat_tiers[i] = find_tier_boundaries(flat[i], n_tiers=n_tiers)

    return flat_tiers.reshape(n_layers, n_kv_heads, head_dim)
