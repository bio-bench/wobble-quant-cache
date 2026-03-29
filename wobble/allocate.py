"""Reverse water-filling bit budget allocation.

Given a target average bit-width, allocate bits across tiers proportional
to their information content. Uses rate-distortion theory: dimensions with
higher variance receive more bits.

For Gaussian sources, the optimal rate is:
    R_i = max(0, 0.5 * log2(variance_i / lambda))
where lambda is the water level found by binary search.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_LAMBDA_SEARCH_MAX_ITER = 200
_LAMBDA_SEARCH_RTOL = 1e-12


def optimize_bit_allocation(
    tier_dims: dict[int, int],
    tier_variances: dict[int, float],
    total_budget: int,
) -> dict[int, int]:
    """Reverse water-filling: allocate bits proportional to log(variance).

    Binary searches for the water level lambda that satisfies the budget
    constraint. The continuous optimum is discretized to integer total bits
    per tier using the largest-remainder method.

    Args:
        tier_dims: {tier: n_dimensions}. E.g., {1: 40, 2: 50, 3: 38}.
        tier_variances: {tier: avg_variance}. From profiling.
        total_budget: Total bits per vector.

    Returns:
        bits_per_tier: {tier: total_bits_for_this_tier}.
    """
    if not tier_dims:
        raise ValueError("tier_dims is empty")
    if not tier_variances:
        raise ValueError("tier_variances is empty")
    if set(tier_dims.keys()) != set(tier_variances.keys()):
        raise ValueError(
            f"Key mismatch: tier_dims keys {set(tier_dims.keys())} "
            f"!= tier_variances keys {set(tier_variances.keys())}"
        )
    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")

    tiers = sorted(tier_dims.keys())
    dims = np.array([tier_dims[t] for t in tiers], dtype=np.int64)
    variances = np.array([tier_variances[t] for t in tiers], dtype=np.float64)

    if np.any(dims <= 0):
        raise ValueError(f"All tier dimensions must be positive")
    if np.any(variances <= 0):
        raise ValueError(f"All tier variances must be positive")

    if total_budget == 0:
        return {t: 0 for t in tiers}

    total_dims = int(np.sum(dims))
    logger.info(
        "Bit allocation: %d tiers, %d total dims, %d total budget (%.2f avg bits/dim)",
        len(tiers), total_dims, total_budget, total_budget / total_dims,
    )

    # Step 1: continuous optimal rates via binary search on lambda
    continuous_rates = _solve_continuous_waterfilling(variances, dims, total_budget)

    # Step 2: discretize to integer total bits per tier
    continuous_bits = continuous_rates * dims.astype(np.float64)
    floored_bits = np.maximum(np.floor(continuous_bits).astype(np.int64), 0)
    remaining = total_budget - int(np.sum(floored_bits))

    if remaining > 0:
        # Largest-remainder method: give extra bits to tiers with largest fractional parts
        fractional_parts = continuous_bits - floored_bits.astype(np.float64)
        order = np.argsort(-fractional_parts)
        for idx in order:
            if remaining <= 0:
                break
            floored_bits[idx] += 1
            remaining -= 1
    elif remaining < 0:
        fractional_parts = continuous_bits - floored_bits.astype(np.float64)
        order = np.argsort(fractional_parts)
        for idx in order:
            if remaining >= 0:
                break
            if floored_bits[idx] > 0:
                floored_bits[idx] -= 1
                remaining += 1

    final_total = int(np.sum(floored_bits))
    if final_total != total_budget:
        raise ValueError(
            f"Bit allocation failed: achieved {final_total} bits vs budget {total_budget}"
        )

    bits_per_tier = {t: int(floored_bits[i]) for i, t in enumerate(tiers)}
    logger.info("Bit allocation result: %s (total=%d)", bits_per_tier, final_total)
    return bits_per_tier


def _solve_continuous_waterfilling(
    variances: np.ndarray,
    dims: np.ndarray,
    total_budget: int,
) -> np.ndarray:
    """Find continuous optimal rates via binary search on lambda."""
    total_dims = int(np.sum(dims))
    max_var = float(np.max(variances))
    lambda_high = max_var * 2.0
    lambda_low = float(np.min(variances)) * 2.0 ** (-2.0 * total_budget / total_dims)
    if lambda_low <= 0:
        lambda_low = 1e-300

    # Expand lower bound until it produces enough bits
    for _ in range(200):
        rates = _continuous_rates_at_lambda(variances, lambda_low)
        bits = float(np.sum(rates * dims))
        if bits >= total_budget:
            break
        lambda_low /= 1024.0
        if lambda_low < 1e-300:
            raise ValueError(f"Cannot find sufficient lower bound for budget {total_budget}")

    # Binary search
    rates = None
    for _ in range(_LAMBDA_SEARCH_MAX_ITER):
        lambda_mid = (lambda_low + lambda_high) / 2.0
        rates = _continuous_rates_at_lambda(variances, lambda_mid)
        bits = float(np.sum(rates * dims))

        if abs(bits - total_budget) < 1e-6:
            return rates
        elif bits > total_budget:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid

        if (lambda_high - lambda_low) / max(lambda_high, 1e-300) < _LAMBDA_SEARCH_RTOL:
            return rates

    return rates


def _continuous_rates_at_lambda(variances: np.ndarray, lam: float) -> np.ndarray:
    """R_i = max(0, 0.5 * log2(var_i / lambda)). Vectorized."""
    ratio = variances / lam
    return np.where(ratio > 1.0, 0.5 * np.log2(ratio), 0.0)
