"""Distribution fitting for per-head KV value distributions.

Fits to Gaussian, Laplacian, and Student-t distributions and computes
effective rank via SVD (Roy & Vetterli, 2007).
"""

import logging

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

_DISTRIBUTION_FAMILIES = {
    "gaussian": sp_stats.norm,
    "laplacian": sp_stats.laplace,
    "student_t": sp_stats.t,
}


def fit_distribution(samples: np.ndarray) -> dict:
    """Fit KV value samples to known distribution families.

    Tests Gaussian, Laplacian, and Student-t. Selects best fit
    by lowest mean KS statistic across dimensions.

    Args:
        samples: Shape [n_samples, head_dim].

    Returns:
        Dict with best_fit, ks_stats, params.
    """
    if samples.size == 0:
        raise ValueError("Cannot fit distributions to empty array")
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {samples.shape}")

    n_samples, head_dim = samples.shape
    ks_stats: dict[str, float] = {}
    params: dict[str, dict] = {}

    for family_name, dist in _DISTRIBUTION_FAMILIES.items():
        dim_ks = np.empty(head_dim, dtype=np.float64)
        dim_params_list: list[tuple] = []

        for d in range(head_dim):
            # Per-dimension MLE fitting (inherently sequential)
            col = samples[:, d]
            fitted_params = dist.fit(col)
            ks_stat, _ = sp_stats.kstest(col, dist.cdf, args=fitted_params)
            dim_ks[d] = ks_stat
            dim_params_list.append(fitted_params)

        ks_stats[family_name] = float(np.mean(dim_ks))
        param_array = np.array(dim_params_list, dtype=np.float64)
        avg_params = tuple(float(v) for v in np.mean(param_array, axis=0))

        if family_name in ("gaussian", "laplacian"):
            params[family_name] = {"loc": avg_params[0], "scale": avg_params[1]}
        elif family_name == "student_t":
            params[family_name] = {"df": avg_params[0], "loc": avg_params[1],
                                   "scale": avg_params[2]}

    best_fit = min(ks_stats, key=ks_stats.get)
    return {"best_fit": best_fit, "ks_stats": ks_stats, "params": params}


def compute_effective_rank(kv_vectors: np.ndarray) -> float:
    """Compute effective rank via SVD (Roy & Vetterli, 2007).

    effective_rank = exp(-sum(p_i * log(p_i)))
    where p_i = sigma_i / sum(sigma).

    Args:
        kv_vectors: Shape [n_samples, head_dim].

    Returns:
        Effective rank (1.0 = rank-1, head_dim = full rank).
    """
    if kv_vectors.size == 0:
        raise ValueError("Cannot compute effective rank of empty array")

    centered = kv_vectors - np.mean(kv_vectors, axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    sv_sum = np.sum(singular_values)
    if sv_sum == 0.0:
        return 0.0

    p = singular_values / sv_sum
    nonzero = p[p > 0.0]
    entropy = -np.sum(nonzero * np.log(nonzero))
    return float(np.exp(entropy))
