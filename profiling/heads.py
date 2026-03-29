"""Cross-head distribution analysis and head grouping via JS divergence.

Computes pairwise Jensen-Shannon divergence between KV heads to determine
which heads need separate quantization configs. Groups similar heads
via agglomerative clustering.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


def _kl_divergence_gaussians(mu_p, var_p, mu_q, var_q):
    """KL(P || Q) for univariate Gaussians, vectorized over dimensions."""
    eps = 1e-30
    var_p = np.maximum(var_p, eps)
    var_q = np.maximum(var_q, eps)
    return (
        0.5 * np.log(var_q / var_p)
        + (var_p + (mu_p - mu_q) ** 2) / (2.0 * var_q)
        - 0.5
    )


def _js_divergence_gaussians(mu_p, var_p, mu_q, var_q):
    """Jensen-Shannon divergence between two Gaussians (averaged over dims)."""
    mu_m = 0.5 * (mu_p + mu_q)
    var_m = 0.5 * (var_p + var_q) + 0.25 * (mu_p - mu_q) ** 2

    kl_p_m = _kl_divergence_gaussians(mu_p, var_p, mu_m, var_m)
    kl_q_m = _kl_divergence_gaussians(mu_q, var_q, mu_m, var_m)

    jsd_per_dim = 0.5 * kl_p_m + 0.5 * kl_q_m
    return float(np.mean(jsd_per_dim))


def compute_js_divergence_matrix(
    stats: dict[str, np.ndarray],
    layer_idx: int,
) -> np.ndarray:
    """Compute pairwise Jensen-Shannon divergence between KV heads.

    Args:
        stats: Must contain k_mean, k_variance, v_mean, v_variance,
            each of shape [n_layers, n_kv_heads, head_dim].
        layer_idx: Which layer to analyze.

    Returns:
        js_matrix: Shape [n_kv_heads, n_kv_heads], symmetric, in nats.
    """
    k_mean_l = stats["k_mean"][layer_idx]
    k_var_l = stats["k_variance"][layer_idx]
    v_mean_l = stats["v_mean"][layer_idx]
    v_var_l = stats["v_variance"][layer_idx]

    n_kv_heads = k_mean_l.shape[0]
    js_matrix = np.zeros((n_kv_heads, n_kv_heads), dtype=np.float64)

    # n_kv_heads is small (typically 4-8), so pairwise iteration is fine.
    for i in range(n_kv_heads):
        for j in range(i + 1, n_kv_heads):
            jsd_k = _js_divergence_gaussians(
                k_mean_l[i], k_var_l[i], k_mean_l[j], k_var_l[j])
            jsd_v = _js_divergence_gaussians(
                v_mean_l[i], v_var_l[i], v_mean_l[j], v_var_l[j])
            jsd = 0.5 * (jsd_k + jsd_v)
            js_matrix[i, j] = jsd
            js_matrix[j, i] = jsd

    return js_matrix


def check_head_diversity(
    js_matrix: np.ndarray,
    go_threshold: float = 0.1,
    nogo_threshold: float = 0.01,
) -> dict[str, float | bool]:
    """Evaluate go/no-go criteria for head diversity.

    Args:
        js_matrix: Pairwise JS divergence, shape [n_kv_heads, n_kv_heads].
        go_threshold: JS divergence above which heads are diverse.
        nogo_threshold: JS divergence below which heads are identical.

    Returns:
        Dict with median_js, heads_are_diverse, heads_are_identical.
    """
    n = js_matrix.shape[0]
    upper_tri = js_matrix[np.triu_indices(n, k=1)]
    median_js = float(np.median(upper_tri))

    return {
        "median_js": median_js,
        "heads_are_diverse": median_js > go_threshold,
        "heads_are_identical": median_js < nogo_threshold,
    }


def group_heads(
    js_matrix: np.ndarray,
    max_groups: int,
    divergence_threshold: float,
) -> dict[int, list[int]]:
    """Agglomerative clustering of heads by distribution similarity.

    Heads within the same group share a quantization config.

    Args:
        js_matrix: Pairwise JS divergence, shape [n_kv_heads, n_kv_heads].
        max_groups: Maximum number of groups.
        divergence_threshold: JS divergence threshold for merging.

    Returns:
        {group_id: [head_indices]}, 0-indexed.
    """
    n = js_matrix.shape[0]
    if n == 1:
        return {0: [0]}

    condensed = squareform(js_matrix, checks=True)
    Z = linkage(condensed, method="average")
    labels_by_threshold = fcluster(Z, t=divergence_threshold, criterion="distance")

    if len(np.unique(labels_by_threshold)) > max_groups:
        labels = fcluster(Z, t=max_groups, criterion="maxclust")
    else:
        labels = labels_by_threshold

    groups: dict[int, list[int]] = defaultdict(list)
    for head_idx, label in enumerate(labels):
        groups[int(label) - 1].append(head_idx)

    groups = {gid: sorted(heads) for gid, heads in sorted(groups.items())}
    logger.info("Grouped %d heads into %d groups: %s", n, len(groups), groups)
    return groups
