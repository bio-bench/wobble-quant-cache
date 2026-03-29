"""Profiling report generation with go/no-go assessment.

If both wobble positions AND head diversity are absent, the adaptive
quantization approach is falsified and the pipeline should stop.
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def assess_wobble(
    stats: dict[str, np.ndarray],
    go_threshold: float = 10.0,
    nogo_threshold: float = 2.0,
) -> dict[str, float | bool]:
    """Assess whether wobble positions exist (dimension importance varies).

    Computes the max/min variance ratio across dimensions.

    Args:
        stats: Must contain k_variance and v_variance.
        go_threshold: Variance ratio above which signal is strong.
        nogo_threshold: Variance ratio below which wobble is unlikely.

    Returns:
        Dict with median_ratio, wobble_exists, wobble_strong.
    """
    combined_var = 0.5 * (stats["k_variance"] + stats["v_variance"])

    dim_max = np.max(combined_var, axis=-1)
    dim_min = np.min(combined_var, axis=-1)
    safe_min = np.where(dim_min > 0, dim_min, 1e-30)
    ratios = dim_max / safe_min

    median_ratio = float(np.median(ratios))
    return {
        "median_ratio": median_ratio,
        "mean_ratio": float(np.mean(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
        "wobble_exists": median_ratio > nogo_threshold,
        "wobble_strong": median_ratio > go_threshold,
    }


def generate_report(
    stats: dict[str, np.ndarray],
    head_diversity: dict[str, float | bool],
    output_dir: Path,
    go_threshold: float = 10.0,
    nogo_threshold: float = 2.0,
    js_go: float = 0.1,
    js_nogo: float = 0.01,
) -> dict[str, bool | str]:
    """Generate profiling report with go/no-go assessment.

    Args:
        stats: Raw profiling statistics from StatsAccumulator.finalize().
        head_diversity: Output from check_head_diversity().
        output_dir: Where to save report files.
        go_threshold: Variance ratio go threshold.
        nogo_threshold: Variance ratio no-go threshold.
        js_go: JS divergence go threshold.
        js_nogo: JS divergence no-go threshold.

    Returns:
        Dict with go_decision, reason, wobble_exists, heads_diverse.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wobble = assess_wobble(stats, go_threshold, nogo_threshold)
    wobble_exists = wobble["wobble_exists"]
    heads_diverse = bool(head_diversity.get("heads_are_diverse", False))
    median_js = float(head_diversity.get("median_js", 0.0))

    if wobble_exists and heads_diverse:
        go = True
        reason = (
            f"PROCEED: Wobble positions ({wobble['median_ratio']:.1f}:1) "
            f"and head diversity ({median_js:.4f} nats) confirmed."
        )
    elif wobble_exists:
        go = True
        reason = (
            f"PROCEED WITH GLOBAL CONFIG: Wobble positions exist "
            f"({wobble['median_ratio']:.1f}:1) but heads are similar."
        )
    elif heads_diverse:
        go = True
        reason = (
            f"PROCEED WITH HEAD-ADAPTIVE ONLY: No wobble positions "
            f"but heads are diverse ({median_js:.4f} nats)."
        )
    else:
        go = False
        reason = (
            f"STOP: Neither wobble positions ({wobble['median_ratio']:.1f}:1) "
            f"nor head diversity ({median_js:.4f} nats) detected."
        )

    logger.info("GO/NO-GO: %s", reason)

    report = {
        "go_decision": go,
        "reason": reason,
        "wobble_exists": wobble_exists,
        "heads_diverse": heads_diverse,
        "wobble_details": wobble,
        "head_diversity_details": head_diversity,
    }

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(output_dir / "profiling_report.json", "w") as f:
        json.dump(json.loads(json.dumps(report, default=_convert)), f, indent=2)

    return {"go_decision": go, "reason": reason,
            "wobble_exists": wobble_exists, "heads_diverse": heads_diverse}


def plot_dimension_importance(
    importance: np.ndarray,
    tier_assignments: np.ndarray,
    output_path: Path,
) -> None:
    """Plot sorted dimension importance with tier coloring.

    Args:
        importance: Shape [n_layers, n_kv_heads, head_dim].
        tier_assignments: Shape [n_layers, n_kv_heads, head_dim].
        output_path: Where to save the plot.
    """
    n_layers = importance.shape[0]
    sample_pairs = [
        (0, 0), (n_layers // 3, 0), (2 * n_layers // 3, 0), (n_layers - 1, 0),
    ]
    tier_colors = {1: "red", 2: "orange", 3: "blue"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (li, hi) in zip(axes.flat, sample_pairs):
        imp = importance[li, hi]
        tiers = tier_assignments[li, hi]
        sorted_idx = np.argsort(-imp)
        for rank, dim_idx in enumerate(sorted_idx):
            color = tier_colors.get(tiers[dim_idx], "gray")
            ax.bar(rank, imp[dim_idx], color=color, alpha=0.7, width=1.0)
        ax.set_title(f"Layer {li}, Head {hi}")
        ax.set_xlabel("Rank (sorted)")
        ax.set_ylabel("Importance")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=f"Tier {t}") for t, c in tier_colors.items()]
    axes[0, 0].legend(handles=legend, loc="upper right", fontsize=8)

    fig.suptitle("Sorted Dimension Importance with Tier Assignments")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved importance plot to %s", output_path)
