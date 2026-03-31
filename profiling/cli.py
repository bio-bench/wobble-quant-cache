#!/usr/bin/env python3
"""CLI entry point for wobble-profile.

Profile any HuggingFace causal LM's KV cache in one command:

    wobble-profile --model meta-llama/Llama-3-8B
    wobble-profile --model mistralai/Mistral-7B-v0.3 --output results/mistral

Outputs a variance histogram, go/no-go assessment, and profiling JSON.
"""

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from profiling.capture import profile_kv_cache
from profiling.heads import (
    check_head_diversity,
    compute_js_divergence_matrix,
    group_heads,
)
from profiling.importance import compute_importance_scores, rank_all_dimensions
from profiling.report import assess_wobble, generate_report, plot_dimension_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
)
logger = logging.getLogger("wobble-profile")


def _detect_architecture(config) -> tuple[int, int, int]:
    """Extract (n_layers, n_kv_heads, head_dim) from a HuggingFace model config.

    Raises ValueError with a clear message if any parameter cannot be determined.
    """
    # Number of layers
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        raise ValueError(
            f"Cannot determine num_hidden_layers from {type(config).__name__}. "
            "Pass --n-layers explicitly."
        )

    # Number of KV heads (GQA-aware)
    n_kv_heads = getattr(config, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(config, "num_attention_heads", None)
    if n_kv_heads is None:
        raise ValueError(
            f"Cannot determine num_key_value_heads from {type(config).__name__}. "
            "Pass --n-kv-heads explicitly."
        )

    # Head dimension
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(config, "hidden_size", None)
        n_attn_heads = getattr(config, "num_attention_heads", None)
        if hidden_size is not None and n_attn_heads is not None:
            head_dim = hidden_size // n_attn_heads
        else:
            raise ValueError(
                f"Cannot determine head_dim from {type(config).__name__}. "
                "Pass --head-dim explicitly."
            )

    return int(n_layers), int(n_kv_heads), int(head_dim)


def _load_calibration_texts(tokenizer, n_texts, max_seq_length):
    """Load calibration texts from WikiText-103 train split."""
    from datasets import load_dataset

    logger.info("Loading calibration texts from WikiText-103 train...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = []
    for sample in ds:
        text = sample["text"]
        if not text or len(text.strip()) < 100:
            continue
        enc = tokenizer(text, truncation=True, max_length=max_seq_length,
                        return_tensors="pt")
        if enc["input_ids"].size(1) >= 64:
            texts.append(text)
        if len(texts) >= n_texts:
            break
    logger.info("Collected %d calibration texts", len(texts))
    return texts


def _aggregate_head_diversity(stats, n_layers):
    """Compute aggregate head diversity across all layers."""
    all_median_js = []
    for li in range(n_layers):
        js_matrix = compute_js_divergence_matrix(stats, layer_idx=li)
        diversity = check_head_diversity(js_matrix)
        all_median_js.append(diversity["median_js"])

    median_js = float(np.median(all_median_js))
    return {
        "median_js": median_js,
        "heads_are_diverse": median_js > 0.1,
        "heads_are_identical": median_js < 0.01,
        "per_layer_median_js": all_median_js,
    }


def main():
    parser = argparse.ArgumentParser(
        prog="wobble-profile",
        description="Profile a model's KV cache for adaptive quantization.",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path (e.g. meta-llama/Llama-3-8B)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: wobble_profile_<model_name>)",
    )
    parser.add_argument(
        "--n-texts", type=int, default=50,
        help="Number of calibration texts (default: 50)",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Maximum sequence length per text (default: 2048)",
    )
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    # Manual architecture overrides
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override: number of transformer layers")
    parser.add_argument("--n-kv-heads", type=int, default=None,
                        help="Override: number of KV attention heads")
    parser.add_argument("--head-dim", type=int, default=None,
                        help="Override: head dimension")

    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")

    # Resolve output directory
    if args.output is None:
        model_slug = args.model.replace("/", "_").replace("\\", "_")
        output_dir = Path(f"wobble_profile_{model_slug}")
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect architecture
    logger.info("Loading config for %s...", args.model)
    config = AutoConfig.from_pretrained(args.model, token=hf_token or None)
    auto_n_layers, auto_n_kv_heads, auto_head_dim = _detect_architecture(config)

    n_layers = args.n_layers if args.n_layers is not None else auto_n_layers
    n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else auto_n_kv_heads
    head_dim = args.head_dim if args.head_dim is not None else auto_head_dim

    logger.info("Architecture: %d layers, %d KV heads, %d head_dim", n_layers, n_kv_heads, head_dim)

    # Load model
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    logger.info("Loading model %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=hf_token or None,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load calibration texts
    cal_texts = _load_calibration_texts(tokenizer, args.n_texts, args.max_seq_length)

    # Profile KV cache
    t0 = time.time()
    stats, reservoir = profile_kv_cache(
        model, tokenizer, cal_texts,
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        max_seq_length=args.max_seq_length,
    )
    profile_time = time.time() - t0
    logger.info("Profiling completed in %.1f seconds", profile_time)

    # Free model memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Head diversity analysis
    logger.info("Computing head diversity...")
    head_diversity = _aggregate_head_diversity(stats, n_layers)

    # Generate report
    logger.info("Generating report...")
    report = generate_report(stats, head_diversity, output_dir)

    # Compute importance and plot variance histogram
    logger.info("Computing dimension importance and generating plots...")
    importance = compute_importance_scores(
        stats["k_variance"], stats["k_kurtosis"],
    )
    tiers = rank_all_dimensions(stats)
    plot_dimension_importance(importance, tiers, output_dir / "variance_histogram.png")

    # Variance ratio plot (the key "wobble" visualization)
    _plot_variance_ratios(stats, output_dir / "variance_ratios.png")

    # Print summary
    wobble = assess_wobble(stats)
    print()
    print("=" * 60)
    print(f"WOBBLE PROFILE: {args.model}")
    print("=" * 60)
    print(f"  Architecture:     {n_layers} layers, {n_kv_heads} KV heads, {head_dim} head_dim")
    print(f"  Calibration:      {len(cal_texts)} texts, {args.max_seq_length} max tokens")
    print(f"  Profile time:     {profile_time:.0f}s")
    print()
    print(f"  Variance ratio:   {wobble['median_ratio']:.1f}:1 median "
          f"(max {wobble['max_ratio']:.0f}:1)")
    print(f"  Head JS div:      {head_diversity['median_js']:.4f} nats median")
    print()
    print(f"  Wobble positions: {'YES' if wobble['wobble_exists'] else 'NO'}"
          f" (threshold: >{2.0}:1)")
    print(f"  Head diversity:   {'YES' if head_diversity['heads_are_diverse'] else 'NO'}"
          f" (threshold: >{0.1} nats)")
    print()
    print(f"  Decision:         {report['reason']}")
    print()
    print(f"  Outputs saved to: {output_dir.resolve()}")
    print(f"    - profiling_report.json")
    print(f"    - variance_histogram.png")
    print(f"    - variance_ratios.png")
    print("=" * 60)


def _plot_variance_ratios(stats, output_path):
    """Plot per-layer variance ratio (max/min across dimensions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combined_var = 0.5 * (stats["k_variance"] + stats["v_variance"])
    n_layers = combined_var.shape[0]

    dim_max = np.max(combined_var, axis=-1)
    dim_min = np.min(combined_var, axis=-1)
    safe_min = np.where(dim_min > 0, dim_min, 1e-30)
    ratios = dim_max / safe_min

    mean_ratio_per_layer = np.mean(ratios, axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(n_layers), mean_ratio_per_layer, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Mean Variance Ratio (max/min)")
    ax1.set_title("Variance Ratio by Layer")
    ax1.set_yscale("log")

    all_ratios = ratios.ravel()
    ax2.hist(all_ratios, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Variance Ratio (max/min)")
    ax2.set_ylabel("Count (layer x head)")
    ax2.set_title("Distribution of Variance Ratios")
    ax2.set_xscale("log")

    fig.suptitle("KV Cache Dimension Variance Ratios")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved variance ratios plot to %s", output_path)


if __name__ == "__main__":
    main()
