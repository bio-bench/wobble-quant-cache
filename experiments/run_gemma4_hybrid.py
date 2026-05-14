#!/usr/bin/env python3
"""Block-Hybrid KV Cache Quantization on Gemma 4 E4B.

Validates the hypothesis: block-hybrid (FWHT-rotated 4-bit VIPs + 2-bit
wobbles) achieves near-lossless perplexity on Gemma 4 E4B at ~2.25 avg
bits per dimension.

Pipeline:
    1. Load Gemma 4 E4B in BF16 (~16 GB VRAM)
    2. Profile KV cache variance from calibration texts
    3. Select VIP dimensions per layer (top by variance, power-of-2 count)
    4. Evaluate FP16 baseline perplexity (WikiText-103)
    5. Evaluate Block-Hybrid perplexity
    6. Evaluate KIVI-2bit baseline for comparison
    7. Report results

Usage:
    pip install -e .
    HF_TOKEN=... python -u experiments/run_gemma4_hybrid.py

Requires ~16 GB VRAM (Gemma-4-E4B in BF16) + headroom for KV cache.
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from profiling.capture import extract_kv_from_output
from wobble.evaluate import evaluate_perplexity
from wobble.hybrid import build_hybrid_configs_from_variance
from wobble.patch import (
    patch_gemma4,
    quantize_hybrid_wrapper,
    quantize_kivi_wrapper,
    quantize_turboquant_wrapper,
    quantize_wobble_simple_wrapper,
    restore_model,
)
from wobble.quantize import build_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_gemma4_hybrid")


# -----------------------------------------------------------------------
# Profiling: capture per-dimension variance from calibration texts
# -----------------------------------------------------------------------


def profile_kv_variance(
    model,
    tokenizer,
    texts: list[str],
    max_seq_length: int = 2048,
) -> dict[int, dict]:
    """Profile per-dimension KV variance across calibration texts.

    Returns {layer_idx: {'variance': ndarray[head_dim], 'head_dim': int}}
    for each non-shared layer (shared layers reuse source KV).

    All reductions are vectorized -- no loop over dims, heads, or tokens.
    The only Python loop is over calibration texts (unavoidable: one
    forward pass per text).
    """
    device = next(model.parameters()).device

    # Accumulators: {layer_idx: {'sum_var': Tensor, 'count': int, 'head_dim': int}}
    accum: dict[int, dict] = {}

    n_texts = len(texts)
    t0 = time.monotonic()
    for i, text in enumerate(texts):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)

        kv_pairs = extract_kv_from_output(outputs.past_key_values)

        for li, (k, v) in enumerate(kv_pairs):
            # k, v: [1, n_heads, seq_len, head_dim]
            # Variance across token positions (dim=2), averaged over heads (dim=1)
            k_var = k.squeeze(0).float().var(dim=1).mean(dim=0)  # [head_dim]
            v_var = v.squeeze(0).float().var(dim=1).mean(dim=0)  # [head_dim]
            avg_var = (k_var + v_var) / 2.0

            if li not in accum:
                accum[li] = {
                    "sum_var": torch.zeros_like(avg_var),
                    "count": 0,
                    "head_dim": k.shape[-1],
                }
            accum[li]["sum_var"] += avg_var
            accum[li]["count"] += 1

        del outputs, kv_pairs
        torch.cuda.empty_cache()

        elapsed = time.monotonic() - t0
        if (i + 1) % 5 == 0 or i == 0:
            logger.info(
                "Profiled %d/%d texts (%.1fs elapsed)", i + 1, n_texts, elapsed
            )

    # Average variance across texts
    result = {}
    for li, acc in accum.items():
        result[li] = {
            "variance": (acc["sum_var"] / acc["count"]).cpu().numpy(),
            "head_dim": acc["head_dim"],
        }

    logger.info(
        "Profiling complete: %d texts, %d layers with unique KV",
        n_texts,
        len(result),
    )
    return result


# -----------------------------------------------------------------------
# Calibration data
# -----------------------------------------------------------------------


def get_calibration_texts(tokenizer, n_texts=30, max_seq_length=2048):
    """Load calibration texts from WikiText-103 train split."""
    logger.info("Loading calibration texts from WikiText-103 train...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200]
    result = []
    for t in texts:
        if len(result) >= n_texts:
            break
        enc = tokenizer(
            t, truncation=True, max_length=max_seq_length, return_tensors="pt"
        )
        if enc["input_ids"].size(1) >= 128:
            result.append(t)
    logger.info("Collected %d calibration texts", len(result))
    return result


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Block-Hybrid KV cache quantization on Gemma 4 E4B"
    )
    parser.add_argument(
        "--model", default="google/gemma-4-E4B", help="Model name or path"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50_000, help="Max eval tokens"
    )
    parser.add_argument(
        "--cal-texts", type=int, default=30, help="Number of calibration texts"
    )
    parser.add_argument(
        "--vip-fraction",
        type=float,
        default=0.125,
        help="Fraction of dims as VIP (default 12.5%%)",
    )
    parser.add_argument(
        "--vip-bits", type=int, default=4, help="Bit-width for VIP dims"
    )
    parser.add_argument(
        "--wobble-bits", type=int, default=2, help="Bit-width for Wobble dims"
    )
    parser.add_argument(
        "--group-size", type=int, default=32, help="Per-group scaling window"
    )
    parser.add_argument(
        "--output", default="results/gemma4_e4b_hybrid.json"
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", args.model)
    t0 = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_time = time.monotonic() - t0
    logger.info("Model loaded in %.1fs", load_time)

    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / 1e9
        logger.info("GPU memory after model load: %.2f GB", alloc_gb)

    # ------------------------------------------------------------------
    # Step 2: Profile KV cache variance
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PROFILING KV CACHE")
    logger.info("=" * 60)

    cal_texts = get_calibration_texts(tokenizer, n_texts=args.cal_texts)
    layer_variances = profile_kv_variance(model, tokenizer, cal_texts)

    # Log variance ratio (VIP signal strength)
    for li in sorted(layer_variances.keys()):
        var = layer_variances[li]["variance"]
        ratio = var.max() / (var.min() + 1e-12)
        hd = layer_variances[li]["head_dim"]
        if li < 3 or li % 6 == 0:
            logger.info(
                "  Layer %2d (dim=%d): var ratio = %.1f:1", li, hd, ratio
            )

    del cal_texts
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: BF16 baseline (run once, reuse)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BF16 BASELINE")
    logger.info("=" * 60)

    results = {}
    t0 = time.monotonic()
    fp16_result = evaluate_perplexity(
        model, tokenizer, max_tokens=args.max_tokens
    )
    results["BF16"] = fp16_result["perplexity"]
    logger.info(
        "BF16 perplexity: %.4f (%.1fs)",
        results["BF16"],
        time.monotonic() - t0,
    )

    # ------------------------------------------------------------------
    # Helper: build configs, patch, evaluate, restore
    # ------------------------------------------------------------------
    def _compute_avg_bits(configs):
        total_b, total_d = 0, 0
        for li, cfg in configs.items():
            hd = layer_variances[li]["head_dim"]
            nv = len(cfg.vip_indices)
            total_b += nv * cfg.vip_bits + (hd - nv) * cfg.wobble_bits
            total_d += hd
        return total_b / total_d if total_d > 0 else 0

    def run_hybrid_variant(label, configs):
        avg_bits = _compute_avg_bits(configs)
        tag = f"{label} ({avg_bits:.2f}bit)"
        logger.info("=" * 60)
        logger.info("HYBRID: %s", tag)
        logger.info("=" * 60)

        q_fn = partial(quantize_hybrid_wrapper, hybrid_configs=configs)
        orig = patch_gemma4(model, q_fn)
        t0 = time.monotonic()
        r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
        results[tag] = r["perplexity"]
        logger.info(
            "%s perplexity: %.4f (%.1fs)", tag, r["perplexity"],
            time.monotonic() - t0,
        )
        restore_model(orig)
        return avg_bits

    # ------------------------------------------------------------------
    # Step 4: Sweep hybrid configurations
    # ------------------------------------------------------------------
    sweep_meta = []

    # 4a. Fixed 25% VIP fraction
    logger.info("=" * 60)
    logger.info("BUILDING CONFIGS: Fixed 25%% VIP")
    logger.info("=" * 60)
    cfgs_25 = build_hybrid_configs_from_variance(
        layer_variances, vip_fraction=0.25,
        vip_bits=args.vip_bits, wobble_bits=args.wobble_bits,
        group_size=args.group_size, device=device,
    )
    ab = run_hybrid_variant("Fixed-25%", cfgs_25)
    sweep_meta.append({"label": "Fixed-25%", "avg_bits": ab})

    # 4b. Adaptive 50% energy
    logger.info("=" * 60)
    logger.info("BUILDING CONFIGS: Adaptive 50%% energy")
    logger.info("=" * 60)
    cfgs_a50 = build_hybrid_configs_from_variance(
        layer_variances, adaptive_energy=0.5,
        vip_bits=args.vip_bits, wobble_bits=args.wobble_bits,
        group_size=args.group_size, device=device,
    )
    ab = run_hybrid_variant("Adaptive-50%energy", cfgs_a50)
    sweep_meta.append({"label": "Adaptive-50%energy", "avg_bits": ab})

    # 4c. Adaptive 70% energy
    logger.info("=" * 60)
    logger.info("BUILDING CONFIGS: Adaptive 70%% energy")
    logger.info("=" * 60)
    cfgs_a70 = build_hybrid_configs_from_variance(
        layer_variances, adaptive_energy=0.7,
        vip_bits=args.vip_bits, wobble_bits=args.wobble_bits,
        group_size=args.group_size, device=device,
    )
    ab = run_hybrid_variant("Adaptive-70%energy", cfgs_a70)
    sweep_meta.append({"label": "Adaptive-70%energy", "avg_bits": ab})

    # 4d. Fixed 50% VIP fraction (half dims are VIP)
    logger.info("=" * 60)
    logger.info("BUILDING CONFIGS: Fixed 50%% VIP")
    logger.info("=" * 60)
    cfgs_50 = build_hybrid_configs_from_variance(
        layer_variances, vip_fraction=0.5,
        vip_bits=args.vip_bits, wobble_bits=args.wobble_bits,
        group_size=args.group_size, device=device,
    )
    ab = run_hybrid_variant("Fixed-50%", cfgs_50)
    sweep_meta.append({"label": "Fixed-50%", "avg_bits": ab})

    # ------------------------------------------------------------------
    # Step 5: TurboQuant (FWHT rotate all dims, uniform quantize)
    # ------------------------------------------------------------------
    for tq_bits in [2, 3]:
        tag = f"TurboQuant-{tq_bits}bit"
        logger.info("=" * 60)
        logger.info("%s", tag)
        logger.info("=" * 60)

        q_fn_tq = partial(
            quantize_turboquant_wrapper,
            n_bits=tq_bits,
            group_size=args.group_size,
        )
        orig = patch_gemma4(model, q_fn_tq)
        t0 = time.monotonic()
        r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
        results[tag] = r["perplexity"]
        logger.info(
            "%s perplexity: %.4f (%.1fs)", tag, r["perplexity"],
            time.monotonic() - t0,
        )
        restore_model(orig)

    # ------------------------------------------------------------------
    # Step 6: Wobble adaptive scalar (variable bits per dim, no rotation)
    # ------------------------------------------------------------------
    for target_bits in [2, 3]:
        tag = f"Wobble-{target_bits}bit"
        logger.info("=" * 60)
        logger.info("%s", tag)
        logger.info("=" * 60)

        # Build AdaptiveScalarConfig per layer from profiled variance
        wobble_configs = {}
        total_budget = int(target_bits * 256)  # default head_dim
        for li, stats in layer_variances.items():
            hd = stats["head_dim"]
            var = stats["variance"]
            budget = int(target_bits * hd)
            # Use variance as proxy; build_config needs min/max/mean
            # Approximate from variance: mean=0, min/max = +/- 3*std
            std = np.sqrt(var)
            dim_min = -3.0 * std
            dim_max = 3.0 * std
            dim_mean = np.zeros_like(var)
            wobble_configs[li] = build_config(
                dim_variances=var,
                dim_means=dim_mean.astype(np.float32),
                dim_mins=dim_min.astype(np.float32),
                dim_maxs=dim_max.astype(np.float32),
                total_budget=budget,
                head_group_id=0,
                layer_idx=li,
                min_bits=1,
            )
            if li < 2:
                logger.info("  %s", wobble_configs[li].summary())

        q_fn_w = partial(
            quantize_wobble_simple_wrapper,
            wobble_configs=wobble_configs,
            group_size=args.group_size,
        )
        orig = patch_gemma4(model, q_fn_w)
        t0 = time.monotonic()
        r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
        results[tag] = r["perplexity"]
        logger.info(
            "%s perplexity: %.4f (%.1fs)", tag, r["perplexity"],
            time.monotonic() - t0,
        )
        restore_model(orig)

    # ------------------------------------------------------------------
    # Step 7: KIVI baselines
    # ------------------------------------------------------------------
    for kivi_bits in [2, 3]:
        tag = f"KIVI-{kivi_bits}bit"
        logger.info("=" * 60)
        logger.info("%s BASELINE", tag)
        logger.info("=" * 60)

        q_fn_kivi = partial(quantize_kivi_wrapper, n_bits=kivi_bits)
        orig = patch_gemma4(model, q_fn_kivi)
        t0 = time.monotonic()
        r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
        results[tag] = r["perplexity"]
        logger.info(
            "%s perplexity: %.4f (%.1fs)", tag, r["perplexity"],
            time.monotonic() - t0,
        )
        restore_model(orig)

    # ------------------------------------------------------------------
    # Step 6: Results table
    # ------------------------------------------------------------------
    bf16_ppl = results["BF16"]

    print("\n" + "=" * 70, flush=True)
    print(f"GEMMA 4 E4B BLOCK-HYBRID SWEEP ({args.model})", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Method':<35} {'PPL':>10} {'vs BF16':>10}", flush=True)
    print("-" * 60, flush=True)
    for method, ppl in results.items():
        delta = ((ppl - bf16_ppl) / bf16_ppl) * 100
        print(f"{method:<35} {ppl:>10.2f} {delta:>+9.2f}%", flush=True)
    print("=" * 70, flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "model": args.model,
                "vip_bits": args.vip_bits,
                "wobble_bits": args.wobble_bits,
                "group_size": args.group_size,
                "max_tokens": args.max_tokens,
                "sweep": sweep_meta,
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
