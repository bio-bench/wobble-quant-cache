#!/usr/bin/env python3
"""Reproduce Wobble quantization results on Mistral-7B.

Expected results:
    FP16:          5.38 PPL
    Wobble-3bit:   6.06 PPL (+12.6%)
    Wobble-2bit:  42.83 PPL
    KIVI-3bit:     6.35 PPL (+18%)
    KIVI-2bit:   198.09 PPL

Usage:
    pip install -e .
    python experiments/reproduce_mistral.py [--model mistralai/Mistral-7B-Instruct-v0.3]

Requires ~14GB VRAM (Mistral-7B in FP16).
Set HF_TOKEN environment variable for gated model access.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from profiling.capture import profile_kv_cache
from profiling.heads import compute_js_divergence_matrix, group_heads
from wobble.evaluate import evaluate_perplexity
from wobble.patch import (
    patch_mistral, restore_model, quantize_wobble, quantize_kivi_wrapper,
)
from wobble.quantize import build_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("reproduce_mistral")

# Mistral-7B architecture
N_LAYERS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
GROUP_SIZE = 32


def get_calibration_texts(tokenizer, n_texts=80, max_seq_length=2048):
    """Load calibration texts from WikiText-103 train."""
    from datasets import load_dataset
    logger.info("Loading calibration texts...")
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 100]
    result = []
    for t in texts:
        if len(result) >= n_texts:
            break
        enc = tokenizer(t, truncation=True, max_length=max_seq_length, return_tensors="pt")
        if enc["input_ids"].size(1) >= 64:
            result.append(t)
    logger.info("Collected %d calibration texts", len(result))
    return result


def build_wobble_configs(dim_var, dim_min, dim_max, dim_mean, js_data,
                         target_bits, min_bits=0):
    """Build adaptive scalar configs for all layers."""
    total_budget = int(target_bits * HEAD_DIM)
    all_configs, all_h2g = {}, {}

    for li in range(N_LAYERS):
        js_mat = js_data[f"layer_{li}"]
        head_groups = group_heads(js_mat, min(8, N_KV_HEADS), 0.15)
        layer_configs, layer_h2g = {}, {}

        for gid, heads in head_groups.items():
            h0 = heads[0]
            cfg = build_config(
                dim_variances=dim_var[li, h0], dim_means=dim_mean[li, h0],
                dim_mins=dim_min[li, h0], dim_maxs=dim_max[li, h0],
                total_budget=total_budget, head_group_id=gid,
                layer_idx=li, min_bits=min_bits)
            layer_configs[gid] = cfg
            for h in heads:
                layer_h2g[h] = gid

        all_configs[li] = layer_configs
        all_h2g[li] = layer_h2g

    return all_configs, all_h2g


def main():
    parser = argparse.ArgumentParser(description="Reproduce Wobble results on Mistral-7B")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--output", default="results/mistral_reproduced.json")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")

    logger.info("Loading model: %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto", token=hf_token)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Profiling — uses numerically stable Chan et al. streaming stats
    cal_texts = get_calibration_texts(tokenizer)
    stats, _ = profile_kv_cache(model, tokenizer, cal_texts,
                                N_LAYERS, N_KV_HEADS, HEAD_DIM)

    # Average K/V statistics for build_config (applies same config to both)
    dim_var = (stats["k_variance"] + stats["v_variance"]) / 2.0
    dim_mean = ((stats["k_mean"] + stats["v_mean"]) / 2.0).astype(np.float32)
    dim_min = np.minimum(stats["k_min"], stats["v_min"]).astype(np.float32)
    dim_max = np.maximum(stats["k_max"], stats["v_max"]).astype(np.float32)

    # Analytical JS divergence between heads (Gaussian-parametric)
    js_data = {}
    for li in range(N_LAYERS):
        js_data[f"layer_{li}"] = compute_js_divergence_matrix(stats, li)

    del cal_texts
    gc.collect()

    results = {}

    # FP16 baseline
    logger.info("=" * 50)
    fp16 = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
    results["FP16"] = fp16["perplexity"]

    # 3-bit
    logger.info("=" * 50)
    configs_3, h2g_3 = build_wobble_configs(dim_var, dim_min, dim_max, dim_mean,
                                            js_data, target_bits=3.0, min_bits=1)
    q_fn = partial(quantize_wobble, all_configs=configs_3, all_h2g=h2g_3,
                   group_size=GROUP_SIZE)
    orig = patch_mistral(model, q_fn)
    r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
    results["Wobble-3bit"] = r["perplexity"]
    restore_model(orig)

    q_fn = partial(quantize_kivi_wrapper, n_bits=3)
    orig = patch_mistral(model, q_fn)
    r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
    results["KIVI-3bit"] = r["perplexity"]
    restore_model(orig)

    # 2-bit
    logger.info("=" * 50)
    configs_2, h2g_2 = build_wobble_configs(dim_var, dim_min, dim_max, dim_mean,
                                            js_data, target_bits=2.0, min_bits=1)
    q_fn = partial(quantize_wobble, all_configs=configs_2, all_h2g=h2g_2,
                   group_size=GROUP_SIZE)
    orig = patch_mistral(model, q_fn)
    r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
    results["Wobble-2bit"] = r["perplexity"]
    restore_model(orig)

    q_fn = partial(quantize_kivi_wrapper, n_bits=2)
    orig = patch_mistral(model, q_fn)
    r = evaluate_perplexity(model, tokenizer, max_tokens=args.max_tokens)
    results["KIVI-2bit"] = r["perplexity"]
    restore_model(orig)

    # Results
    fp16_ppl = results["FP16"]
    print("\n" + "=" * 60)
    print(f"MISTRAL-7B RESULTS ({args.model})")
    print("=" * 60)
    print(f"{'Method':<20} {'PPL':>10} {'vs FP16':>10}")
    print("-" * 45)
    for method in ["FP16", "Wobble-3bit", "KIVI-3bit", "Wobble-2bit", "KIVI-2bit"]:
        if method in results:
            delta = ((results[method] - fp16_ppl) / fp16_ppl) * 100
            print(f"{method:<20} {results[method]:>10.2f} {delta:>+9.1f}%")
    print("=" * 60)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
