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

from profiling.capture import extract_kv_from_output
from profiling.heads import group_heads
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


def profile_post_rope(model, tokenizer, cal_texts):
    """Collect per-dimension statistics from post-RoPE KV values."""
    logger.info("Profiling post-RoPE KV cache...")
    device = next(model.parameters()).device
    shape = (N_LAYERS, N_KV_HEADS, HEAD_DIM)

    dim_min = np.full(shape, np.inf, dtype=np.float32)
    dim_max = np.full(shape, -np.inf, dtype=np.float32)
    dim_sum = np.zeros(shape, dtype=np.float64)
    dim_sum_sq = np.zeros(shape, dtype=np.float64)
    dim_count = np.zeros((N_LAYERS, N_KV_HEADS), dtype=np.int64)

    for i, text in enumerate(cal_texts):
        if (i + 1) % 20 == 0:
            logger.info("Profiling: text %d / %d", i + 1, len(cal_texts))
        inputs = tokenizer(text, return_tensors="pt", max_length=2048,
                           truncation=True, padding=False).to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, return_dict=True)

        kv_pairs = extract_kv_from_output(out.past_key_values)
        for li, (k, v) in enumerate(kv_pairs):
            for kv_t in [k, v]:
                vals = kv_t.squeeze(0).float().cpu().numpy()
                for h in range(N_KV_HEADS):
                    dim_min[li, h] = np.minimum(dim_min[li, h], vals[h].min(axis=0))
                    dim_max[li, h] = np.maximum(dim_max[li, h], vals[h].max(axis=0))
                    dim_sum[li, h] += vals[h].sum(axis=0)
                    dim_sum_sq[li, h] += (vals[h] ** 2).sum(axis=0)
                    dim_count[li, h] += vals[h].shape[0]
        del out, kv_pairs

    safe_count = np.maximum(dim_count[:, :, np.newaxis], 1)
    dim_mean = (dim_sum / safe_count).astype(np.float32)
    dim_var = ((dim_sum_sq / safe_count) - dim_mean ** 2).astype(np.float32)
    dim_var = np.maximum(dim_var, 1e-10)

    logger.info("Profiling complete. Variance range: [%.4f, %.4f]",
                dim_var.min(), dim_var.max())
    logger.info("Variance ratio (max/min per head): median=%.1f",
                np.median(dim_var.max(axis=-1) / np.maximum(dim_var.min(axis=-1), 1e-10)))
    return dim_min, dim_max, dim_mean, dim_var


def compute_js_matrices(model, tokenizer, cal_texts):
    """Compute JS divergence between KV heads for grouping."""
    logger.info("Computing JS divergence matrices...")
    device = next(model.parameters()).device
    n_bins = 100
    hist_range = (-5.0, 5.0)
    histograms = np.zeros((N_LAYERS, N_KV_HEADS, n_bins), dtype=np.float64)

    for i, text in enumerate(cal_texts[:30]):
        if (i + 1) % 10 == 0:
            logger.info("JS matrices: text %d / 30", i + 1)
        inputs = tokenizer(text, return_tensors="pt", max_length=2048,
                           truncation=True, padding=False).to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True, return_dict=True)
        kv_pairs = extract_kv_from_output(out.past_key_values)
        for li, (k, v) in enumerate(kv_pairs):
            for kv_t in [k, v]:
                vals = kv_t.squeeze(0).float().cpu().numpy()
                for h in range(N_KV_HEADS):
                    hist, _ = np.histogram(vals[h].ravel(), bins=n_bins,
                                           range=hist_range, density=False)
                    histograms[li, h] += hist
        del out, kv_pairs

    histograms = histograms + 1e-10
    hist_probs = histograms / histograms.sum(axis=-1, keepdims=True)

    js_data = {}
    for li in range(N_LAYERS):
        js_mat = np.zeros((N_KV_HEADS, N_KV_HEADS), dtype=np.float32)
        for h1 in range(N_KV_HEADS):
            for h2 in range(h1 + 1, N_KV_HEADS):
                m = 0.5 * (hist_probs[li, h1] + hist_probs[li, h2])
                kl1 = np.sum(hist_probs[li, h1] * np.log(hist_probs[li, h1] / m))
                kl2 = np.sum(hist_probs[li, h2] * np.log(hist_probs[li, h2] / m))
                js = 0.5 * (kl1 + kl2)
                js_mat[h1, h2] = js
                js_mat[h2, h1] = js
        js_data[f"layer_{li}"] = js_mat
    return js_data


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

    # Profiling
    cal_texts = get_calibration_texts(tokenizer)
    dim_min, dim_max, dim_mean, dim_var = profile_post_rope(model, tokenizer, cal_texts)
    js_data = compute_js_matrices(model, tokenizer, cal_texts)
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
                                            js_data, target_bits=3.0)
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
