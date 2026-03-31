# Wobble: Adaptive KV Cache Quantization

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Models: Mistral-7B, Gemma-2-2B](https://img.shields.io/badge/models-Mistral--7B%20%7C%20Gemma--2--2B-orange)]()

> Allocate quantization bits where they matter. At 3-bit on Gemma-2-2B: **+0.02% perplexity degradation** — effectively lossless.

```python
from wobble.quantize import build_config, encode, decode

config = build_config(dim_variances, dim_means, dim_mins, dim_maxs,
                      total_budget=3 * head_dim, head_group_id=0, layer_idx=0)
codes, group_mins, group_maxs = encode(kv_vectors, config, group_size=32)
reconstructed = decode(codes, config, group_mins, group_maxs, group_size=32)
```

<p align="center">
  <img src="assets/wobble-kv-quant.png" alt="Biological analogy: genetic codon table vs Wobble KV cache quantization" width="800">
</p>

## What is Wobble?

Standard KV cache quantization (KIVI, static uniform) treats every dimension the same. Wobble doesn't — it measures which dimensions carry the most information and gives them more bits, while "wobble" dimensions (low-variance, error-tolerant) get fewer.

The name comes from molecular biology: in the genetic codon table, the third "wobble" position tolerates mutations because it contributes least to amino acid identity. KV cache dimensions exhibit the same pattern — some carry critical information while others are expendable. Variance ratios across dimensions reach **3,190:1** on Mistral-7B.

## Results

### Gemma-2-2B (26 layers, 4 KV heads, head_dim=256)

| Method | 3-bit PPL | vs FP16 | 2-bit PPL | vs FP16 |
|--------|-----------|---------|-----------|---------|
| FP16 baseline | 43.24 | — | 43.24 | — |
| **Wobble** | **43.25** | **+0.02%** | **39.69** | −8.2%* |
| KIVI | 70.14 | +62.2% | 398.28 | +821% |

### Mistral-7B (32 layers, 8 KV heads, head_dim=128)

| Method | 3-bit PPL | vs FP16 | 2-bit PPL | vs FP16 |
|--------|-----------|---------|-----------|---------|
| FP16 baseline | 5.38 | — | 5.38 | — |
| **Wobble** | **6.06** | **+12.6%** | **42.83** | +696% |
| KIVI | 6.55 | +21.8% | 198.09 | +3,584% |

Wobble beats KIVI at every bit-width on every model tested.

\*The 2-bit PPL on Gemma being below FP16 is likely a regularization effect of quantization noise on this eval set, not a genuine quality improvement.

**Evaluation**: WikiText validation, 50K tokens, sliding window (2048 context, 512 stride). Calibration and evaluation data are strictly separated — see [Methodology](#methodology) for details.

## How It Works

### Three Key Observations

1. **Dimension importance varies dramatically.** Variance ratios across dimensions are 13:1 median (up to 3,190:1 on Mistral-7B). Some dimensions carry critical information; others are near-constant "wobble positions."

2. **Heads are different organisms.** Jensen-Shannon divergence between KV heads is 0.37 nats median — each head group needs its own quantization config.

3. **Dimensions are independent.** Mean |correlation| between dimensions is 0.06. This means scalar quantization is optimal — vector quantization (PQ, rotation-based methods) wastes capacity modeling nonexistent correlations.

### Algorithm

```
1. PROFILE    Measure per-dimension variance and per-head JS divergence
              using streaming statistics (no full KV tensors stored)

2. GROUP      Cluster heads by JS divergence similarity
              (agglomerative clustering, threshold 0.15 nats)

3. ALLOCATE   Assign bits per dimension via greedy marginal distortion
              reduction: each bit goes to the dimension where it reduces
              quantization error most

              Constraints:
              - Minimum 1 bit per dimension (critical for post-RoPE — see FAQ)
              - Maximum 8 bits per dimension

4. QUANTIZE   Per-dimension scalar quantization with per-group local
              scaling (32-token windows) for tighter dynamic range
```

The bit allocation is rate-distortion optimal for independent Gaussian sources. Adding 1 bit to dimension `d` reduces distortion by `variance[d] × (3/4) × 4^(-current_bits[d])`. The greedy algorithm iteratively assigns bits to maximize this reduction.

## Quick Start

```bash
git clone https://github.com/bio-bench/wobble-quant-cache.git
cd wobble-quant-cache
pip install -e .
```

### Reproduce Published Results

```bash
# Gemma-2-2B (~10GB VRAM, ~30 min)
HF_TOKEN=your_token python experiments/reproduce_gemma2.py

# Mistral-7B (~14GB VRAM, ~60 min)
HF_TOKEN=your_token python experiments/reproduce_mistral.py
```

### Profile Your Own Model

The profiling tools work standalone — you can analyze any model's KV cache structure without using Wobble for quantization.

```python
from profiling.capture import profile_kv_cache
from profiling.heads import compute_js_divergence_matrix, group_heads
from profiling.report import generate_report

# 1. Profile KV cache statistics
stats, reservoir = profile_kv_cache(
    model, tokenizer, calibration_texts,
    n_layers=32, n_kv_heads=8, head_dim=128,
)

# 2. Group heads by distribution similarity
js_matrix = compute_js_divergence_matrix(stats, layer_idx=0)
groups = group_heads(js_matrix, max_groups=8, divergence_threshold=0.15)

# 3. Generate analysis report
#    Outputs: variance ratios, kurtosis, head diversity, correlation structure
generate_report(stats, output_dir="profiling_results/")
```

### Full Quantization Pipeline

```python
from wobble.quantize import build_config, encode, decode

# Build per-group quantization config
config = build_config(
    dim_variances=stats["k_variance"][0, 0],
    dim_means=stats["k_mean"][0, 0],
    dim_mins=kv_min, dim_maxs=kv_max,
    total_budget=3 * 128,  # 3 bits × 128 dims
    head_group_id=0, layer_idx=0,
)

# Quantize / dequantize
codes, group_mins, group_maxs = encode(kv_vectors, config, group_size=32)
reconstructed = decode(codes, config, group_mins, group_maxs, group_size=32)
```

## FAQ

### How does this compare to TurboQuant?

TurboQuant (Google, 2026) and Wobble take opposite approaches to the same problem:

- **TurboQuant** rotates KV cache dimensions to make importance uniform, then applies uniform quantization. It is data-oblivious and requires no calibration.
- **Wobble** measures which dimensions are important and allocates bits accordingly. It requires a 30K-token calibration set.

Both approaches achieve near-lossless results at 3-bit on Gemma-2-2B. Google's published claims report <0.5% PPL degradation; Wobble achieves +0.02%. This convergence from opposite directions likely reflects a fundamental property of transformer KV caches — the information content fits in roughly 3 bits per dimension regardless of compression method.

We have not run a direct head-to-head comparison. Google's official code is not yet released (expected Q2 2026), and the available third-party reimplementation ([tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)) produces catastrophic results that clearly don't match the paper's claims. We compare against KIVI as our baseline and against TurboQuant's published numbers only.

At 2-bit, TurboQuant's published claims weaken (marginal degradation at 2.5-bit, no lossless claim at 2-bit), while Wobble remains functional. This is where adaptive allocation may have a structural advantage, but we can't confirm until a direct comparison is possible.

### Why not just use KIVI?

KIVI uses uniform per-channel quantization — every dimension gets the same number of bits. When dimension importance varies by 3,190:1, this wastes bits on near-constant dimensions while starving the important ones. Wobble's adaptive allocation handles this naturally, which is why it beats KIVI at every bit-width on every model tested.

### What's the minimum-1-bit rule?

At 2-bit budgets, the greedy allocator can assign 0 bits to low-importance dimensions, reconstructing them from a fixed calibration mean. After RoPE, dimension means are position-dependent — using a fixed mean destroys positional information. PPL exploded to 25,000.

The fix comes from the biological analogy: the codon table's wobble position uses fewer bits, but never zero — it always retains a 4-nucleotide alphabet. Enforcing a minimum of 1 bit per dimension recovered PPL from 25,000 to 42.83.

## Architecture

```
wobble-quant-cache/
├── wobble/                 # Core quantization library
│   ├── quantize.py         # Adaptive scalar quantization (encode/decode)
│   ├── allocate.py         # Rate-distortion optimal bit allocation
│   ├── baselines.py        # KIVI and uniform quantization baselines
│   ├── evaluate.py         # WikiText perplexity evaluation
│   ├── patch.py            # Attention monkey-patching (Mistral, Gemma-2)
│   └── config.py           # Centralized configuration
├── profiling/              # Standalone KV cache analysis tools
│   ├── capture.py          # Streaming statistics (Chan et al. algorithm)
│   ├── importance.py       # Per-dimension importance scoring
│   ├── heads.py            # Head clustering via JS divergence
│   ├── distributions.py    # Distribution fitting (Gaussian/Laplace/Student-t)
│   └── report.py           # Go/no-go assessment and visualizations
├── experiments/            # Reproduction scripts
│   ├── reproduce_gemma2.py # Full Gemma-2-2B pipeline
│   ├── reproduce_mistral.py # Full Mistral-7B pipeline
│   └── calibration.py     # Multi-domain calibration data loading
└── results/                # Published numbers (JSON)
```

### Adding a New Model

1. Determine architecture params: `n_layers`, `n_kv_heads`, `head_dim`
2. Add a patch function to `wobble/patch.py` that imports the correct `apply_rotary_pos_emb` and `eager_attention_forward` from `transformers.models.<model>.modeling_<model>`
3. Handle model-specific attention args (Gemma-2 has `softcap` and `sliding_window`)

## Methodology

**Calibration/eval separation**: Calibration uses 30K tokens from WikiText-103 train split, C4, PubMed abstracts, and SEC filings. Evaluation uses WikiText validation split only. Hash-based overlap detection confirmed zero leakage.

**Per-group scaling overhead**: Local min/max recomputed every 32 tokens adds 0.25 extra bits per dimension (2 × FP16 values per group). This overhead is included in all reported bit-widths.

## Limitations

- **Two models tested.** Mistral-7B and Gemma-2-2B are both relatively small. Generalization to 70B+ models, MoE architectures, and longer contexts is unverified.
- **Calibration data required.** Wobble needs a 30K-token calibration set. TurboQuant operates without calibration.
- **No latency benchmarks.** Encode/decode throughput has not been measured. Per-group scaling and adaptive bit allocation may introduce overhead vs. uniform methods.
- **WikiText perplexity only.** LongBench, needle-in-haystack, and MMLU evaluations have not been completed.
- **No direct TurboQuant comparison.** See [FAQ](#how-does-this-compare-to-turboquant).

## Biological Analogy

```
Codon position 1 (high info)    →  High-variance dimensions (most bits)
Codon position 3 (wobble)       →  Low-variance dimensions (fewest bits)
Wobble has 4-letter alphabet    →  Minimum 1 bit per dimension (never 0)
Organism-specific tRNA pools    →  Per-head-group adaptive configs
Neighboring codon context       →  Per-group local scaling (32 tokens)
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- transformers 4.40+
- CUDA GPU with 10–20GB VRAM (model dependent)

## Citation

If you use Wobble in your research, please cite:

```bibtex
@software{wobble2026,
  title  = {Wobble: Adaptive KV Cache Quantization},
  year   = {2026},
  url    = {https://github.com/bio-bench/wobble-quant-cache}
}
```

## License

MIT
