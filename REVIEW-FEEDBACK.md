# Review Feedback — Wobble Hypothesis Validation
Date: 2026-04-02
Ready for Builder: YES

## Must Fix

### 1. Set `min_bits=1` universally — silent correctness risk at 3-bit

0-bit dims destroy positional info post-RoPE (documented PPL explosion to 25,000). Currently 3-bit configs use `min_bits=0` and only survive by luck (high budget means no dim actually gets 0 bits for these two models). Not guaranteed for other models.

- `wobble/config.py:55` — change `min_bits_per_dim: int = 0` to `min_bits_per_dim: int = 1`
- `experiments/reproduce_gemma2.py:220` — 3-bit `build_wobble_configs` call omits `min_bits`, defaulting to 0. Pass `min_bits=1` explicitly.
- `experiments/reproduce_mistral.py` — same pattern, same fix.
- `results/gemma2_2b.json:37` and `results/mistral_7b.json` — update `3bit_min_bits_per_dim` from `0` to `1`.

### 2. Reproduction scripts use naive sum/sum-sq instead of `StatsAccumulator`

`reproduce_gemma2.py:70-109` implements its own profiling loop with naive running sum/sum-of-squares. The library provides `profiling/capture.py:StatsAccumulator` with numerically stable Chan et al. parallel updates. These can produce different variance estimates, especially at the 3,190:1 variance ratios this project measures.

**Fix:**

**Step A** — Add min/max tracking to `StatsAccumulator` in `profiling/capture.py`. It tracks mean/m2/m4 but not min/max, which `build_config` needs for clipping bounds. Add to `__post_init__`:
```python
for prefix in ("k_", "v_"):
    setattr(self, f"{prefix}min", np.full(shape, np.inf, dtype=np.float64))
    setattr(self, f"{prefix}max", np.full(shape, -np.inf, dtype=np.float64))
```
Update `update()` to track element-wise min/max. Return them from `finalize()`.

**Step B** — Delete `profile_post_rope()` from `reproduce_gemma2.py` (lines 70-109). Replace with a call to `profile_kv_cache()` from `profiling/capture.py`. Average K/V stats to match current behavior:
```python
dim_var = (stats["k_variance"] + stats["v_variance"]) / 2.0
dim_mean = (stats["k_mean"] + stats["v_mean"]) / 2.0
dim_min = np.minimum(stats["k_min"], stats["v_min"])
dim_max = np.maximum(stats["k_max"], stats["v_max"])
```

**Step C** — Apply same refactor to `reproduce_mistral.py`.

### 3. JS divergence: histogram-based in scripts vs Gaussian-analytical in library

`reproduce_gemma2.py:112-152` computes JS divergence from raw histograms with hardcoded range `(-5.0, 5.0)` and 100 bins. `profiling/heads.py:compute_js_divergence_matrix` computes it analytically from Gaussian parameters. These are different methods that can produce different head groupings.

**Fix:**

Delete `compute_js_matrices()` from `reproduce_gemma2.py` (lines 112-152). Replace with:
```python
from profiling.heads import compute_js_divergence_matrix

js_data = {}
for li in range(N_LAYERS):
    js_data[f"layer_{li}"] = compute_js_divergence_matrix(stats, li)
```
The `stats` dict from `StatsAccumulator.finalize()` already has the format `compute_js_divergence_matrix` expects. Apply same change to `reproduce_mistral.py`.

## Should Fix
(Not blocking — defer to next pass.)

- No unit tests for core algorithms (`assign_bits_greedy`, encode/decode roundtrip, water-filling, head grouping)
- Sequential loops in `encode()`, `fit_distribution()`, `ReservoirSampler.update()` — violates project code style
- Two bit allocation algorithms exist (`assign_bits_greedy` vs `optimize_bit_allocation`) — unclear which is canonical
- `calibration.py:88` C4 skip logic is arbitrary

## Escalate to Architect
- Hypothesis validated on only 2 models; Mistral-7B degrades significantly — is this sufficient to claim generality?
- TurboQuant comparison uses third-party reimplementation, not official code — credibility risk in published results

## Cleared
Rate-distortion theory, greedy bit allocation, encode/decode logic, Chan et al. streaming stats, head grouping, data separation, evaluation protocol, monkey-patching, config centralization — all reviewed and passed.
