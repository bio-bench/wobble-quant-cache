import numpy as np


def fwht(a):
    """Fast Walsh-Hadamard Transform."""
    n = a.shape[0]
    if n == 1:
        return a
    a_left = fwht(a[0 : n // 2])
    a_right = fwht(a[n // 2 : n])
    res = np.empty_like(a)
    res[0 : n // 2] = a_left + a_right
    res[n // 2 : n] = a_left - a_right
    return res


def quantize_sim(vec, n_bits):
    if n_bits == 0:
        return np.zeros_like(vec)
    n_levels = 2**n_bits - 1
    v_min, v_max = vec.min(), vec.max()
    if v_max == v_min:
        return np.full_like(vec, v_min)
    scale = (v_max - v_min) / n_levels
    q = np.round((vec - v_min) / (scale + 1e-12))
    return q * scale + v_min


# Gemma 4 31B-it Specs (April 2026)
N_LAYERS = 60
N_KV_HEADS = 8
HEAD_DIM = 256
CONTEXT_LEN = 256_000

# Simulation setup
N_SAMPLES = 1000
N_VIPS = 32  # Top 12.5% of dimensions
VIP_BITS = 4
WOBBLE_BITS = 2

# AVG BITS = (32*4 + 224*2) / 256 = (128 + 448) / 256 = 576 / 256 = 2.25 bits

print(f"--- Gemma 4 31B-it Hybrid Quantization Proxy ---")
print(f"Target: 2.25 bits/dim (average)")
print(f"VIPs: {N_VIPS} dims (4-bit + Rotation)")
print(f"Wobbles: {HEAD_DIM - N_VIPS} dims (2-bit Scalar)")

# Generate synthetic "Spiky" vectors
rng = np.random.default_rng(42)
variances = 1.0 / (np.arange(1, HEAD_DIM + 1) ** 1.8)
variances = variances / variances.max() * 100.0
samples = rng.normal(0, 1.0, (N_SAMPLES, HEAD_DIM)) * np.sqrt(variances)

cos_sim_std_2b = []
cos_sim_hybrid = []

for i in range(N_SAMPLES):
    original = samples[i]

    std_2b = quantize_sim(original, 2)
    cos_std = np.dot(original, std_2b) / (
        np.linalg.norm(original) * np.linalg.norm(std_2b) + 1e-12
    )
    cos_sim_std_2b.append(cos_std)

    idx = np.argsort(-np.abs(original))
    vip_idx = idx[:N_VIPS]
    wobble_idx = idx[N_VIPS:]
    vips = original[vip_idx]
    wobbles = original[wobble_idx]

    rotated_vips = fwht(vips) / np.sqrt(N_VIPS)
    q_rot_vips = quantize_sim(rotated_vips, VIP_BITS)
    recon_vips = fwht(q_rot_vips) / np.sqrt(N_VIPS)
    q_wobbles = quantize_sim(wobbles, WOBBLE_BITS)

    recon_hybrid = np.zeros_like(original)
    recon_hybrid[vip_idx] = recon_vips
    recon_hybrid[wobble_idx] = q_wobbles

    cos_hybrid = np.dot(original, recon_hybrid) / (
        np.linalg.norm(original) * np.linalg.norm(recon_hybrid) + 1e-12
    )
    cos_sim_hybrid.append(cos_hybrid)

print(f"\nResults over {N_SAMPLES} vectors:")
print(f"  Standard 2-bit Cosine Similarity: {np.mean(cos_sim_std_2b):.6f}")
print(f"  Block-Hybrid 2.25-bit Cosine Similarity: {np.mean(cos_sim_hybrid):.6f}")

fp16_size_gb = (N_LAYERS * N_KV_HEADS * HEAD_DIM * CONTEXT_LEN * 2 * 2) / 1024**3
hybrid_size_gb = (
    N_LAYERS * N_KV_HEADS * HEAD_DIM * CONTEXT_LEN * 2 * (2.25 / 8)
) / 1024**3

print(f"\nGemma 4 31B (256K Context) VRAM Analysis:")
print(f"  FP16 Cache:     {fp16_size_gb:.2f} GB")
print(f"  Hybrid 2.25-bit: {hybrid_size_gb:.2f} GB")
print(f"  Savings:        {100 * (1 - hybrid_size_gb / fp16_size_gb):.1f}%")
