"""Block-Hybrid KV cache quantization.

Splits each KV vector into VIP (high-variance) and Wobble (low-variance)
dimensions. VIPs are rotated via FWHT to spread concentrated energy, then
quantized at higher precision (4-bit). Wobbles get 2-bit. Per-group
scaling keeps quantization ranges tight.

All data-parallel operations are fully vectorized -- no Python loops
over tokens, heads, or groups.
"""

from dataclasses import dataclass

import torch


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard Transform (butterfly algorithm).

    Input shape: [..., N] where N is a power of 2.
    The log2(N) butterfly passes are inherent to the algorithm and each
    pass is a single vectorized tensor op.
    """
    d = x.shape[-1]
    orig_shape = x.shape
    x = x.reshape(-1, d).float()

    h = 1
    while h < d:  # log2(d) iterations, each fully vectorized
        x = x.view(-1, d // (2 * h), 2, h)
        a, b = x[:, :, 0], x[:, :, 1]
        x = torch.stack([a + b, a - b], dim=2)
        h *= 2

    return x.view(orig_shape) * (d ** -0.5)


def _per_group_quantize_dequantize(
    x: torch.Tensor, n_bits: int, group_size: int
) -> torch.Tensor:
    """Uniform quantize-then-dequantize with per-group scaling.

    Fully vectorized: reshapes [N, D] -> [n_groups, group_size, D],
    computes group-wise min/max in one amin/amax call, quantizes, and
    dequantizes without any Python loop over groups.

    Args:
        x: [N, D] float tensor.
        n_bits: Quantization bit-width (e.g. 2 or 4).
        group_size: Tokens per scaling group.

    Returns:
        [N, D] float tensor (reconstructed).
    """
    N, D = x.shape
    n_levels = (1 << n_bits) - 1

    if group_size <= 0 or N <= group_size:
        # Global scaling fallback
        g_min = x.amin(dim=0, keepdim=True)
        g_max = x.amax(dim=0, keepdim=True)
        scale = (g_max - g_min) / (n_levels + 1e-8)
        codes = ((x - g_min) / (scale + 1e-8)).round().clamp(0, n_levels)
        return codes * scale + g_min

    # Pad N to multiple of group_size
    n_groups = (N + group_size - 1) // group_size
    padded_n = n_groups * group_size
    if padded_n > N:
        x_padded = torch.zeros(padded_n, D, device=x.device, dtype=x.dtype)
        x_padded[:N] = x
    else:
        x_padded = x

    # [n_groups, group_size, D] -- single reshape, no loop
    x_grouped = x_padded.view(n_groups, group_size, D)

    # Per-group min/max: [n_groups, 1, D]
    g_min = x_grouped.amin(dim=1, keepdim=True)
    g_max = x_grouped.amax(dim=1, keepdim=True)
    scale = (g_max - g_min) / (n_levels + 1e-8)

    # Quantize + dequantize (all vectorized)
    codes = ((x_grouped - g_min) / (scale + 1e-8)).round().clamp(0, n_levels)
    recon = codes * scale + g_min

    return recon.view(padded_n, D)[:N]


def turboquant_quantize_dequantize(
    vectors: torch.Tensor, n_bits: int, group_size: int = 32
) -> torch.Tensor:
    """TurboQuant: FWHT rotate ALL dims -> uniform quantize -> inverse FWHT.

    No VIP/wobble split. Rotation spreads outlier energy uniformly,
    then uniform quantization is applied to all dimensions equally.

    Args:
        vectors: [N, D] float tensor.
        n_bits: Uniform bit-width for all dims.
        group_size: Per-group scaling window.

    Returns:
        [N, D] reconstructed float tensor.
    """
    rotated = fwht_torch(vectors)
    recon_rot = _per_group_quantize_dequantize(rotated, n_bits, group_size)
    return fwht_torch(recon_rot).to(vectors.dtype)


@dataclass
class HybridLayerConfig:
    """Per-layer hybrid quantization config."""
    vip_indices: torch.Tensor    # [n_vip] sorted indices of VIP dims
    wobble_indices: torch.Tensor  # [n_wobble] remaining dim indices
    vip_bits: int = 4
    wobble_bits: int = 2
    group_size: int = 32


def hybrid_quantize_dequantize(
    vectors: torch.Tensor,
    layer_cfg: HybridLayerConfig,
) -> torch.Tensor:
    """Block-hybrid quantize-then-dequantize. Fully vectorized.

    1. Gather VIP dims, FWHT rotate, per-group 4-bit quantize, inverse FWHT.
    2. Gather Wobble dims, per-group 2-bit quantize.
    3. Scatter back to original dimension layout.

    Args:
        vectors: [N, D] float tensor (flattened KV cache vectors).
        layer_cfg: HybridLayerConfig with VIP/wobble indices and bit-widths.

    Returns:
        [N, D] reconstructed float tensor.
    """
    vips = vectors[:, layer_cfg.vip_indices]
    wobbles = vectors[:, layer_cfg.wobble_indices]

    # VIP path: rotate -> quantize -> dequantize -> inverse rotate
    # FWHT is self-inverse (with normalization built in)
    rotated = fwht_torch(vips)
    recon_rot = _per_group_quantize_dequantize(
        rotated, layer_cfg.vip_bits, layer_cfg.group_size
    )
    recon_vips = fwht_torch(recon_rot)

    # Wobble path: direct quantize -> dequantize
    recon_wobbles = _per_group_quantize_dequantize(
        wobbles, layer_cfg.wobble_bits, layer_cfg.group_size
    )

    # Scatter back
    out = torch.empty_like(vectors)
    out[:, layer_cfg.vip_indices] = recon_vips.to(vectors.dtype)
    out[:, layer_cfg.wobble_indices] = recon_wobbles.to(vectors.dtype)
    return out


def _adaptive_vip_count(
    variance: "np.ndarray",
    energy_fraction: float = 0.5,
    min_vip: int = 16,
) -> int:
    """Compute per-layer VIP count based on variance energy concentration.

    Sorts dimensions by variance descending, finds the smallest set that
    captures >= energy_fraction of total variance energy. Rounds result
    to nearest power of 2 (required by FWHT).

    Layers with concentrated variance -> fewer VIPs.
    Layers with uniform variance -> more VIPs.
    """
    import numpy as np

    sorted_var = np.sort(variance)[::-1]
    cumsum = np.cumsum(sorted_var)
    total = cumsum[-1]

    if total < 1e-12:
        return min_vip

    # Index where cumulative energy first reaches the target fraction
    threshold_idx = int(np.searchsorted(cumsum, energy_fraction * total)) + 1
    threshold_idx = max(threshold_idx, min_vip)
    threshold_idx = min(threshold_idx, len(variance) // 2)

    # Round to nearest power of 2 (use closest, not just floor)
    floor_p2 = 1 << (threshold_idx.bit_length() - 1)
    ceil_p2 = floor_p2 << 1
    if ceil_p2 <= len(variance) // 2 and (ceil_p2 - threshold_idx) < (threshold_idx - floor_p2):
        return ceil_p2
    return floor_p2


def build_hybrid_configs_from_variance(
    layer_variances: dict[int, dict],
    vip_fraction: float | None = 0.125,
    adaptive_energy: float | None = None,
    vip_bits: int = 4,
    wobble_bits: int = 2,
    group_size: int = 32,
    device: torch.device | str = "cpu",
) -> dict[int, HybridLayerConfig]:
    """Build per-layer HybridLayerConfig from profiled dimension variances.

    Two modes:
    - Fixed: vip_fraction sets a uniform fraction across all layers.
    - Adaptive: adaptive_energy sets a target energy fraction; each layer
      gets as many VIPs as needed to capture that fraction of total
      variance. Layers with concentrated variance get fewer VIPs,
      layers with spread variance get more.

    If adaptive_energy is set, it takes precedence over vip_fraction.

    Args:
        layer_variances: {layer_idx: {'variance': np.ndarray[head_dim], 'head_dim': int}}.
        vip_fraction: Fixed fraction of dims as VIP (default 12.5%).
        adaptive_energy: If set, target energy fraction for adaptive selection (e.g. 0.5).
        vip_bits: Bit-width for VIP dims.
        wobble_bits: Bit-width for Wobble dims.
        group_size: Per-group scaling window.
        device: Target device for index tensors.

    Returns:
        {layer_idx: HybridLayerConfig}.
    """
    import numpy as np

    adaptive = adaptive_energy is not None

    configs = {}
    for li, stats in layer_variances.items():
        head_dim = stats["head_dim"]
        var = stats["variance"]

        if adaptive:
            n_vip = _adaptive_vip_count(var, energy_fraction=adaptive_energy)
        else:
            n_vip_raw = max(1, int(head_dim * vip_fraction))
            n_vip = 1 << (n_vip_raw.bit_length() - 1)
        n_vip = min(n_vip, head_dim)

        # Top-n_vip dimensions by variance
        top_k = var.argsort()[-n_vip:]
        top_k_sorted = top_k[top_k.argsort()]

        all_dims = set(range(head_dim))
        vip_set = set(top_k_sorted.tolist())
        wobble_sorted = sorted(all_dims - vip_set)

        configs[li] = HybridLayerConfig(
            vip_indices=torch.tensor(top_k_sorted, dtype=torch.long, device=device),
            wobble_indices=torch.tensor(wobble_sorted, dtype=torch.long, device=device),
            vip_bits=vip_bits,
            wobble_bits=wobble_bits,
            group_size=group_size,
        )

        avg_bits = (n_vip * vip_bits + (head_dim - n_vip) * wobble_bits) / head_dim
        if li < 3 or li % 10 == 0:
            print(
                f"  Layer {li:2d}: head_dim={head_dim}, "
                f"n_vip={n_vip}/{head_dim} ({n_vip/head_dim*100:.0f}%), "
                f"avg_bits={avg_bits:.2f}",
                flush=True,
            )

    return configs
