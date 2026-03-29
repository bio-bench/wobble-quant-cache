"""Adaptive per-dimension scalar quantization.

Each dimension gets an integer bit-width assigned by greedy marginal
distortion reduction. High-variance dimensions get more bits,
low-variance "wobble" dimensions get fewer (potentially 0 = use mean).

This is optimal for independent Gaussian dimensions -- and KV cache
dimensions are largely uncorrelated (avg |corr| = 0.06), making
scalar quantization strictly better than vector quantization.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveScalarConfig:
    """Per-dimension bit assignment and quantization parameters.

    Stored per (layer, head_group).
    """

    bits_per_dim: np.ndarray   # [head_dim], values in [0, max_bits]
    dim_min: np.ndarray        # [head_dim], calibration min per dim
    dim_max: np.ndarray        # [head_dim], calibration max per dim
    dim_mean: np.ndarray       # [head_dim], calibration mean (for 0-bit dims)
    head_group_id: int
    layer_idx: int

    @property
    def head_dim(self) -> int:
        return len(self.bits_per_dim)

    @property
    def total_bits(self) -> int:
        return int(np.sum(self.bits_per_dim))

    @property
    def avg_bits_per_dim(self) -> float:
        return self.total_bits / self.head_dim

    def summary(self) -> str:
        bits, counts = np.unique(self.bits_per_dim, return_counts=True)
        dist = ", ".join(f"{b}bit:{c}" for b, c in zip(bits, counts))
        return (
            f"AdaptiveScalar(layer={self.layer_idx}, group={self.head_group_id}, "
            f"avg={self.avg_bits_per_dim:.2f}, total={self.total_bits}, "
            f"dist=[{dist}])"
        )


def assign_bits_greedy(
    dim_variances: np.ndarray,
    total_budget: int,
    max_bits: int = 8,
    min_bits: int = 0,
) -> np.ndarray:
    """Greedy per-dimension bit allocation via marginal distortion reduction.

    Iteratively assigns the next bit to the dimension where it reduces
    distortion most. For scalar quantization of a Gaussian source, adding
    1 bit to dimension d reduces distortion by:

        delta = variance[d] * (3/4) * 4^(-current_bits[d])

    This is optimal for independent Gaussian dimensions and produces
    exactly the requested total budget.

    Args:
        dim_variances: Shape [head_dim]. Per-dimension variance from profiling.
        total_budget: Total bits to distribute across all dimensions.
        max_bits: Maximum bits per dimension (default 8).
        min_bits: Minimum bits per dimension (default 0 = full degeneracy).

    Returns:
        bits_per_dim: Shape [head_dim], integer bit assignments.
    """
    head_dim = len(dim_variances)

    if total_budget < 0:
        raise ValueError(f"total_budget must be non-negative, got {total_budget}")
    if total_budget > head_dim * max_bits:
        raise ValueError(
            f"total_budget ({total_budget}) exceeds max possible "
            f"({head_dim * max_bits} = {head_dim} dims * {max_bits} bits)"
        )

    bits = np.full(head_dim, min_bits, dtype=np.int32)
    remaining = total_budget - int(np.sum(bits))

    if remaining < 0:
        raise ValueError(
            f"min_bits={min_bits} * head_dim={head_dim} = {min_bits * head_dim} "
            f"exceeds total_budget={total_budget}"
        )

    var = dim_variances.astype(np.float64)
    gain = var * 0.75 * (4.0 ** (-bits.astype(np.float64)))

    for _ in range(remaining):
        masked_gain = np.where(bits < max_bits, gain, -1.0)
        best = np.argmax(masked_gain)

        if masked_gain[best] <= 0:
            logger.warning(
                "No dimension can accept more bits (all at max=%d). "
                "%d bits unassigned.", max_bits, remaining,
            )
            break

        bits[best] += 1
        gain[best] *= 0.25  # next marginal gain is 4x smaller

    actual_total = int(np.sum(bits))
    if actual_total != total_budget:
        logger.warning("Bit assignment: allocated %d / %d bits", actual_total, total_budget)

    return bits


def build_config(
    dim_variances: np.ndarray,
    dim_means: np.ndarray,
    dim_mins: np.ndarray,
    dim_maxs: np.ndarray,
    total_budget: int,
    head_group_id: int,
    layer_idx: int,
    max_bits: int = 8,
    min_bits: int = 0,
) -> AdaptiveScalarConfig:
    """Build adaptive scalar config for one (layer, head_group).

    Args:
        dim_variances: Shape [head_dim]. Per-dimension variance.
        dim_means: Shape [head_dim]. Per-dimension mean.
        dim_mins: Shape [head_dim]. Per-dimension min (from calibration).
        dim_maxs: Shape [head_dim]. Per-dimension max (from calibration).
        total_budget: Total bits for this vector.
        head_group_id: Head group index.
        layer_idx: Layer index.
        max_bits: Maximum bits per dimension.
        min_bits: Minimum bits per dimension.

    Returns:
        AdaptiveScalarConfig.
    """
    bits = assign_bits_greedy(dim_variances, total_budget, max_bits=max_bits,
                              min_bits=min_bits)

    # Widen min/max by a small margin for clipping safety
    margin = 0.01 * (dim_maxs - dim_mins + 1e-10)
    safe_mins = dim_mins - margin
    safe_maxs = dim_maxs + margin

    config = AdaptiveScalarConfig(
        bits_per_dim=bits,
        dim_min=safe_mins.astype(np.float32),
        dim_max=safe_maxs.astype(np.float32),
        dim_mean=dim_means.astype(np.float32),
        head_group_id=head_group_id,
        layer_idx=layer_idx,
    )
    logger.info("Built %s", config.summary())
    return config


def encode(
    kv_vectors: torch.Tensor,
    config: AdaptiveScalarConfig,
    group_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Encode KV vectors using per-dimension adaptive scalar quantization.

    When group_size > 0, computes local min/max per group of tokens
    instead of using global calibration min/max. This gives tighter
    quantization ranges at the cost of storing per-group scale factors.

    Args:
        kv_vectors: Shape [N, head_dim] in float.
        config: AdaptiveScalarConfig with per-dim bit assignments.
        group_size: If > 0, use per-group local scaling. 0 = global.

    Returns:
        (codes, group_mins, group_maxs):
            codes: Shape [N, head_dim], dtype uint8.
            group_mins: Shape [n_groups, head_dim] or None if global.
            group_maxs: Shape [n_groups, head_dim] or None if global.
    """
    N, D = kv_vectors.shape
    device = kv_vectors.device
    kv_float = kv_vectors.float()

    bits = torch.from_numpy(config.bits_per_dim).to(device)
    n_levels = ((1 << bits.long()) - 1).float()
    zero_mask = bits == 0

    if group_size > 0 and N > group_size:
        n_groups = (N + group_size - 1) // group_size
        codes = torch.zeros(N, D, dtype=torch.uint8, device=device)
        group_mins = torch.zeros(n_groups, D, dtype=torch.float32, device=device)
        group_maxs = torch.zeros(n_groups, D, dtype=torch.float32, device=device)

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, N)
            group = kv_float[start:end]

            g_min = group.amin(dim=0)
            g_max = group.amax(dim=0)
            group_mins[g] = g_min
            group_maxs[g] = g_max

            drange = g_max - g_min
            safe_range = torch.where(drange > 0, drange, torch.ones_like(drange))
            scale = safe_range / torch.where(n_levels > 0, n_levels, torch.ones_like(n_levels))

            normalized = (group - g_min) / scale
            g_codes = torch.round(normalized)
            g_codes = torch.minimum(g_codes, n_levels.unsqueeze(0))
            g_codes = torch.maximum(g_codes, torch.zeros_like(g_codes))
            g_codes[:, zero_mask] = 0
            codes[start:end] = g_codes.to(torch.uint8)

        return codes, group_mins, group_maxs
    else:
        vmin = torch.from_numpy(config.dim_min).to(device).float()
        vmax = torch.from_numpy(config.dim_max).to(device).float()

        drange = vmax - vmin
        safe_range = torch.where(drange > 0, drange, torch.ones_like(drange))
        scale = safe_range / torch.where(n_levels > 0, n_levels, torch.ones_like(n_levels))

        normalized = (kv_float - vmin) / scale
        codes = torch.round(normalized)
        codes = torch.minimum(codes, n_levels.unsqueeze(0))
        codes = torch.maximum(codes, torch.zeros_like(codes))
        codes[:, zero_mask] = 0

        return codes.to(torch.uint8), None, None


def decode(
    codes: torch.Tensor,
    config: AdaptiveScalarConfig,
    group_mins: torch.Tensor | None = None,
    group_maxs: torch.Tensor | None = None,
    group_size: int = 0,
) -> torch.Tensor:
    """Decode quantized codes back to approximate KV vectors.

    Args:
        codes: Shape [N, head_dim], dtype uint8.
        config: AdaptiveScalarConfig.
        group_mins: Shape [n_groups, head_dim] or None.
        group_maxs: Shape [n_groups, head_dim] or None.
        group_size: Group size used during encoding. 0 = global.

    Returns:
        Reconstructed vectors, shape [N, head_dim], float32.
    """
    N, D = codes.shape
    device = codes.device

    bits = torch.from_numpy(config.bits_per_dim).to(device)
    n_levels = ((1 << bits.long()) - 1).float()
    zero_mask = bits == 0
    vmean = torch.from_numpy(config.dim_mean).to(device).float()

    if group_mins is not None and group_maxs is not None and group_size > 0:
        decoded = torch.zeros(N, D, dtype=torch.float32, device=device)
        n_groups = group_mins.shape[0]

        for g in range(n_groups):
            start = g * group_size
            end = min(start + group_size, N)
            g_min = group_mins[g]
            g_max = group_maxs[g]

            drange = g_max - g_min
            safe_range = torch.where(drange > 0, drange, torch.ones_like(drange))
            scale = safe_range / torch.where(n_levels > 0, n_levels, torch.ones_like(n_levels))
            decoded[start:end] = codes[start:end].float() * scale + g_min

        decoded[:, zero_mask] = vmean[zero_mask]
        return decoded
    else:
        vmin = torch.from_numpy(config.dim_min).to(device).float()
        vmax = torch.from_numpy(config.dim_max).to(device).float()

        drange = vmax - vmin
        safe_range = torch.where(drange > 0, drange, torch.ones_like(drange))
        scale = safe_range / torch.where(n_levels > 0, n_levels, torch.ones_like(n_levels))

        decoded = codes.float() * scale + vmin
        decoded[:, zero_mask] = vmean[zero_mask]
        return decoded


def quantize_dequantize(
    kv_vectors: torch.Tensor,
    config: AdaptiveScalarConfig,
    group_size: int = 0,
) -> torch.Tensor:
    """Quantize then immediately dequantize (simulate quantization error).

    Convenience function for use in attention patching.

    Args:
        kv_vectors: Shape [N, head_dim].
        config: AdaptiveScalarConfig.
        group_size: Per-group local scaling window. 0 = global.

    Returns:
        Reconstructed vectors, shape [N, head_dim], float32.
    """
    codes, g_mins, g_maxs = encode(kv_vectors, config, group_size=group_size)
    return decode(codes, config, g_mins, g_maxs, group_size=group_size)
