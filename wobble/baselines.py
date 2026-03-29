"""Static quantization baselines for comparison.

Implements uniform scalar quantization and KIVI methodology
(per-channel keys, per-token values) as baselines.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def quantize_uniform(
    tensor: torch.Tensor,
    n_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform asymmetric scalar quantization (per-tensor min/max).

    Args:
        tensor: Input tensor (any shape), float.
        n_bits: Number of bits (1-8).

    Returns:
        (codes, scale, zero_point).
    """
    if n_bits < 1 or n_bits > 8:
        raise ValueError(f"n_bits must be in [1, 8], got {n_bits}")

    n_levels = (1 << n_bits) - 1
    t_min = tensor.min()
    t_max = tensor.max()
    t_range = t_max - t_min

    if t_range == 0:
        scale = torch.ones(1, device=tensor.device, dtype=tensor.dtype)
        zero_point = t_min
        codes = torch.zeros_like(tensor, dtype=torch.int8)
        return codes, scale, zero_point

    scale = t_range / n_levels
    zero_point = t_min
    codes = torch.round((tensor - zero_point) / scale).clamp(0, n_levels).to(torch.int8)
    return codes, scale, zero_point


def dequantize_uniform(
    codes: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Inverse of quantize_uniform: x_approx = codes * scale + zero_point."""
    return codes.float() * scale + zero_point


def quantize_per_channel(
    tensor: torch.Tensor,
    n_bits: int,
    channel_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform quantization with per-channel scale/zero_point."""
    if n_bits < 1 or n_bits > 8:
        raise ValueError(f"n_bits must be in [1, 8], got {n_bits}")

    n_levels = (1 << n_bits) - 1
    reduce_dims = [d for d in range(tensor.dim()) if d != channel_dim]

    t_min = tensor.amin(dim=reduce_dims, keepdim=True)
    t_max = tensor.amax(dim=reduce_dims, keepdim=True)
    t_range = t_max - t_min

    safe_range = torch.where(t_range > 0, t_range, torch.ones_like(t_range))
    scales = safe_range / n_levels
    zero_points = t_min

    codes = torch.round((tensor - zero_points) / scales).clamp(0, n_levels).to(torch.int8)
    return codes, scales, zero_points


def quantize_kivi(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    n_bits: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KIVI quantization: per-channel keys, per-token values.

    This asymmetry matches the KIVI paper's observation that key vectors
    have outlier channels while value vectors have outlier tokens.

    Args:
        key_states: Shape [batch, n_kv_heads, seq_len, head_dim].
        value_states: Shape [batch, n_kv_heads, seq_len, head_dim].
        n_bits: Bit width.

    Returns:
        (dequantized_keys, dequantized_values).
    """
    if key_states.dim() != 4:
        raise ValueError(f"key_states must be 4D, got shape {key_states.shape}")
    if value_states.dim() != 4:
        raise ValueError(f"value_states must be 4D, got shape {value_states.shape}")

    # Keys: per-channel (dim 3 = head_dim)
    k_codes, k_scales, k_zp = quantize_per_channel(key_states, n_bits, channel_dim=3)
    dequant_keys = k_codes.float() * k_scales + k_zp

    # Values: per-token (dim 2 = seq_len)
    v_codes, v_scales, v_zp = quantize_per_channel(value_states, n_bits, channel_dim=2)
    dequant_values = v_codes.float() * v_scales + v_zp

    return dequant_keys, dequant_values
