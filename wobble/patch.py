"""Model-specific attention patching for post-RoPE KV cache quantization.

The monkey-patching approach overrides each attention layer's forward
method to inject quantization AFTER rotary position embeddings are applied.
This ensures profiling statistics match the actual quantization point.

Supported models:
- Mistral (mistralai/Mistral-7B-*)
- Gemma 2 (google/gemma-2-*)

When adding a new model, implement a patch function that:
1. Imports the correct apply_rotary_pos_emb and eager_attention_forward
2. Handles model-specific attention kwargs (softcap, sliding_window, etc.)
"""

import logging
import types
from collections import defaultdict

import torch

from wobble.quantize import AdaptiveScalarConfig, encode, decode
from wobble.baselines import quantize_kivi, quantize_uniform, dequantize_uniform

logger = logging.getLogger(__name__)


def quantize_wobble(key_states, value_states, layer_idx,
                    all_configs, all_h2g, group_size=32):
    """Apply Wobble adaptive quantization to K/V tensors.

    Args:
        key_states: [batch, n_kv_heads, seq_len, head_dim].
        value_states: [batch, n_kv_heads, seq_len, head_dim].
        layer_idx: Layer index.
        all_configs: {layer: {group_id: AdaptiveScalarConfig}}.
        all_h2g: {layer: {head_idx: group_id}}.
        group_size: Per-group local scaling window.

    Returns:
        (quantized_keys, quantized_values).
    """
    B, H, S, D = key_states.shape
    q_keys = torch.zeros_like(key_states, dtype=torch.float32)
    q_vals = torch.zeros_like(value_states, dtype=torch.float32)

    group_to_heads = defaultdict(list)
    for h, g in all_h2g[layer_idx].items():
        group_to_heads[g].append(h)

    for gid, heads in group_to_heads.items():
        cfg = all_configs[layer_idx][gid]
        for h in heads:
            k_flat = key_states[:, h, :, :].reshape(-1, D)
            k_codes, k_gm, k_gx = encode(k_flat, cfg, group_size=group_size)
            q_keys[:, h, :, :] = decode(
                k_codes, cfg, k_gm, k_gx, group_size=group_size
            ).reshape(B, S, D)

            v_flat = value_states[:, h, :, :].reshape(-1, D)
            v_codes, v_gm, v_gx = encode(v_flat, cfg, group_size=group_size)
            q_vals[:, h, :, :] = decode(
                v_codes, cfg, v_gm, v_gx, group_size=group_size
            ).reshape(B, S, D)

    return q_keys.to(key_states.dtype), q_vals.to(value_states.dtype)


def quantize_kivi_wrapper(key_states, value_states, layer_idx, n_bits=2):
    """KIVI baseline wrapper matching the quantize_fn signature."""
    dk, dv = quantize_kivi(key_states.float(), value_states.float(), n_bits)
    return dk.to(key_states.dtype), dv.to(value_states.dtype)


def quantize_static_wrapper(key_states, value_states, layer_idx, n_bits=2):
    """Static uniform baseline wrapper matching the quantize_fn signature."""
    k_codes, k_s, k_zp = quantize_uniform(key_states.float(), n_bits)
    dk = dequantize_uniform(k_codes, k_s, k_zp).to(key_states.dtype)
    v_codes, v_s, v_zp = quantize_uniform(value_states.float(), n_bits)
    dv = dequantize_uniform(v_codes, v_s, v_zp).to(value_states.dtype)
    return dk, dv


# ---------------------------------------------------------------------------
# Mistral attention patching
# ---------------------------------------------------------------------------

def patch_mistral(model, quantize_fn):
    """Patch Mistral model attention to inject post-RoPE quantization.

    Args:
        model: A MistralForCausalLM model.
        quantize_fn: Callable(key_states, value_states, layer_idx) -> (qk, qv).

    Returns:
        List of (attn_module, original_forward) for restoration.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.mistral.modeling_mistral import (
        apply_rotary_pos_emb, eager_attention_forward,
    )

    originals = []
    for layer in model.model.layers:
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        li = attn.layer_idx

        def make_fwd(layer_idx, q_fn):
            def fwd(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                # Post-RoPE quantization
                key_states, value_states = q_fn(key_states, value_states, layer_idx)

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos,
                                    "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs)

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward)
                attn_output, attn_weights = attention_interface(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=getattr(self.config, "sliding_window", None),
                    **kwargs)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = self.o_proj(attn_output)
                return attn_output, attn_weights
            return fwd

        attn.forward = types.MethodType(make_fwd(li, quantize_fn), attn)

    return originals


# ---------------------------------------------------------------------------
# Gemma 2 attention patching
# ---------------------------------------------------------------------------

def patch_gemma2(model, quantize_fn):
    """Patch Gemma-2 model attention to inject post-RoPE quantization.

    Handles Gemma-2 specific features: attention softcapping, per-layer
    sliding window.

    Args:
        model: A Gemma2ForCausalLM model.
        quantize_fn: Callable(key_states, value_states, layer_idx) -> (qk, qv).

    Returns:
        List of (attn_module, original_forward) for restoration.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma2.modeling_gemma2 import (
        apply_rotary_pos_emb, eager_attention_forward,
    )

    originals = []
    for layer in model.model.layers:
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        li = attn.layer_idx

        def make_fwd(layer_idx, q_fn):
            def fwd(self, hidden_states, position_embeddings, attention_mask=None,
                    past_key_values=None, cache_position=None, **kwargs):
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)

                query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin)

                # Post-RoPE quantization
                key_states, value_states = q_fn(key_states, value_states, layer_idx)

                if past_key_values is not None:
                    cache_kwargs = {"sin": sin, "cos": cos,
                                    "cache_position": cache_position}
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, self.layer_idx, cache_kwargs)

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    self.config._attn_implementation, eager_attention_forward)
                attn_output, attn_weights = attention_interface(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=getattr(self, "sliding_window", None),
                    softcap=getattr(self, "attn_logit_softcapping", None),
                    **kwargs)

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                return self.o_proj(attn_output), attn_weights
            return fwd

        attn.forward = types.MethodType(make_fwd(li, quantize_fn), attn)

    return originals


# ---------------------------------------------------------------------------
# Generic interface
# ---------------------------------------------------------------------------

_MODEL_PATCHERS = {
    "mistral": patch_mistral,
    "gemma2": patch_gemma2,
}


def patch_model(model, quantize_fn, architecture: str | None = None):
    """Patch model attention to inject post-RoPE quantization.

    Auto-detects architecture from model class name if not specified.

    Args:
        model: A HuggingFace causal LM model.
        quantize_fn: Callable(key_states, value_states, layer_idx) -> (qk, qv).
        architecture: "mistral" or "gemma2". Auto-detected if None.

    Returns:
        List of (attn_module, original_forward) for restoration via restore_model().
    """
    if architecture is None:
        class_name = type(model).__name__.lower()
        if "mistral" in class_name:
            architecture = "mistral"
        elif "gemma2" in class_name or "gemma" in class_name:
            architecture = "gemma2"
        else:
            raise ValueError(
                f"Cannot auto-detect architecture from {type(model).__name__}. "
                f"Supported: {list(_MODEL_PATCHERS.keys())}. "
                f"Pass architecture= explicitly."
            )

    patcher = _MODEL_PATCHERS.get(architecture)
    if patcher is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported: {list(_MODEL_PATCHERS.keys())}"
        )

    logger.info("Patching %s model for post-RoPE quantization", architecture)
    return patcher(model, quantize_fn)


def restore_model(originals):
    """Restore original attention forward methods.

    Args:
        originals: List of (attn_module, original_forward) from patch_model().
    """
    for attn, orig_forward in originals:
        attn.forward = orig_forward
