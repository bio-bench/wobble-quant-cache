"""Streaming KV cache statistics capture from transformer models.

Uses Chan et al. parallel algorithm for numerically stable streaming
computation of mean, variance, and kurtosis. Never stores full KV
tensors -- processes one batch at a time via model forward passes.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class StatsAccumulator:
    """Online accumulator for per-dimension KV statistics.

    Uses Chan et al. parallel algorithm for numerically stable streaming
    computation of mean, variance, and kurtosis.
    """

    n_layers: int
    n_kv_heads: int
    head_dim: int

    k_count: np.ndarray = field(init=False)
    k_mean: np.ndarray = field(init=False)
    k_m2: np.ndarray = field(init=False)
    k_m4: np.ndarray = field(init=False)

    v_count: np.ndarray = field(init=False)
    v_mean: np.ndarray = field(init=False)
    v_m2: np.ndarray = field(init=False)
    v_m4: np.ndarray = field(init=False)

    v_norm_sum: np.ndarray = field(init=False)
    v_norm_count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        shape = (self.n_layers, self.n_kv_heads, self.head_dim)
        for prefix in ("k_", "v_"):
            setattr(self, f"{prefix}count", np.zeros(shape, dtype=np.int64))
            setattr(self, f"{prefix}mean", np.zeros(shape, dtype=np.float64))
            setattr(self, f"{prefix}m2", np.zeros(shape, dtype=np.float64))
            setattr(self, f"{prefix}m4", np.zeros(shape, dtype=np.float64))

        head_shape = (self.n_layers, self.n_kv_heads)
        self.v_norm_sum = np.zeros(head_shape, dtype=np.float64)
        self.v_norm_count = np.zeros(head_shape, dtype=np.int64)

    def _parallel_update(self, existing_count, existing_mean, existing_m2,
                         existing_m4, layer_idx, new_data):
        """Chan et al. parallel algorithm: merge new batch into running stats."""
        n_heads, seq_len, head_dim = new_data.shape

        batch_count = seq_len
        batch_mean = np.mean(new_data, axis=1)
        batch_var = np.var(new_data, axis=1, ddof=0)
        batch_m2 = batch_var * batch_count

        deviations = new_data - batch_mean[:, np.newaxis, :]
        batch_m4 = np.sum(deviations ** 4, axis=1)

        na = existing_count[layer_idx]
        nb = batch_count
        n_total = na + nb
        delta = batch_mean - existing_mean[layer_idx]
        safe_total = np.where(n_total > 0, n_total, 1)

        new_mean = existing_mean[layer_idx] + delta * nb / safe_total
        new_m2 = existing_m2[layer_idx] + batch_m2 + delta ** 2 * na * nb / safe_total
        new_m4 = (
            existing_m4[layer_idx] + batch_m4
            + 6.0 * delta ** 2 * na * nb * (na - nb) / (safe_total ** 2)
            + 4.0 * delta * (na * batch_m2 - nb * existing_m2[layer_idx]) / safe_total
        )

        existing_count[layer_idx] = n_total
        existing_mean[layer_idx] = new_mean
        existing_m2[layer_idx] = new_m2
        existing_m4[layer_idx] = new_m4

    def update(self, layer_idx: int, key_states: torch.Tensor,
               value_states: torch.Tensor) -> None:
        """Update running statistics with new KV tensors.

        Args:
            layer_idx: Which transformer layer.
            key_states: Shape [batch, n_kv_heads, seq_len, head_dim].
            value_states: Shape [batch, n_kv_heads, seq_len, head_dim].
        """
        k_np = key_states.squeeze(0).float().cpu().numpy()
        v_np = value_states.squeeze(0).float().cpu().numpy()

        self._parallel_update(self.k_count, self.k_mean, self.k_m2, self.k_m4,
                              layer_idx, k_np)
        self._parallel_update(self.v_count, self.v_mean, self.v_m2, self.v_m4,
                              layer_idx, v_np)

        v_norms = np.linalg.norm(v_np, axis=-1)
        self.v_norm_sum[layer_idx] += np.sum(v_norms, axis=-1)
        self.v_norm_count[layer_idx] += v_np.shape[1]

    def finalize(self) -> dict[str, np.ndarray]:
        """Compute final statistics from accumulated values.

        Returns dict with keys: k_variance, k_kurtosis, k_mean, k_count,
        v_variance, v_kurtosis, v_mean, v_count, v_mean_norm.
        """
        results = {}
        for prefix, count, mean, m2, m4 in [
            ("k", self.k_count, self.k_mean, self.k_m2, self.k_m4),
            ("v", self.v_count, self.v_mean, self.v_m2, self.v_m4),
        ]:
            safe_count = np.where(count > 1, count, 2)
            variance = m2 / (safe_count - 1)

            safe_m2 = np.where(m2 > 0, m2, 1.0)
            kurtosis = (m4 * count) / (safe_m2 ** 2) - 3.0
            kurtosis = np.clip(kurtosis, -10.0, 1000.0)

            results[f"{prefix}_variance"] = variance
            results[f"{prefix}_kurtosis"] = kurtosis
            results[f"{prefix}_mean"] = mean
            results[f"{prefix}_count"] = count

        safe_norm_count = np.where(self.v_norm_count > 0, self.v_norm_count, 1)
        results["v_mean_norm"] = self.v_norm_sum / safe_norm_count
        return results


class ReservoirSampler:
    """Reservoir sampling to collect a fixed-size subsample of KV vectors.

    Used for distribution fitting and effective rank computation.
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 reservoir_size: int = 4096, seed: int = 42) -> None:
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.key_reservoir = np.zeros(
            (n_layers, n_kv_heads, reservoir_size, head_dim), dtype=np.float32)
        self.value_reservoir = np.zeros(
            (n_layers, n_kv_heads, reservoir_size, head_dim), dtype=np.float32)
        self.counts = np.zeros((n_layers, n_kv_heads), dtype=np.int64)
        self.stored = np.zeros((n_layers, n_kv_heads), dtype=np.int64)

    def update(self, layer_idx: int, key_states: np.ndarray,
               value_states: np.ndarray) -> None:
        """Add a batch of KV vectors. key/value shape: [n_kv_heads, seq_len, head_dim]."""
        n_heads, seq_len, _ = key_states.shape
        for h in range(n_heads):
            for t in range(seq_len):
                n = self.counts[layer_idx, h]
                if n < self.reservoir_size:
                    idx = n
                    self.key_reservoir[layer_idx, h, idx] = key_states[h, t]
                    self.value_reservoir[layer_idx, h, idx] = value_states[h, t]
                    self.stored[layer_idx, h] = n + 1
                else:
                    j = self.rng.integers(0, n + 1)
                    if j < self.reservoir_size:
                        self.key_reservoir[layer_idx, h, j] = key_states[h, t]
                        self.value_reservoir[layer_idx, h, j] = value_states[h, t]
                self.counts[layer_idx, h] = n + 1

    def get_samples(self, layer_idx: int, head_idx: int,
                    cache_type: str = "key") -> np.ndarray:
        """Get stored samples. Returns shape [n_stored, head_dim]."""
        n = int(self.stored[layer_idx, head_idx])
        reservoir = self.key_reservoir if cache_type == "key" else self.value_reservoir
        return reservoir[layer_idx, head_idx, :n].copy()


def extract_kv_from_output(past_key_values):
    """Extract (key, value) tensors from model output's past_key_values.

    Handles multiple HuggingFace cache formats.

    Returns:
        List of (key_states, value_states) per layer.
        Each tensor shape: [batch, n_kv_heads, seq_len, head_dim].
    """
    if hasattr(past_key_values, "key_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "layers"):
        return [(layer.keys, layer.values) for layer in past_key_values.layers]

    if isinstance(past_key_values, (tuple, list)):
        return [(kv[0], kv[1]) for kv in past_key_values]

    raise TypeError(f"Unexpected past_key_values type: {type(past_key_values)}")


def profile_kv_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    max_seq_length: int = 2048,
    reservoir_size: int = 4096,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], ReservoirSampler]:
    """Run profiling over calibration texts, accumulating KV statistics.

    Uses streaming accumulation -- never stores full KV tensors.

    Args:
        model: Loaded model on GPU.
        tokenizer: Corresponding tokenizer.
        texts: Calibration texts.
        n_layers: Number of transformer layers.
        n_kv_heads: Number of KV heads.
        head_dim: Head dimension.
        max_seq_length: Maximum sequence length per text.
        reservoir_size: Samples to keep per (layer, head) for distribution fitting.
        seed: Random seed.

    Returns:
        (stats_dict, reservoir_sampler).
    """
    accumulator = StatsAccumulator(n_layers=n_layers, n_kv_heads=n_kv_heads,
                                   head_dim=head_dim)
    reservoir = ReservoirSampler(n_layers=n_layers, n_kv_heads=n_kv_heads,
                                 head_dim=head_dim, reservoir_size=reservoir_size,
                                 seed=seed)

    device = next(model.parameters()).device
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_texts = len(texts)
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0:
            logger.info("Profiling text %d / %d", i + 1, n_texts)

        inputs = tokenizer(text, return_tensors="pt", max_length=max_seq_length,
                           truncation=True, padding=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, return_dict=True)

        kv_pairs = extract_kv_from_output(outputs.past_key_values)
        for layer_idx, (key_states, value_states) in enumerate(kv_pairs):
            accumulator.update(layer_idx, key_states, value_states)
            k_np = key_states.squeeze(0).float().cpu().numpy()
            v_np = value_states.squeeze(0).float().cpu().numpy()
            reservoir.update(layer_idx, k_np, v_np)

        del outputs, kv_pairs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stats = accumulator.finalize()
    logger.info("Profiling complete: %d texts, %d layers", n_texts, n_layers)
    return stats, reservoir
