"""Centralized configuration for Wobble quantization. No magic numbers elsewhere."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Target model parameters."""

    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    n_layers: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "float16"


@dataclass
class CalibrationConfig:
    """Calibration data parameters."""

    wikitext_tokens: int = 10_000
    c4_tokens: int = 10_000
    pubmed_tokens: int = 5_000
    max_seq_length: int = 2048

    @property
    def total_tokens(self) -> int:
        return self.wikitext_tokens + self.c4_tokens + self.pubmed_tokens


@dataclass
class ProfilingConfig:
    """Stage 1 profiling thresholds (go/no-go criteria)."""

    # Wobble detection: max/min variance ratio across dimensions
    dim_importance_variance_go: float = 10.0     # > 10:1 = strong signal
    dim_importance_variance_nogo: float = 2.0    # < 2:1 = wobble unlikely

    # Head diversity: Jensen-Shannon divergence between heads
    cross_head_js_go: float = 0.1     # > 0.1 nats = heads are different
    cross_head_js_nogo: float = 0.01  # < 0.01 = heads nearly identical

    n_tiers: int = 3


@dataclass
class QuantizationConfig:
    """Adaptive scalar quantization parameters."""

    target_bits: list[float] = field(default_factory=lambda: [2.0, 3.0])
    max_groups_per_layer: int = 8
    js_divergence_threshold: float = 0.15
    group_size: int = 32        # per-group local scaling window
    max_bits_per_dim: int = 8
    min_bits_per_dim: int = 0   # 0 = use mean (full degeneracy)


@dataclass
class EvalConfig:
    """Evaluation parameters."""

    max_tokens: int = 50_000
    max_seq_length: int = 2048
    stride: int = 512


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    device: str = "cuda"
    seed: int = 42
