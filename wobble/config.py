"""Centralized configuration for Wobble quantization. No magic numbers elsewhere."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Target model parameters. Auto-detected from model.config when possible."""

    model_name: str = "Qwen/Qwen3-1.7B"
    n_layers: int = 28
    n_kv_heads: int = 8
    head_dim: int = 128
    dtype: str = "bfloat16"


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
    """Profiling thresholds (go/no-go criteria)."""

    # Wobble detection: max/min variance ratio across dimensions
    dim_importance_variance_go: float = 10.0     # > 10:1 = strong signal
    dim_importance_variance_nogo: float = 2.0    # < 2:1 = wobble unlikely

    # Head diversity: Jensen-Shannon divergence between heads
    cross_head_js_go: float = 0.1     # > 0.1 nats = heads are different
    cross_head_js_nogo: float = 0.01  # < 0.01 = heads nearly identical


@dataclass
class QuantizationConfig:
    """Adaptive scalar quantization parameters.

    CRITICAL: target_bits must include non-integer values to activate
    adaptive allocation. At integer averages (2.0, 3.0), the greedy
    allocator gives every dimension the same bits (uniform).

    min_bits_per_dim must be >= 1. Setting to 0 replaces dims with their
    mean, which destroys too much information.
    """

    target_bits: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    max_groups_per_layer: int = 8
    js_divergence_threshold: float = 0.15
    group_size: int = 32        # per-group local scaling window
    max_bits_per_dim: int = 8
    min_bits_per_dim: int = 1   # MUST be >= 1, never 0


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
