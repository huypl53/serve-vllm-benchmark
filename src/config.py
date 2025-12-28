"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkSettings:
    """Benchmark execution settings."""

    warmup_iterations: int = 2
    num_iterations: int = 3
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    inference_timeout: int = 120
    max_retries: int = 3
    retry_delay_base: float = 2.0


@dataclass
class ServerSettings:
    """Server connection settings."""

    startup_timeout: int = 300
    health_check_interval: int = 5
    ports: dict = field(default_factory=lambda: {"vllm": 8000, "tgi": 8080, "tensorrt": 8001})


@dataclass
class GPUSettings:
    """GPU monitoring settings."""

    monitor_interval_ms: int = 100
    target_gpus: list = field(default_factory=lambda: [0])


@dataclass
class OutputSettings:
    """Output configuration."""

    results_dir: str = "./results"
    save_model_outputs: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"


@dataclass
class PromptSettings:
    """Prompt templates."""

    default: str = "Describe this image in detail."
    alternatives: dict = field(default_factory=dict)


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    log_file: str = "./results/benchmark.log"
    json_format: bool = False


@dataclass
class BenchmarkConfig:
    """Main configuration container."""

    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    gpu: GPUSettings = field(default_factory=GPUSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    prompts: PromptSettings = field(default_factory=PromptSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


@dataclass
class PlatformConfig:
    """Platform-specific configuration."""

    supported: bool = False
    args: dict = field(default_factory=dict)
    backend: str = "default"


@dataclass
class ModelConfig:
    """Model configuration."""

    huggingface_id: str
    display_name: str
    memory_estimate_gb: int
    min_vram_gb: int
    platforms: dict = field(default_factory=dict)
    quantization_for_t4: str | None = None


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return BenchmarkConfig(
        benchmark=BenchmarkSettings(**data.get("benchmark", {})),
        server=ServerSettings(**data.get("server", {})),
        gpu=GPUSettings(**data.get("gpu", {})),
        output=OutputSettings(**data.get("output", {})),
        prompts=PromptSettings(**data.get("prompts", {})),
        logging=LoggingSettings(**data.get("logging", {})),
    )


def load_models_config(config_path: str | Path) -> dict[str, ModelConfig]:
    """Load models configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Models config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    models = {}
    for model_id, model_data in data.get("models", {}).items():
        models[model_id] = ModelConfig(
            huggingface_id=model_data["huggingface_id"],
            display_name=model_data.get("display_name", model_id),
            memory_estimate_gb=model_data.get("memory_estimate_gb", 16),
            min_vram_gb=model_data.get("min_vram_gb", 16),
            platforms=model_data.get("platforms", {}),
            quantization_for_t4=model_data.get("quantization_for_t4"),
        )

    return models


def get_gpu_profile(config_path: str | Path) -> dict[str, Any]:
    """Load GPU profiles from config."""
    config_path = Path(config_path)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return data.get("gpu_profiles", {})
