"""Inference platform implementations."""

from .base import BasePlatform, InferenceResult, ModelInfo
from .vllm_platform import VLLMPlatform
from .tgi_platform import TGIPlatform
from .tensorrt_platform import TensorRTPlatform
from .lmdeploy_platform import LMdeployPlatform

__all__ = [
    "BasePlatform",
    "InferenceResult",
    "ModelInfo",
    "VLLMPlatform",
    "TGIPlatform",
    "TensorRTPlatform",
    "LMdeployPlatform",
]

# Platform registry for dynamic instantiation
PLATFORM_REGISTRY = {
    "vllm": VLLMPlatform,
    "tgi": TGIPlatform,
    "tensorrt": TensorRTPlatform,
    "lmdeploy": LMdeployPlatform,
}


def get_platform_class(platform_name: str) -> type[BasePlatform]:
    """Get platform class by name."""
    if platform_name not in PLATFORM_REGISTRY:
        raise ValueError(f"Unknown platform: {platform_name}. Available: {list(PLATFORM_REGISTRY.keys())}")
    return PLATFORM_REGISTRY[platform_name]
