"""LMdeploy platform implementation using OpenAI-compatible API."""

import base64
import logging
import time
from io import BytesIO

from PIL import Image

from .base import BasePlatform, InferenceResult, ModelInfo

# Import for type hints
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)


class LMdeployPlatform(BasePlatform):
    """LMdeploy platform implementation using OpenAI-compatible API."""

    @property
    def platform_name(self) -> str:
        return "lmdeploy"

    def _do_connect(self, server_url: str) -> bool:
        """Connect to LMdeploy server."""
        try:
            from openai import OpenAI, AsyncOpenAI

            # LMdeploy provides OpenAI-compatible API
            self._client = OpenAI(
                base_url=f"{server_url}/v1",
                api_key="not-needed",  # LMdeploy doesn't require real API key
                timeout=60.0,
            )
            # Also create async client for concurrent requests
            self._async_client = AsyncOpenAI(
                base_url=f"{server_url}/v1",
                api_key="not-needed",
                timeout=60.0,
            )
            return True
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        except Exception as e:
            logger.error(f"Failed to create LMdeploy client: {e}")
            return False

    def _do_health_check(self) -> bool:
        """Check if LMdeploy server is healthy."""
        try:
            models = self._client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"LMdeploy health check failed: {e}")
            return False

    def _encode_image(self, image: Image.Image, iteration: int = 0) -> str:
        """
        Convert PIL Image to base64 data URL with optional cache bypass.

        Args:
            image: PIL Image to encode
            iteration: Iteration number (used for cache bypass tweak)

        Returns:
            Base64 data URL string
        """
        # Apply cache bypass tweak if iteration > 0
        if iteration > 0:
            image = self._bypass_image_cache(image, iteration)

        buffer = BytesIO()
        # Use PNG for lossless encoding (preserves the tiny tweak)
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def _do_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        iteration: int = 0,
        **kwargs,
    ) -> InferenceResult:
        """Run inference on LMdeploy server."""
        image_url = self._encode_image(image, iteration=iteration)

        # Build messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        # Get model ID
        model_id = self.model_config.get("huggingface_id", "")

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        completion_tokens = 0

        try:
            # Use streaming to capture TTFT accurately
            stream = self._client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                stream=True,
            )

            for chunk in stream:
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000

                if chunk.choices and chunk.choices[0].delta.content:
                    output_text += chunk.choices[0].delta.content
                    completion_tokens += 1

            total_latency = (time.perf_counter() - start_time) * 1000

            # Calculate tokens per second
            generation_time = total_latency - (ttft or 0)
            tokens_per_second = (
                completion_tokens / (generation_time / 1000) if generation_time > 0 else 0
            )

            return InferenceResult(
                output_text=output_text,
                prompt_tokens=0,  # Not available in streaming mode
                completion_tokens=completion_tokens,
                total_tokens=completion_tokens,
                time_to_first_token_ms=ttft or 0,
                total_latency_ms=total_latency,
                tokens_per_second=tokens_per_second,
                success=True,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"LMdeploy inference error: {e}")
            return InferenceResult(
                output_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=total_latency,
                tokens_per_second=0,
                success=False,
                error_message=str(e),
            )

    def _do_get_model_info(self) -> ModelInfo:
        """Get model info from LMdeploy server."""
        try:
            models = self._client.models.list()
            model_id = models.data[0].id if models.data else "unknown"
        except Exception:
            model_id = self.model_config.get("huggingface_id", "unknown")

        return ModelInfo(
            model_id=model_id,
            platform=self.platform_name,
            quantization=self.platform_config.get("quantization"),
            max_model_len=self.platform_config.get("args", {}).get("max_model_len", 8192),
        )

    def inference_batch(
        self,
        images: list[tuple[str, Image.Image]],
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[tuple[str, InferenceResult]]:
        """
        Run inference on multiple images sequentially.

        Args:
            images: List of (filename, PIL.Image) tuples
            prompt: Text prompt
            max_new_tokens: Max tokens
            temperature: Temperature

        Returns:
            List of (filename, InferenceResult) tuples
        """
        results = []
        for filename, image in images:
            result = self.inference_with_retry(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append((filename, result))
        return results

    async def _do_inference_async(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        iteration: int = 0,
        **kwargs,
    ) -> InferenceResult:
        """Run async inference on LMdeploy server."""
        image_url = self._encode_image(image, iteration=iteration)

        # Build messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        # Get model ID
        model_id = self.model_config.get("huggingface_id", "")

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        completion_tokens = 0

        try:
            # Use async streaming to capture TTFT accurately
            stream = await self._async_client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                stream=True,
            )

            async for chunk in stream:
                if ttft is None:
                    ttft = (time.perf_counter() - start_time) * 1000

                if chunk.choices and chunk.choices[0].delta.content:
                    output_text += chunk.choices[0].delta.content
                    completion_tokens += 1

            total_latency = (time.perf_counter() - start_time) * 1000

            # Calculate tokens per second
            generation_time = total_latency - (ttft or 0)
            tokens_per_second = (
                completion_tokens / (generation_time / 1000) if generation_time > 0 else 0
            )

            return InferenceResult(
                output_text=output_text,
                prompt_tokens=0,  # Not available in streaming mode
                completion_tokens=completion_tokens,
                total_tokens=completion_tokens,
                time_to_first_token_ms=ttft or 0,
                total_latency_ms=total_latency,
                tokens_per_second=tokens_per_second,
                success=True,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"LMdeploy async inference error: {e}")
            return InferenceResult(
                output_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=total_latency,
                tokens_per_second=0,
                success=False,
                error_message=str(e),
            )
