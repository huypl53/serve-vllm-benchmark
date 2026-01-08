"""vLLM platform implementation using OpenAI-compatible API."""

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


class VLLMPlatform(BasePlatform):
    """vLLM platform implementation using OpenAI-compatible API."""

    @property
    def platform_name(self) -> str:
        return "vllm"

    def _do_connect(self, server_url: str) -> bool:
        """Connect to vLLM server."""
        try:
            from openai import OpenAI, AsyncOpenAI

            # vLLM provides OpenAI-compatible API
            self._client = OpenAI(
                base_url=f"{server_url}/v1",
                api_key="not-needed",  # vLLM doesn't require real API key
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
            logger.error(f"Failed to create vLLM client: {e}")
            return False

    def _do_health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            models = self._client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.debug(f"vLLM health check failed: {e}")
            return False

    def _bypass_image_cache(self, image: Image.Image, iteration: int) -> Image.Image:
        """
        Apply invisible pixel tweak to bypass vLLM's image caching.

        This modifies a single pixel by a tiny amount (±1 RGB value) that is
        imperceptible to the model but changes the image hash, preventing caching.

        Args:
            image: PIL Image to modify
            iteration: Iteration number (determines which pixel to tweak)

        Returns:
            Modified PIL Image (copy, original is not changed)
        """
        # Create a copy to avoid modifying the original
        tweaked = image.copy()

        # Convert to RGB if needed
        if tweaked.mode != "RGB":
            tweaked = tweaked.convert("RGB")

        # Load pixel data
        pixels = tweaked.load()

        # Use iteration number to deterministically select a pixel
        # This ensures each iteration gets a different tweak
        width, height = tweaked.size
        pixel_x = (iteration * 7) % width  # Prime number for better distribution
        pixel_y = (iteration * 13) % height

        # Get current pixel value
        r, g, b = pixels[pixel_x, pixel_y]

        # Apply tiny tweak (±1 to one channel)
        # Using a deterministic pattern based on iteration
        tweak_amount = 1 if (iteration % 2) == 0 else -1
        channel = iteration % 3  # 0=R, 1=G, 2=B

        if channel == 0:
            r = max(0, min(255, r + tweak_amount))
        elif channel == 1:
            g = max(0, min(255, g + tweak_amount))
        else:
            b = max(0, min(255, b + tweak_amount))

        # Set the tweaked pixel
        pixels[pixel_x, pixel_y] = (r, g, b)

        return tweaked

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
        """Run inference on vLLM server."""
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
            logger.error(f"vLLM inference error: {e}")
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
        """Get model info from vLLM server."""
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
        """Run async inference on vLLM server."""
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
            logger.error(f"vLLM async inference error: {e}")
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
