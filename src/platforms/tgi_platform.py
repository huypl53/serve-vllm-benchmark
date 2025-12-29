"""TGI (Text Generation Inference) platform implementation."""

import base64
import json
import logging
import time
from io import BytesIO

import requests
from PIL import Image

from .base import BasePlatform, InferenceResult, ModelInfo

# Optional httpx for async support
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class TGIPlatform(BasePlatform):
    """Text Generation Inference platform implementation."""

    @property
    def platform_name(self) -> str:
        return "tgi"

    def _do_connect(self, server_url: str) -> bool:
        """Connect to TGI server."""
        self._server_url = server_url.rstrip("/")

        # TGI uses huggingface_hub InferenceClient or direct HTTP
        try:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(base_url=self._server_url)
            self._use_hf_client = True
        except ImportError:
            logger.warning("huggingface_hub not available, using direct HTTP")
            self._use_hf_client = False

        return True

    def _do_health_check(self) -> bool:
        """Check if TGI server is healthy."""
        try:
            response = requests.get(f"{self._server_url}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"TGI health check failed: {e}")
            return False

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_data}"

    def _do_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> InferenceResult:
        """Run inference on TGI server."""
        image_url = self._encode_image(image)

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        completion_tokens = 0

        try:
            if self._use_hf_client:
                return self._inference_with_hf_client(
                    image_url, prompt, max_new_tokens, temperature, start_time
                )
            else:
                return self._inference_with_http(
                    image_url, prompt, max_new_tokens, temperature, start_time
                )
        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"TGI inference error: {e}")
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

    def _inference_with_hf_client(
        self,
        image_url: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        start_time: float,
    ) -> InferenceResult:
        """Run inference using HuggingFace InferenceClient."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        ttft = None
        output_text = ""
        completion_tokens = 0

        # Use streaming for accurate TTFT
        stream = self._client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            if ttft is None:
                ttft = (time.perf_counter() - start_time) * 1000

            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                output_text += chunk.choices[0].delta.content
                completion_tokens += 1

        total_latency = (time.perf_counter() - start_time) * 1000
        generation_time = total_latency - (ttft or 0)
        tokens_per_second = (
            completion_tokens / (generation_time / 1000) if generation_time > 0 else 0
        )

        return InferenceResult(
            output_text=output_text,
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            time_to_first_token_ms=ttft or 0,
            total_latency_ms=total_latency,
            tokens_per_second=tokens_per_second,
            success=True,
        )

    def _inference_with_http(
        self,
        image_url: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        start_time: float,
    ) -> InferenceResult:
        """Run inference using direct HTTP requests."""
        # TGI v2 chat completions endpoint
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        payload = {
            "model": "tgi",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        ttft = None
        output_text = ""
        completion_tokens = 0

        # Stream response
        response = requests.post(
            f"{self._server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    if ttft is None:
                        ttft = (time.perf_counter() - start_time) * 1000

                    data = line[6:]
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get(
                            "content"
                        ):
                            output_text += chunk["choices"][0]["delta"]["content"]
                            completion_tokens += 1
                    except json.JSONDecodeError:
                        continue

        total_latency = (time.perf_counter() - start_time) * 1000
        generation_time = total_latency - (ttft or 0)
        tokens_per_second = (
            completion_tokens / (generation_time / 1000) if generation_time > 0 else 0
        )

        return InferenceResult(
            output_text=output_text,
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            time_to_first_token_ms=ttft or 0,
            total_latency_ms=total_latency,
            tokens_per_second=tokens_per_second,
            success=True,
        )

    def _do_get_model_info(self) -> ModelInfo:
        """Get model info from TGI server."""
        try:
            response = requests.get(f"{self._server_url}/info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                return ModelInfo(
                    model_id=info.get("model_id", "unknown"),
                    platform=self.platform_name,
                    quantization=info.get("quantization"),
                    max_model_len=info.get("max_input_length", 8192),
                )
        except Exception as e:
            logger.debug(f"Failed to get TGI model info: {e}")

        return ModelInfo(
            model_id=self.model_config.get("huggingface_id", "unknown"),
            platform=self.platform_name,
        )

    async def _do_inference_async(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> InferenceResult:
        """Run async inference on TGI server using httpx."""
        if not HTTPX_AVAILABLE:
            # Fallback to base class executor-based implementation
            return await super()._do_inference_async(
                image, prompt, max_new_tokens, temperature, **kwargs
            )

        image_url = self._encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ]

        payload = {
            "model": "tgi",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        completion_tokens = 0

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._server_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            if ttft is None:
                                ttft = (time.perf_counter() - start_time) * 1000

                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)
                                if chunk.get("choices") and chunk["choices"][0].get(
                                    "delta", {}
                                ).get("content"):
                                    output_text += chunk["choices"][0]["delta"]["content"]
                                    completion_tokens += 1
                            except json.JSONDecodeError:
                                continue

            total_latency = (time.perf_counter() - start_time) * 1000
            generation_time = total_latency - (ttft or 0)
            tokens_per_second = (
                completion_tokens / (generation_time / 1000) if generation_time > 0 else 0
            )

            return InferenceResult(
                output_text=output_text,
                prompt_tokens=0,
                completion_tokens=completion_tokens,
                total_tokens=completion_tokens,
                time_to_first_token_ms=ttft or 0,
                total_latency_ms=total_latency,
                tokens_per_second=tokens_per_second,
                success=True,
            )

        except Exception as e:
            total_latency = (time.perf_counter() - start_time) * 1000
            logger.error(f"TGI async inference error: {e}")
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
