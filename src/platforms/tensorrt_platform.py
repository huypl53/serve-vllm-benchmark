"""TensorRT-LLM platform implementation via Triton Server or trtllm-serve."""

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


class TensorRTPlatform(BasePlatform):
    """TensorRT-LLM platform implementation."""

    @property
    def platform_name(self) -> str:
        return "tensorrt"

    def _do_connect(self, server_url: str) -> bool:
        """Connect to TensorRT-LLM server (Triton or trtllm-serve)."""
        self._server_url = server_url.rstrip("/")

        # Detect server type (Triton vs trtllm-serve)
        self._server_type = self._detect_server_type()
        logger.info(f"Detected TensorRT server type: {self._server_type}")

        return True

    def _detect_server_type(self) -> str:
        """Detect if server is Triton or trtllm-serve."""
        # Try Triton health endpoint first
        try:
            response = requests.get(f"{self._server_url}/v2/health/ready", timeout=5)
            if response.status_code == 200:
                return "triton"
        except Exception:
            pass

        # Try OpenAI-compatible endpoint (trtllm-serve)
        try:
            response = requests.get(f"{self._server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                return "openai"
        except Exception:
            pass

        return "unknown"

    def _do_health_check(self) -> bool:
        """Check if TensorRT-LLM server is healthy."""
        try:
            if self._server_type == "triton":
                response = requests.get(f"{self._server_url}/v2/health/ready", timeout=10)
                return response.status_code == 200
            elif self._server_type == "openai":
                response = requests.get(f"{self._server_url}/v1/models", timeout=10)
                return response.status_code == 200
            else:
                # Try both
                for endpoint in ["/v2/health/ready", "/v1/models", "/health"]:
                    try:
                        response = requests.get(f"{self._server_url}{endpoint}", timeout=5)
                        if response.status_code == 200:
                            return True
                    except Exception:
                        continue
                return False
        except Exception as e:
            logger.debug(f"TensorRT health check failed: {e}")
            return False

    def _encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _do_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        iteration: int = 0,
        **kwargs,
    ) -> InferenceResult:
        """Run inference on TensorRT-LLM server."""
        if self._server_type == "openai":
            return self._inference_openai_compat(
                image, prompt, max_new_tokens, temperature
            )
        elif self._server_type == "triton":
            return self._inference_triton(image, prompt, max_new_tokens, temperature)
        else:
            # Try OpenAI-compatible first, then Triton
            try:
                return self._inference_openai_compat(
                    image, prompt, max_new_tokens, temperature
                )
            except Exception:
                return self._inference_triton(image, prompt, max_new_tokens, temperature)

    def _inference_openai_compat(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> InferenceResult:
        """Run inference using OpenAI-compatible API (trtllm-serve)."""
        image_b64 = self._encode_image(image)
        image_url = f"data:image/png;base64,{image_b64}"

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
            "model": self.model_config.get("huggingface_id", "model"),
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.perf_counter()
        ttft = None
        output_text = ""
        completion_tokens = 0

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

    def _inference_triton(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> InferenceResult:
        """Run inference using Triton Inference Server."""
        image_b64 = self._encode_image(image)

        start_time = time.perf_counter()

        # Triton inference request format
        # Note: Actual payload format depends on model configuration
        payload = {
            "inputs": [
                {
                    "name": "text_input",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt],
                },
                {
                    "name": "image",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [image_b64],
                },
                {
                    "name": "max_tokens",
                    "shape": [1],
                    "datatype": "INT32",
                    "data": [max_new_tokens],
                },
            ]
        }

        # Try different model names
        model_names = ["ensemble", "tensorrt_llm", "vllm_model"]

        for model_name in model_names:
            try:
                response = requests.post(
                    f"{self._server_url}/v2/models/{model_name}/infer",
                    json=payload,
                    timeout=120,
                )
                if response.status_code == 200:
                    break
            except Exception:
                continue
        else:
            raise RuntimeError("Failed to find valid model endpoint on Triton server")

        total_latency = (time.perf_counter() - start_time) * 1000

        result = response.json()
        output_text = ""

        # Extract output from Triton response
        for output in result.get("outputs", []):
            if output.get("name") in ["text_output", "output"]:
                data = output.get("data", [])
                if data:
                    output_text = data[0] if isinstance(data[0], str) else str(data[0])
                break

        # Estimate tokens (Triton may not return token counts)
        completion_tokens = len(output_text.split()) if output_text else 0

        return InferenceResult(
            output_text=output_text,
            prompt_tokens=0,
            completion_tokens=completion_tokens,
            total_tokens=completion_tokens,
            time_to_first_token_ms=0,  # Batch inference, no streaming
            total_latency_ms=total_latency,
            tokens_per_second=completion_tokens / (total_latency / 1000) if total_latency > 0 else 0,
            success=True,
        )

    def _do_get_model_info(self) -> ModelInfo:
        """Get model info from TensorRT-LLM server."""
        try:
            if self._server_type == "openai":
                response = requests.get(f"{self._server_url}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data"):
                        return ModelInfo(
                            model_id=data["data"][0].get("id", "unknown"),
                            platform=self.platform_name,
                        )
            elif self._server_type == "triton":
                response = requests.get(f"{self._server_url}/v2/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("models"):
                        return ModelInfo(
                            model_id=data["models"][0].get("name", "unknown"),
                            platform=self.platform_name,
                        )
        except Exception as e:
            logger.debug(f"Failed to get TensorRT model info: {e}")

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
        iteration: int = 0,
        **kwargs,
    ) -> InferenceResult:
        """Run async inference on TensorRT-LLM server using httpx."""
        if not HTTPX_AVAILABLE:
            # Fallback to base class executor-based implementation
            return await super()._do_inference_async(
                image, prompt, max_new_tokens, temperature, iteration, **kwargs
            )

        # Only OpenAI-compatible endpoint supports async streaming
        if self._server_type != "openai":
            return await super()._do_inference_async(
                image, prompt, max_new_tokens, temperature, iteration, **kwargs
            )

        image_b64 = self._encode_image(image)
        image_url = f"data:image/png;base64,{image_b64}"

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
            "model": self.model_config.get("huggingface_id", "model"),
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
            logger.error(f"TensorRT async inference error: {e}")
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
