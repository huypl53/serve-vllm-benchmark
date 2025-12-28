"""Abstract base class for inference platforms with retry logic."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    output_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_second: float
    success: bool = True
    error_message: str | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class ModelInfo:
    """Model metadata."""

    model_id: str
    platform: str
    quantization: str | None = None
    max_model_len: int = 4096


class InferenceError(Exception):
    """Custom exception for inference errors."""

    pass


class ServerNotReadyError(Exception):
    """Exception raised when server is not ready."""

    pass


class BasePlatform(ABC):
    """Abstract base class for inference platforms with built-in retry logic."""

    def __init__(
        self,
        model_config: dict,
        platform_config: dict | None = None,
        max_retries: int = 3,
        retry_delay_base: float = 2.0,
    ):
        """
        Initialize platform.

        Args:
            model_config: Model configuration dict
            platform_config: Platform-specific configuration
            max_retries: Maximum retry attempts
            retry_delay_base: Base delay for exponential backoff
        """
        self.model_config = model_config
        self.platform_config = platform_config or {}
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self._client = None
        self._server_url: str | None = None
        self._is_connected = False

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform identifier."""
        pass

    @abstractmethod
    def _do_connect(self, server_url: str) -> bool:
        """
        Internal connection implementation.

        Args:
            server_url: URL of the inference server

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def _do_health_check(self) -> bool:
        """
        Internal health check implementation.

        Returns:
            True if server is healthy
        """
        pass

    @abstractmethod
    def _do_inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        **kwargs,
    ) -> InferenceResult:
        """
        Internal inference implementation.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            InferenceResult with timing and output
        """
        pass

    @abstractmethod
    def _do_get_model_info(self) -> ModelInfo:
        """Internal model info implementation."""
        pass

    def connect(self, server_url: str, timeout: int = 300) -> bool:
        """
        Connect to inference server with retry and timeout.

        Args:
            server_url: URL of the inference server
            timeout: Maximum seconds to wait for server

        Returns:
            True if connected successfully

        Raises:
            ServerNotReadyError: If server not ready within timeout
        """
        self._server_url = server_url
        start_time = time.time()
        last_error = None

        logger.info(f"Connecting to {self.platform_name} at {server_url}...")

        while time.time() - start_time < timeout:
            try:
                if self._do_connect(server_url):
                    if self.health_check():
                        self._is_connected = True
                        logger.info(f"Connected to {self.platform_name} successfully")
                        return True
            except Exception as e:
                last_error = e
                logger.debug(f"Connection attempt failed: {e}")

            # Wait before retry
            time.sleep(5)

        raise ServerNotReadyError(
            f"Failed to connect to {self.platform_name} at {server_url} "
            f"within {timeout}s. Last error: {last_error}"
        )

    def health_check(self) -> bool:
        """
        Check if server is healthy.

        Returns:
            True if healthy
        """
        try:
            return self._do_health_check()
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    @retry(
        retry=retry_if_exception_type((InferenceError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def inference(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> InferenceResult:
        """
        Run inference with automatic retry on transient failures.

        Args:
            image: PIL Image to analyze
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InferenceResult with timing and output
        """
        if not self._is_connected:
            raise ConnectionError(f"{self.platform_name} is not connected")

        try:
            return self._do_inference(
                image=image,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Inference failed on {self.platform_name}: {e}")
            # Return error result instead of raising for graceful degradation
            return InferenceResult(
                output_text="",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                time_to_first_token_ms=0,
                total_latency_ms=0,
                tokens_per_second=0,
                success=False,
                error_message=str(e),
            )

    def inference_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        max_retries: int | None = None,
        **kwargs,
    ) -> InferenceResult:
        """
        Run inference with configurable retry logic.

        Args:
            image: PIL Image
            prompt: Text prompt
            max_new_tokens: Max tokens
            temperature: Temperature
            max_retries: Override default max retries

        Returns:
            InferenceResult
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_result = None

        for attempt in range(retries + 1):
            try:
                result = self._do_inference(
                    image=image,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                if result.success:
                    return result
                last_result = result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retries + 1} failed: {e}")
                last_result = InferenceResult(
                    output_text="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    time_to_first_token_ms=0,
                    total_latency_ms=0,
                    tokens_per_second=0,
                    success=False,
                    error_message=str(e),
                )

            if attempt < retries:
                delay = self.retry_delay_base ** attempt
                logger.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)

        return last_result or InferenceResult(
            output_text="",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            time_to_first_token_ms=0,
            total_latency_ms=0,
            tokens_per_second=0,
            success=False,
            error_message="All retries exhausted",
        )

    def get_model_info(self) -> ModelInfo:
        """Get model information."""
        try:
            return self._do_get_model_info()
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
            return ModelInfo(
                model_id=self.model_config.get("huggingface_id", "unknown"),
                platform=self.platform_name,
            )

    def disconnect(self):
        """Disconnect from server."""
        self._is_connected = False
        self._client = None
        logger.info(f"Disconnected from {self.platform_name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
