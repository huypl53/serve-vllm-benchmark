"""GPU monitoring using pynvml with graceful fallback."""

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import pynvml, but don't fail if not available
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available. GPU monitoring will be disabled.")


@dataclass
class GPUMetrics:
    """GPU metrics snapshot."""

    timestamp_ms: float
    memory_used_mb: float
    memory_total_mb: float
    memory_utilization_pct: float
    gpu_utilization_pct: float
    temperature_c: int = 0
    power_draw_w: float = 0.0


@dataclass
class GPUMonitorResult:
    """Aggregated GPU monitoring results."""

    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    min_memory_mb: float = 0.0
    peak_utilization_pct: float = 0.0
    avg_utilization_pct: float = 0.0
    avg_temperature_c: float = 0.0
    avg_power_w: float = 0.0
    samples: list[GPUMetrics] = field(default_factory=list)
    monitoring_enabled: bool = True


class GPUMonitor:
    """Monitors GPU metrics during inference with graceful fallback."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        """
        Initialize GPU monitor.

        Args:
            gpu_index: GPU index to monitor
            interval_ms: Sampling interval in milliseconds
        """
        self.gpu_index = gpu_index
        self.interval_s = interval_ms / 1000.0
        self._samples: list[GPUMetrics] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._handle = None
        self._initialized = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                self._initialized = True
                logger.info(f"GPU monitor initialized for GPU {gpu_index}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitor: {e}")
                self._initialized = False
        else:
            self._initialized = False

    def _collect_sample(self) -> GPUMetrics | None:
        """Collect a single GPU metrics sample."""
        if not self._initialized or self._handle is None:
            return None

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)

            # Optional metrics (may not be available on all GPUs)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = 0

            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW to W
            except Exception:
                power = 0.0

            return GPUMetrics(
                timestamp_ms=time.perf_counter() * 1000,
                memory_used_mb=mem_info.used / (1024 * 1024),
                memory_total_mb=mem_info.total / (1024 * 1024),
                memory_utilization_pct=(mem_info.used / mem_info.total) * 100,
                gpu_utilization_pct=util.gpu,
                temperature_c=temp,
                power_draw_w=power,
            )
        except Exception as e:
            logger.debug(f"Failed to collect GPU sample: {e}")
            return None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            sample = self._collect_sample()
            if sample is not None:
                self._samples.append(sample)
            time.sleep(self.interval_s)

    def start(self):
        """Start background monitoring."""
        self._samples = []

        if not self._initialized:
            logger.debug("GPU monitoring disabled (pynvml not available)")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug("GPU monitoring started")

    def stop(self) -> GPUMonitorResult:
        """
        Stop monitoring and return aggregated results.

        Returns:
            GPUMonitorResult with aggregated metrics
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if not self._samples:
            return GPUMonitorResult(monitoring_enabled=self._initialized)

        memories = [s.memory_used_mb for s in self._samples]
        utils = [s.gpu_utilization_pct for s in self._samples]
        temps = [s.temperature_c for s in self._samples if s.temperature_c > 0]
        powers = [s.power_draw_w for s in self._samples if s.power_draw_w > 0]

        return GPUMonitorResult(
            peak_memory_mb=max(memories),
            avg_memory_mb=sum(memories) / len(memories),
            min_memory_mb=min(memories),
            peak_utilization_pct=max(utils),
            avg_utilization_pct=sum(utils) / len(utils),
            avg_temperature_c=sum(temps) / len(temps) if temps else 0,
            avg_power_w=sum(powers) / len(powers) if powers else 0,
            samples=self._samples,
            monitoring_enabled=True,
        )

    def get_current_memory(self) -> tuple[float, float] | None:
        """
        Get current GPU memory usage.

        Returns:
            Tuple of (used_mb, total_mb) or None if unavailable
        """
        if not self._initialized or self._handle is None:
            return None

        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return (
                mem_info.used / (1024 * 1024),
                mem_info.total / (1024 * 1024),
            )
        except Exception:
            return None

    def get_gpu_name(self) -> str | None:
        """Get GPU name."""
        if not self._initialized or self._handle is None:
            return None

        try:
            return pynvml.nvmlDeviceGetName(self._handle)
        except Exception:
            return None

    def shutdown(self):
        """Clean up pynvml resources."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def __del__(self):
        """Destructor to clean up resources."""
        self.shutdown()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def get_gpu_info() -> dict | None:
    """
    Get information about available GPUs.

    Returns:
        Dict with GPU info or None if unavailable
    """
    if not PYNVML_AVAILABLE:
        return None

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpus.append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_gb": mem_info.total / (1024**3),
                    "memory_free_gb": mem_info.free / (1024**3),
                }
            )

        pynvml.nvmlShutdown()

        return {"device_count": device_count, "gpus": gpus}
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
        return None


def detect_gpu_type() -> str | None:
    """
    Detect GPU type for automatic model selection.

    Returns:
        GPU type string (e.g., 'A100', 'A10', 'T4') or None
    """
    info = get_gpu_info()
    if not info or not info.get("gpus"):
        return None

    name = info["gpus"][0]["name"].upper()

    # Common GPU types
    if "A100" in name:
        return "A100"
    elif "H100" in name:
        return "H100"
    elif "A10" in name:
        return "A10"
    elif "L4" in name:
        return "L4"
    elif "T4" in name:
        return "T4"
    elif "4090" in name:
        return "RTX4090"
    elif "3090" in name:
        return "RTX3090"
    else:
        return name
