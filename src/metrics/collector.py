"""Metrics aggregation and statistics calculation."""

import statistics
from dataclasses import dataclass, field

from ..platforms.base import InferenceResult
from .gpu_monitor import GPUMonitorResult


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple iterations."""

    # Latency metrics (ms)
    avg_ttft_ms: float = 0.0
    min_ttft_ms: float = 0.0
    max_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    std_ttft_ms: float = 0.0

    avg_total_latency_ms: float = 0.0
    min_total_latency_ms: float = 0.0
    max_total_latency_ms: float = 0.0
    p50_total_latency_ms: float = 0.0
    p95_total_latency_ms: float = 0.0
    p99_total_latency_ms: float = 0.0
    std_total_latency_ms: float = 0.0

    # Throughput metrics
    avg_tokens_per_second: float = 0.0
    min_tokens_per_second: float = 0.0
    max_tokens_per_second: float = 0.0
    total_tokens_generated: int = 0

    # GPU metrics
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    min_vram_mb: float = 0.0
    gpu_monitoring_enabled: bool = False

    # Counts
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    success_rate: float = 0.0


def percentile(data: list[float], p: float) -> float:
    """
    Calculate percentile of data.

    Args:
        data: List of values
        p: Percentile (0-100)

    Returns:
        Percentile value
    """
    if not data:
        return 0.0

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return sorted_data[0]

    k = (n - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, n - 1)

    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def safe_stdev(data: list[float]) -> float:
    """Calculate standard deviation, returning 0 for insufficient data."""
    if len(data) < 2:
        return 0.0
    return statistics.stdev(data)


class MetricsCollector:
    """Collects and aggregates inference metrics."""

    def __init__(self):
        self.results: list[InferenceResult] = []
        self.gpu_result: GPUMonitorResult | None = None

    def add_result(self, result: InferenceResult):
        """Add an inference result."""
        self.results.append(result)

    def set_gpu_result(self, gpu_result: GPUMonitorResult):
        """Set GPU monitoring result."""
        self.gpu_result = gpu_result

    def clear(self):
        """Clear collected results."""
        self.results = []
        self.gpu_result = None

    def aggregate(self) -> AggregatedMetrics:
        """
        Aggregate all collected metrics.

        Returns:
            AggregatedMetrics with computed statistics
        """
        if not self.results:
            return AggregatedMetrics()

        # Separate successful and failed
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if not successful:
            return AggregatedMetrics(
                total_requests=len(self.results),
                successful_requests=0,
                failed_requests=len(failed),
                success_rate=0.0,
            )

        # Extract metrics from successful results
        ttfts = [r.time_to_first_token_ms for r in successful if r.time_to_first_token_ms > 0]
        latencies = [r.total_latency_ms for r in successful]
        tps_values = [r.tokens_per_second for r in successful if r.tokens_per_second > 0]

        # GPU metrics
        peak_vram = 0.0
        avg_vram = 0.0
        min_vram = 0.0
        gpu_enabled = False

        if self.gpu_result and self.gpu_result.monitoring_enabled:
            peak_vram = self.gpu_result.peak_memory_mb
            avg_vram = self.gpu_result.avg_memory_mb
            min_vram = self.gpu_result.min_memory_mb
            gpu_enabled = True

        return AggregatedMetrics(
            # TTFT metrics
            avg_ttft_ms=statistics.mean(ttfts) if ttfts else 0,
            min_ttft_ms=min(ttfts) if ttfts else 0,
            max_ttft_ms=max(ttfts) if ttfts else 0,
            p50_ttft_ms=percentile(ttfts, 50),
            p95_ttft_ms=percentile(ttfts, 95),
            p99_ttft_ms=percentile(ttfts, 99),
            std_ttft_ms=safe_stdev(ttfts),
            # Latency metrics
            avg_total_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_total_latency_ms=min(latencies) if latencies else 0,
            max_total_latency_ms=max(latencies) if latencies else 0,
            p50_total_latency_ms=percentile(latencies, 50),
            p95_total_latency_ms=percentile(latencies, 95),
            p99_total_latency_ms=percentile(latencies, 99),
            std_total_latency_ms=safe_stdev(latencies),
            # Throughput
            avg_tokens_per_second=statistics.mean(tps_values) if tps_values else 0,
            min_tokens_per_second=min(tps_values) if tps_values else 0,
            max_tokens_per_second=max(tps_values) if tps_values else 0,
            total_tokens_generated=sum(r.completion_tokens for r in successful),
            # GPU
            peak_vram_mb=peak_vram,
            avg_vram_mb=avg_vram,
            min_vram_mb=min_vram,
            gpu_monitoring_enabled=gpu_enabled,
            # Counts
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=len(successful) / len(self.results) * 100,
        )


def aggregate_metrics(
    results: list[InferenceResult],
    gpu_result: GPUMonitorResult | None = None,
) -> AggregatedMetrics:
    """
    Convenience function to aggregate metrics.

    Args:
        results: List of inference results
        gpu_result: Optional GPU monitoring result

    Returns:
        AggregatedMetrics
    """
    collector = MetricsCollector()
    for result in results:
        collector.add_result(result)
    if gpu_result:
        collector.set_gpu_result(gpu_result)
    return collector.aggregate()
