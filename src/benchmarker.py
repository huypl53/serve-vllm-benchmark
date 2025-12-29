"""Main benchmarking orchestrator."""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import BenchmarkConfig, ModelConfig, load_benchmark_config, load_models_config
from .data.image_loader import ImageLoader
from .metrics.collector import MetricsCollector, AggregatedMetrics
from .metrics.gpu_monitor import GPUMonitor, detect_gpu_type, get_gpu_info
from .output.csv_writer import CSVWriter, create_detailed_result_row, create_model_output_row
from .platforms import PLATFORM_REGISTRY, get_platform_class
from .platforms.base import BasePlatform, InferenceResult

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(
        self,
        config: BenchmarkConfig,
        models_config: dict[str, ModelConfig],
        image_folder: str,
        output_dir: str | None = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Benchmark configuration
            models_config: Model configurations
            image_folder: Path to image folder
            output_dir: Output directory (overrides config)
        """
        self.config = config
        self.models_config = models_config
        self.image_loader = ImageLoader(image_folder)
        self.output_dir = output_dir or config.output.results_dir
        self.csv_writer = CSVWriter(self.output_dir, config.output.timestamp_format)

        # GPU monitoring
        gpu_index = config.gpu.target_gpus[0] if config.gpu.target_gpus else 0
        self.gpu_monitor = GPUMonitor(
            gpu_index=gpu_index,
            interval_ms=config.gpu.monitor_interval_ms,
        )

        # Signal handling for cleanup
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._platform: BasePlatform | None = None

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def handler(signum, frame):
            console.print("\n[yellow]Interrupted! Cleaning up...[/yellow]")
            if self._platform:
                self._platform.disconnect()
            self.gpu_monitor.shutdown()
            sys.exit(1)

        signal.signal(signal.SIGINT, handler)

    def _restore_signal_handlers(self):
        """Restore original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint)

    def run_single_benchmark(
        self,
        platform_name: str,
        model_id: str,
        server_url: str,
        prompt: str | None = None,
        max_images: int | None = None,
        concurrency: int | None = None,
    ) -> dict[str, Any]:
        """
        Run benchmark for a single platform/model combination.

        Args:
            platform_name: Platform name (vllm, tgi, tensorrt)
            model_id: Model identifier from config
            server_url: Server URL
            prompt: VQA prompt (uses default if None)
            max_images: Max images to process
            concurrency: Number of concurrent requests (None uses config default)

        Returns:
            Dict with results and file paths
        """
        # Use config default if not specified
        concurrency = concurrency or self.config.benchmark.concurrency_level
        self._setup_signal_handlers()

        try:
            # Get configurations
            model_config = self.models_config.get(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in config")

            platform_cfg = model_config.platforms.get(platform_name, {})
            if not platform_cfg.get("supported", False):
                raise ValueError(f"Model {model_id} not supported on {platform_name}")

            # Create platform
            platform_class = get_platform_class(platform_name)
            platform = platform_class(
                model_config={
                    "huggingface_id": model_config.huggingface_id,
                    "display_name": model_config.display_name,
                },
                platform_config=platform_cfg,
                max_retries=self.config.benchmark.max_retries,
                retry_delay_base=self.config.benchmark.retry_delay_base,
            )
            self._platform = platform

            # Connect to server
            console.print(f"[blue]Connecting to {platform_name} at {server_url}...[/blue]")
            platform.connect(server_url, timeout=self.config.server.startup_timeout)

            model_info = platform.get_model_info()
            console.print(f"[green]Connected! Model: {model_info.model_id}[/green]")

            # Use default prompt if not specified
            prompt = prompt or self.config.prompts.default

            # Get images
            images = list(self.image_loader)
            if max_images:
                images = images[:max_images]

            console.print(f"[blue]Benchmarking {len(images)} images...[/blue]")

            # Warmup
            if self.config.benchmark.warmup_iterations > 0:
                console.print(
                    f"[dim]Running {self.config.benchmark.warmup_iterations} warmup iterations...[/dim]"
                )
                warmup_img = images[0][1]
                for _ in range(self.config.benchmark.warmup_iterations):
                    platform.inference(
                        image=warmup_img,
                        prompt=prompt,
                        max_new_tokens=self.config.benchmark.max_new_tokens,
                        temperature=self.config.benchmark.temperature,
                    )

            # Main benchmark
            metrics_collector = MetricsCollector()
            detailed_results: list[dict] = []
            model_outputs: list[dict] = []

            # Start GPU monitoring
            self.gpu_monitor.start()

            # Show concurrency info
            if concurrency > 1:
                console.print(f"[blue]Using concurrent processing with {concurrency} parallel requests[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Benchmarking {model_id} on {platform_name}",
                    total=len(images) * self.config.benchmark.num_iterations,
                )

                if concurrency > 1:
                    # Concurrent batch processing
                    for iteration in range(self.config.benchmark.num_iterations):
                        # Run all images concurrently for this iteration
                        batch_results = asyncio.run(
                            platform.inference_batch_concurrent(
                                images=images,
                                prompt=prompt,
                                max_new_tokens=self.config.benchmark.max_new_tokens,
                                temperature=self.config.benchmark.temperature,
                                concurrency=concurrency,
                            )
                        )

                        for filename, result in batch_results:
                            metrics_collector.add_result(result)

                            # Record detailed result
                            detailed_results.append(
                                create_detailed_result_row(
                                    platform=platform_name,
                                    model_id=model_info.model_id,
                                    image_file=filename,
                                    iteration=iteration,
                                    result=result,
                                )
                            )

                            # Save first iteration output for quality review
                            if iteration == 0 and self.config.output.save_model_outputs:
                                model_outputs.append(
                                    create_model_output_row(
                                        platform=platform_name,
                                        model_id=model_info.model_id,
                                        image_file=filename,
                                        prompt=prompt,
                                        output=result.output_text,
                                    )
                                )

                            progress.advance(task)
                else:
                    # Sequential processing (original behavior)
                    for filename, image in images:
                        for iteration in range(self.config.benchmark.num_iterations):
                            result = platform.inference_with_retry(
                                image=image,
                                prompt=prompt,
                                max_new_tokens=self.config.benchmark.max_new_tokens,
                                temperature=self.config.benchmark.temperature,
                            )

                            metrics_collector.add_result(result)

                            # Record detailed result
                            detailed_results.append(
                                create_detailed_result_row(
                                    platform=platform_name,
                                    model_id=model_info.model_id,
                                    image_file=filename,
                                    iteration=iteration,
                                    result=result,
                                )
                            )

                            # Save first iteration output for quality review
                            if iteration == 0 and self.config.output.save_model_outputs:
                                model_outputs.append(
                                    create_model_output_row(
                                        platform=platform_name,
                                        model_id=model_info.model_id,
                                        image_file=filename,
                                        prompt=prompt,
                                        output=result.output_text,
                                    )
                                )

                            progress.advance(task)

            # Stop GPU monitoring
            gpu_result = self.gpu_monitor.stop()
            metrics_collector.set_gpu_result(gpu_result)

            # Aggregate metrics
            aggregated = metrics_collector.aggregate()

            # Write results
            paths = self.csv_writer.write_benchmark_run(
                platform=platform_name,
                model_id=model_info.model_id,
                metrics=aggregated,
                detailed_results=detailed_results,
                model_outputs=model_outputs if self.config.output.save_model_outputs else None,
            )

            # Print summary
            self._print_summary(platform_name, model_info.model_id, aggregated)

            return {
                "platform": platform_name,
                "model_id": model_info.model_id,
                "metrics": aggregated,
                "paths": paths,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            console.print(f"[red]Benchmark failed: {e}[/red]")
            return {
                "platform": platform_name,
                "model_id": model_id,
                "error": str(e),
                "success": False,
            }

        finally:
            if self._platform:
                self._platform.disconnect()
                self._platform = None
            self._restore_signal_handlers()

    def _print_summary(
        self,
        platform: str,
        model_id: str,
        metrics: AggregatedMetrics,
    ):
        """Print summary table to console."""
        table = Table(title=f"Benchmark Results: {model_id} on {platform}")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", str(metrics.total_requests))
        table.add_row("Success Rate", f"{metrics.success_rate:.1f}%")
        table.add_row("Avg TTFT", f"{metrics.avg_ttft_ms:.1f} ms")
        table.add_row("P95 TTFT", f"{metrics.p95_ttft_ms:.1f} ms")
        table.add_row("Avg Latency", f"{metrics.avg_total_latency_ms:.1f} ms")
        table.add_row("P95 Latency", f"{metrics.p95_total_latency_ms:.1f} ms")
        table.add_row("Avg Tokens/sec", f"{metrics.avg_tokens_per_second:.1f}")
        table.add_row("Total Tokens", str(metrics.total_tokens_generated))

        if metrics.gpu_monitoring_enabled:
            table.add_row("Peak VRAM", f"{metrics.peak_vram_mb:.0f} MB")
            table.add_row("Avg VRAM", f"{metrics.avg_vram_mb:.0f} MB")

        console.print(table)

    def run_all_benchmarks(
        self,
        platforms: list[str] | None = None,
        models: list[str] | None = None,
        server_urls: dict[str, str] | None = None,
        prompt: str | None = None,
        max_images: int | None = None,
        concurrency: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Run benchmarks for all specified platform/model combinations.

        Args:
            platforms: List of platforms (None = all)
            models: List of models (None = all)
            server_urls: Dict mapping platform to URL
            prompt: VQA prompt
            max_images: Max images per benchmark
            concurrency: Number of concurrent requests

        Returns:
            List of results
        """
        platforms = platforms or list(PLATFORM_REGISTRY.keys())
        models = models or list(self.models_config.keys())
        server_urls = server_urls or {}

        # Default server URLs
        default_urls = {
            "vllm": f"http://localhost:{self.config.server.ports.get('vllm', 8000)}",
            "tgi": f"http://localhost:{self.config.server.ports.get('tgi', 8080)}",
            "tensorrt": f"http://localhost:{self.config.server.ports.get('tensorrt', 8001)}",
        }

        results = []

        for platform in platforms:
            for model_id in models:
                # Check if supported
                model_config = self.models_config.get(model_id)
                if not model_config:
                    console.print(f"[yellow]Skipping unknown model: {model_id}[/yellow]")
                    continue

                platform_cfg = model_config.platforms.get(platform, {})
                if not platform_cfg.get("supported", False):
                    console.print(
                        f"[yellow]Skipping {model_id} on {platform} (not supported)[/yellow]"
                    )
                    continue

                server_url = server_urls.get(platform, default_urls.get(platform))
                if not server_url:
                    console.print(f"[red]No server URL for {platform}[/red]")
                    continue

                console.print(f"\n[bold]Running benchmark: {model_id} on {platform}[/bold]")

                result = self.run_single_benchmark(
                    platform_name=platform,
                    model_id=model_id,
                    server_url=server_url,
                    prompt=prompt,
                    max_images=max_images,
                    concurrency=concurrency,
                )

                results.append(result)

        # Print final summary
        self._print_final_summary(results)

        return results

    def _print_final_summary(self, results: list[dict[str, Any]]):
        """Print final summary of all benchmarks."""
        console.print("\n")

        table = Table(title="All Benchmark Results")
        table.add_column("Platform", style="cyan")
        table.add_column("Model", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Avg TPS", style="yellow")
        table.add_column("P95 Latency", style="magenta")
        table.add_column("Peak VRAM", style="red")

        for result in results:
            if result.get("success"):
                metrics = result["metrics"]
                table.add_row(
                    result["platform"],
                    result["model_id"],
                    "[green]OK[/green]",
                    f"{metrics.avg_tokens_per_second:.1f}",
                    f"{metrics.p95_total_latency_ms:.0f} ms",
                    f"{metrics.peak_vram_mb:.0f} MB" if metrics.gpu_monitoring_enabled else "N/A",
                )
            else:
                table.add_row(
                    result["platform"],
                    result["model_id"],
                    f"[red]FAILED[/red]",
                    "-",
                    "-",
                    "-",
                )

        console.print(table)
        console.print(f"\n[green]Results saved to: {self.output_dir}[/green]")


def run_benchmark(
    platform: str,
    model: str,
    server_url: str,
    image_folder: str,
    output_dir: str = "./results",
    config_path: str = "./config/benchmark.yaml",
    models_config_path: str = "./config/models.yaml",
    prompt: str | None = None,
    max_images: int | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run a single benchmark.

    Args:
        platform: Platform name
        model: Model identifier
        server_url: Server URL
        image_folder: Path to images
        output_dir: Output directory
        config_path: Path to benchmark config
        models_config_path: Path to models config
        prompt: VQA prompt
        max_images: Max images
        concurrency: Number of concurrent requests

    Returns:
        Benchmark result dict
    """
    config = load_benchmark_config(config_path)
    models_config = load_models_config(models_config_path)

    runner = BenchmarkRunner(
        config=config,
        models_config=models_config,
        image_folder=image_folder,
        output_dir=output_dir,
    )

    return runner.run_single_benchmark(
        platform_name=platform,
        model_id=model,
        server_url=server_url,
        prompt=prompt,
        max_images=max_images,
        concurrency=concurrency,
    )
