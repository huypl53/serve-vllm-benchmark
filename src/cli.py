"""Command-line interface for VLM benchmarking."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .benchmarker import BenchmarkRunner
from .config import load_benchmark_config, load_models_config, get_gpu_profile
from .metrics.gpu_monitor import get_gpu_info, detect_gpu_type
from .output.merge_outputs import merge_model_output_csvs
from .platforms import PLATFORM_REGISTRY

console = Console()


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Setup logging with rich handler."""
    handlers = [RichHandler(console=console, rich_tracebacks=True)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )


def cmd_benchmark(args):
    """Run benchmark command."""
    # Load configs
    config = load_benchmark_config(args.config)
    models_config = load_models_config(args.models_config)

    # Setup logging
    setup_logging(
        level=args.log_level or config.logging.level,
        log_file=args.log_file or config.logging.log_file,
    )

    # Validate model
    if args.model not in models_config:
        console.print(f"[red]Error: Model '{args.model}' not found in config[/red]")
        console.print(f"Available models: {', '.join(models_config.keys())}")
        return 1

    # Validate platform
    if args.platform not in PLATFORM_REGISTRY:
        console.print(f"[red]Error: Platform '{args.platform}' not supported[/red]")
        console.print(f"Available platforms: {', '.join(PLATFORM_REGISTRY.keys())}")
        return 1

    # Check platform support for model
    model_cfg = models_config[args.model]
    if not model_cfg.platforms.get(args.platform, {}).get("supported", False):
        console.print(
            f"[red]Error: Model '{args.model}' is not supported on '{args.platform}'[/red]"
        )
        return 1

    # Create runner
    runner = BenchmarkRunner(
        config=config,
        models_config=models_config,
        image_folder=args.images,
        output_dir=args.output,
    )

    # Run benchmark
    result = runner.run_single_benchmark(
        platform_name=args.platform,
        model_id=args.model,
        server_url=args.server_url,
        prompt=args.prompt,
        max_images=args.max_images,
        concurrency=args.concurrency,
    )

    return 0 if result.get("success") else 1


def cmd_benchmark_all(args):
    """Run all benchmarks command."""
    config = load_benchmark_config(args.config)
    models_config = load_models_config(args.models_config)

    setup_logging(
        level=args.log_level or config.logging.level,
        log_file=args.log_file or config.logging.log_file,
    )

    # Parse server URLs
    server_urls = {}
    if args.vllm_url:
        server_urls["vllm"] = args.vllm_url
    if args.tgi_url:
        server_urls["tgi"] = args.tgi_url
    if args.tensorrt_url:
        server_urls["tensorrt"] = args.tensorrt_url

    # Parse platforms and models
    platforms = args.platforms.split(",") if args.platforms else None
    models = args.models.split(",") if args.models else None

    runner = BenchmarkRunner(
        config=config,
        models_config=models_config,
        image_folder=args.images,
        output_dir=args.output,
    )

    results = runner.run_all_benchmarks(
        platforms=platforms,
        models=models,
        server_urls=server_urls,
        prompt=args.prompt,
        max_images=args.max_images,
        concurrency=args.concurrency,
    )

    # Return success if at least one benchmark succeeded
    return 0 if any(r.get("success") for r in results) else 1


def cmd_list(args):
    """List available models and platforms."""
    models_config = load_models_config(args.models_config)

    console.print("\n[bold]Available Platforms:[/bold]")
    for platform in PLATFORM_REGISTRY:
        console.print(f"  - {platform}")

    console.print("\n[bold]Available Models:[/bold]")

    table = Table()
    table.add_column("Model ID", style="cyan")
    table.add_column("Display Name", style="blue")
    table.add_column("HuggingFace ID", style="dim")
    table.add_column("VRAM (GB)", style="yellow")
    table.add_column("vLLM", style="green")
    table.add_column("TGI", style="green")
    table.add_column("TensorRT", style="green")

    for model_id, cfg in models_config.items():
        table.add_row(
            model_id,
            cfg.display_name,
            cfg.huggingface_id,
            str(cfg.memory_estimate_gb),
            "Yes" if cfg.platforms.get("vllm", {}).get("supported") else "No",
            "Yes" if cfg.platforms.get("tgi", {}).get("supported") else "No",
            "Yes" if cfg.platforms.get("tensorrt", {}).get("supported") else "No",
        )

    console.print(table)
    return 0


def cmd_gpu_info(args):
    """Show GPU information."""
    info = get_gpu_info()

    if not info:
        console.print("[red]No GPU information available[/red]")
        console.print("Make sure NVIDIA drivers are installed and pynvml is available")
        return 1

    console.print(f"\n[bold]GPU Information:[/bold]")
    console.print(f"Device count: {info['device_count']}")

    table = Table()
    table.add_column("Index", style="cyan")
    table.add_column("Name", style="blue")
    table.add_column("Total Memory", style="green")
    table.add_column("Free Memory", style="yellow")

    for gpu in info["gpus"]:
        table.add_row(
            str(gpu["index"]),
            gpu["name"],
            f"{gpu['memory_total_gb']:.1f} GB",
            f"{gpu['memory_free_gb']:.1f} GB",
        )

    console.print(table)

    # Detect GPU type
    gpu_type = detect_gpu_type()
    if gpu_type:
        console.print(f"\nDetected GPU type: [cyan]{gpu_type}[/cyan]")

        # Show recommended models
        if args.models_config:
            try:
                profiles = get_gpu_profile(args.models_config)
                if gpu_type in profiles:
                    profile = profiles[gpu_type]
                    console.print(f"Recommended models: {', '.join(profile.get('supported_models', []))}")
                    if profile.get("notes"):
                        console.print(f"Notes: {profile['notes']}")
            except Exception:
                pass

    return 0


def cmd_validate(args):
    """Validate images folder."""
    from .data.image_loader import ImageLoader

    try:
        loader = ImageLoader(args.images)
        console.print(f"[green]Found {len(loader)} images in {args.images}[/green]")

        valid_count, failed = loader.validate_all()
        console.print(f"Valid images: {valid_count}")

        if failed:
            console.print(f"[yellow]Failed to validate {len(failed)} images:[/yellow]")
            for f in failed[:10]:
                console.print(f"  - {f}")
            if len(failed) > 10:
                console.print(f"  ... and {len(failed) - 10} more")
            return 1

        console.print("[green]All images validated successfully![/green]")
        return 0

    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        return 1


def cmd_merge_outputs(args):
    """Merge model output CSVs into a single wide CSV."""
    setup_logging(level=args.log_level or "INFO", log_file=args.log_file)

    output_path = merge_model_output_csvs(
        input_dir=args.input,
        output_path=args.output,
        pattern=args.pattern,
    )

    if not output_path:
        console.print("[yellow]No model output CSVs found to merge[/yellow]")
        return 1

    console.print(f"[green]Merged model outputs saved to {output_path}[/green]")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="VLM Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--config",
        default="./config/benchmark.yaml",
        help="Path to benchmark config file",
    )
    common_parser.add_argument(
        "--models-config",
        default="./config/models.yaml",
        help="Path to models config file",
    )
    common_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    common_parser.add_argument("--log-file", help="Log file path")

    # Benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run a single benchmark",
        parents=[common_parser],
    )
    bench_parser.add_argument(
        "--platform",
        required=True,
        choices=list(PLATFORM_REGISTRY.keys()),
        help="Inference platform",
    )
    bench_parser.add_argument(
        "--model",
        required=True,
        help="Model identifier from config",
    )
    bench_parser.add_argument(
        "--server-url",
        required=True,
        help="URL of running inference server",
    )
    bench_parser.add_argument(
        "--images",
        required=True,
        help="Path to folder containing images",
    )
    bench_parser.add_argument(
        "--output",
        default="./results",
        help="Output directory for results",
    )
    bench_parser.add_argument(
        "--prompt",
        help="VQA prompt (uses default from config if not specified)",
    )
    bench_parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum number of images to process",
    )
    bench_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent requests (default: 1 = sequential)",
    )

    # Benchmark all command
    all_parser = subparsers.add_parser(
        "benchmark-all",
        help="Run benchmarks for all supported combinations",
        parents=[common_parser],
    )
    all_parser.add_argument(
        "--images",
        required=True,
        help="Path to folder containing images",
    )
    all_parser.add_argument(
        "--output",
        default="./results",
        help="Output directory for results",
    )
    all_parser.add_argument(
        "--platforms",
        help="Comma-separated list of platforms (default: all)",
    )
    all_parser.add_argument(
        "--models",
        help="Comma-separated list of models (default: all)",
    )
    all_parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL",
    )
    all_parser.add_argument(
        "--tgi-url",
        default="http://localhost:8080",
        help="TGI server URL",
    )
    all_parser.add_argument(
        "--tensorrt-url",
        default="http://localhost:8001",
        help="TensorRT server URL",
    )
    all_parser.add_argument("--prompt", help="VQA prompt")
    all_parser.add_argument(
        "--max-images",
        type=int,
        help="Maximum images per benchmark",
    )
    all_parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent requests (default: 1 = sequential)",
    )

    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available models and platforms",
        parents=[common_parser],
    )

    # GPU info command
    gpu_parser = subparsers.add_parser(
        "gpu-info",
        help="Show GPU information",
        parents=[common_parser],
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate images folder",
    )
    validate_parser.add_argument(
        "--images",
        required=True,
        help="Path to folder containing images",
    )

    # Merge model outputs command
    merge_parser = subparsers.add_parser(
        "merge-outputs",
        help="Merge model output CSVs into a single wide CSV",
        parents=[common_parser],
    )
    merge_parser.add_argument(
        "--input",
        default="./results",
        help="Directory containing model_outputs_*.csv files",
    )
    merge_parser.add_argument(
        "--output",
        help="Output CSV path (default: <input>/model_outputs_merged.csv)",
    )
    merge_parser.add_argument(
        "--pattern",
        default="model_outputs_*.csv",
        help="Glob pattern for model output CSVs",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Dispatch to command handler
    commands = {
        "benchmark": cmd_benchmark,
        "benchmark-all": cmd_benchmark_all,
        "list": cmd_list,
        "gpu-info": cmd_gpu_info,
        "validate": cmd_validate,
        "merge-outputs": cmd_merge_outputs,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
