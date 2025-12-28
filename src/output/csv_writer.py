"""CSV output writing for benchmark results."""

import csv
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..metrics.collector import AggregatedMetrics
from ..platforms.base import InferenceResult

logger = logging.getLogger(__name__)


class CSVWriter:
    """Writes benchmark results to CSV files."""

    def __init__(self, output_dir: str | Path, timestamp_format: str = "%Y%m%d_%H%M%S"):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory for output files
            timestamp_format: Format for timestamp in filenames
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp_format = timestamp_format
        self._timestamp = datetime.now().strftime(timestamp_format)

    def _get_filepath(self, prefix: str, use_timestamp: bool = True) -> Path:
        """Generate filepath with optional timestamp."""
        if use_timestamp:
            return self.output_dir / f"{prefix}_{self._timestamp}.csv"
        return self.output_dir / f"{prefix}.csv"

    def write_detailed_results(
        self,
        results: list[dict[str, Any]],
        filename: str | None = None,
    ) -> Path | None:
        """
        Write per-image detailed results.

        Args:
            results: List of result dictionaries
            filename: Optional custom filename

        Returns:
            Path to written file or None if empty
        """
        if not results:
            logger.warning("No results to write")
            return None

        filepath = Path(filename) if filename else self._get_filepath("detailed_results")

        # Ensure all rows have same keys
        fieldnames = list(results[0].keys())

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"Wrote {len(results)} detailed results to {filepath}")
        return filepath

    def write_summary(
        self,
        summary: dict[str, Any] | AggregatedMetrics,
        filename: str | None = None,
    ) -> Path:
        """
        Write aggregated summary metrics.

        Args:
            summary: Summary dict or AggregatedMetrics
            filename: Optional custom filename

        Returns:
            Path to written file
        """
        filepath = Path(filename) if filename else self._get_filepath("summary")

        # Convert dataclass to dict if needed
        if hasattr(summary, "__dataclass_fields__"):
            summary = asdict(summary)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        logger.info(f"Wrote summary to {filepath}")
        return filepath

    def write_model_outputs(
        self,
        outputs: list[dict[str, str]],
        filename: str | None = None,
    ) -> Path | None:
        """
        Write model outputs for quality review.

        Args:
            outputs: List of output dictionaries
            filename: Optional custom filename

        Returns:
            Path to written file or None if empty
        """
        if not outputs:
            logger.warning("No model outputs to write")
            return None

        filepath = Path(filename) if filename else self._get_filepath("model_outputs")

        fieldnames = ["timestamp", "platform", "model_id", "image_file", "prompt", "output"]

        # Ensure all required fields exist
        for output in outputs:
            for field in fieldnames:
                if field not in output:
                    output[field] = ""

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(outputs)

        logger.info(f"Wrote {len(outputs)} model outputs to {filepath}")
        return filepath

    def append_to_combined_summary(
        self,
        summary: dict[str, Any],
        filename: str = "benchmark_summary.csv",
    ) -> Path:
        """
        Append summary to combined results file.

        This creates a single CSV file with results from all benchmark runs.

        Args:
            summary: Summary dict to append
            filename: Filename (without timestamp)

        Returns:
            Path to file
        """
        filepath = self.output_dir / filename
        file_exists = filepath.exists()

        fieldnames = list(summary.keys())

        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow(summary)

        logger.info(f"Appended summary to {filepath}")
        return filepath

    def write_benchmark_run(
        self,
        platform: str,
        model_id: str,
        metrics: AggregatedMetrics,
        detailed_results: list[dict[str, Any]],
        model_outputs: list[dict[str, str]] | None = None,
    ) -> dict[str, Path]:
        """
        Write all results for a benchmark run.

        Args:
            platform: Platform name
            model_id: Model identifier
            metrics: Aggregated metrics
            detailed_results: Per-image results
            model_outputs: Optional model outputs

        Returns:
            Dict of output file paths
        """
        paths = {}

        # Add metadata to summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "model_id": model_id,
            **asdict(metrics),
        }

        # Write summary
        paths["summary"] = self.write_summary(summary)

        # Append to combined summary
        paths["combined"] = self.append_to_combined_summary(summary)

        # Write detailed results
        if detailed_results:
            paths["detailed"] = self.write_detailed_results(detailed_results)

        # Write model outputs
        if model_outputs:
            paths["outputs"] = self.write_model_outputs(model_outputs)

        return paths


def create_detailed_result_row(
    platform: str,
    model_id: str,
    image_file: str,
    iteration: int,
    result: InferenceResult,
) -> dict[str, Any]:
    """
    Create a detailed result row dictionary.

    Args:
        platform: Platform name
        model_id: Model identifier
        image_file: Image filename
        iteration: Iteration number
        result: Inference result

    Returns:
        Dict for CSV row
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform,
        "model_id": model_id,
        "image_file": image_file,
        "iteration": iteration,
        "success": result.success,
        "ttft_ms": round(result.time_to_first_token_ms, 2),
        "total_latency_ms": round(result.total_latency_ms, 2),
        "tokens_per_second": round(result.tokens_per_second, 2),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "error_message": result.error_message or "",
    }


def create_model_output_row(
    platform: str,
    model_id: str,
    image_file: str,
    prompt: str,
    output: str,
) -> dict[str, str]:
    """
    Create a model output row dictionary.

    Args:
        platform: Platform name
        model_id: Model identifier
        image_file: Image filename
        prompt: Input prompt
        output: Model output text

    Returns:
        Dict for CSV row
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform,
        "model_id": model_id,
        "image_file": image_file,
        "prompt": prompt,
        "output": output,
    }
