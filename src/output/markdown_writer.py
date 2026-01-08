"""Markdown output writing for benchmark results with embedded images."""

import base64
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from ..metrics.collector import AggregatedMetrics

logger = logging.getLogger(__name__)


class MarkdownWriter:
    """Writes benchmark results to rich markdown files with embedded images."""

    def __init__(self, output_dir: str | Path, timestamp_format: str = "%Y%m%d_%H%M%S"):
        """
        Initialize markdown writer.

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
            return self.output_dir / f"{prefix}_{self._timestamp}.md"
        return self.output_dir / f"{prefix}.md"

    def _resize_to_thumbnail(self, image: Image.Image, max_size: int = 300) -> Image.Image:
        """Resize image to thumbnail for embedding in markdown."""
        if max(image.size) <= max_size:
            return image
        thumbnail = image.copy()
        thumbnail.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return thumbnail

    def _image_to_base64(self, image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
        """Convert PIL Image to base64 data URL."""
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format=format, quality=quality, optimize=True)
        b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{b64_data}"

    def write_example_responses(
        self,
        platform: str,
        model_id: str,
        metrics: AggregatedMetrics,
        model_outputs: list[dict[str, str]],
        images: list[tuple[str, Image.Image]],
        detailed_results: list[dict[str, Any]],
    ) -> Path:
        """
        Write rich markdown with example responses.

        Args:
            platform: Platform name
            model_id: Model identifier
            metrics: Aggregated metrics
            model_outputs: List of model output dictionaries
            images: List of (filename, PIL.Image) tuples
            detailed_results: List of detailed result dictionaries

        Returns:
            Path to written markdown file
        """
        filepath = self._get_filepath("example_responses")

        # Build markdown content
        lines = []

        # Header
        lines.append(f"# Benchmark Results: {model_id} on {platform}")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().isoformat()}")
        lines.append("")

        # Aggregated metrics table
        lines.append("## Aggregated Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        metrics_dict = asdict(metrics)
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.2f} |")
            else:
                lines.append(f"| {key} | {value} |")

        lines.append("")

        # Example responses
        lines.append("## Example Responses")
        lines.append("")

        # Create a lookup for detailed results by image_file and iteration
        results_lookup = {}
        for result in detailed_results:
            key = (result["image_file"], result["iteration"])
            results_lookup[key] = result

        # Process each image/output pair
        for i, (output, (filename, image)) in enumerate(zip(model_outputs, images), 1):
            lines.append(f"### Example {i}: {filename}")
            lines.append("")

            # Add thumbnail image
            thumbnail = self._resize_to_thumbnail(image)
            image_data_url = self._image_to_base64(thumbnail)
            lines.append(f"![{filename}]({image_data_url})")
            lines.append("")

            # Add response
            lines.append("**Response**:")
            lines.append("")
            lines.append("```")
            lines.append(output.get("output", "No output"))
            lines.append("```")
            lines.append("")

            # Add performance metrics
            key = (filename, 0)  # First iteration
            if key in results_lookup:
                result = results_lookup[key]
                lines.append("**Performance**:")
                lines.append("")
                lines.append(f"- TTFT: {result.get('ttft_ms', 'N/A')} ms")
                lines.append(f"- Latency: {result.get('total_latency_ms', 'N/A')} ms")
                lines.append(f"- Tokens/sec: {result.get('tokens_per_second', 'N/A')}")
                lines.append(f"- Prompt tokens: {result.get('prompt_tokens', 'N/A')}")
                lines.append(f"- Completion tokens: {result.get('completion_tokens', 'N/A')}")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Write to file
        content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Wrote {len(model_outputs)} example responses to {filepath}")
        return filepath
