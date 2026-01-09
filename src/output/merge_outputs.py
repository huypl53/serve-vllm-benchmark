"""Merge model output CSV files into a single wide table."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_column_name(model_id: str, platform: str) -> str:
    if platform:
        return f"{model_id} ({platform})"
    return model_id


def merge_model_output_csvs(
    input_dir: str | Path,
    output_path: str | Path | None = None,
    pattern: str = "model_outputs_*.csv",
) -> Path | None:
    """
    Merge model output CSVs into a single wide CSV.

    Rows are keyed by (image_file, prompt). Each model+platform becomes a column.
    """
    input_dir_path = Path(input_dir)
    files = sorted(input_dir_path.glob(pattern))
    if not files:
        logger.warning("No model output CSVs found in %s", input_dir_path)
        return None

    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    columns: set[str] = set()

    for filepath in files:
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_file = (row.get("image_file") or "").strip()
                prompt = (row.get("prompt") or "").strip()
                key = (image_file, prompt)

                if key not in rows_by_key:
                    rows_by_key[key] = {"image_file": image_file, "prompt": prompt}

                model_id = (row.get("model_id") or "").strip()
                platform = (row.get("platform") or "").strip()
                if not model_id:
                    continue

                column = _build_column_name(model_id, platform)
                columns.add(column)
                rows_by_key[key][column] = row.get("output", "") or ""

    if not rows_by_key:
        logger.warning("No rows found in model output CSVs")
        return None

    ordered_columns = ["image_file", "prompt"] + sorted(columns)
    output_file = Path(output_path) if output_path else input_dir_path / "model_outputs_merged.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows_by_key.values(), key=lambda r: (r["image_file"], r["prompt"])):
            writer.writerow(row)

    logger.info("Merged %d rows into %s", len(rows_by_key), output_file)
    return output_file
