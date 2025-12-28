"""Image loading from folder with validation."""

import logging
from pathlib import Path
from typing import Iterator

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


class ImageLoader:
    """Load images from a folder for benchmarking."""

    def __init__(self, folder_path: str | Path, max_images: int | None = None):
        """
        Initialize image loader.

        Args:
            folder_path: Path to folder containing images
            max_images: Maximum number of images to load (None for all)
        """
        self.folder_path = Path(folder_path)
        self.max_images = max_images

        if not self.folder_path.exists():
            raise ValueError(f"Image folder does not exist: {folder_path}")

        if not self.folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        self._images: list[Path] = self._discover_images()

        if not self._images:
            raise ValueError(f"No supported images found in {folder_path}")

        logger.info(f"Found {len(self._images)} images in {folder_path}")

    def _discover_images(self) -> list[Path]:
        """Find all supported image files in folder."""
        images = []

        for ext in SUPPORTED_EXTENSIONS:
            images.extend(self.folder_path.glob(f"*{ext}"))
            images.extend(self.folder_path.glob(f"*{ext.upper()}"))

        # Sort by name for reproducibility
        images = sorted(set(images), key=lambda p: p.name.lower())

        # Apply max_images limit
        if self.max_images is not None:
            images = images[: self.max_images]

        return images

    def __len__(self) -> int:
        """Return number of images."""
        return len(self._images)

    def __iter__(self) -> Iterator[tuple[str, Image.Image]]:
        """Iterate over (filename, PIL.Image) pairs."""
        for path in self._images:
            try:
                image = Image.open(path).convert("RGB")
                yield path.name, image
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue

    def get_image(self, index: int) -> tuple[str, Image.Image]:
        """
        Get specific image by index.

        Args:
            index: Image index

        Returns:
            Tuple of (filename, PIL.Image)
        """
        if index < 0 or index >= len(self._images):
            raise IndexError(f"Image index {index} out of range [0, {len(self._images)})")

        path = self._images[index]
        image = Image.open(path).convert("RGB")
        return path.name, image

    def get_sample(self, n: int = 5) -> list[tuple[str, Image.Image]]:
        """
        Get sample of n images for quick testing.

        Args:
            n: Number of images to sample

        Returns:
            List of (filename, PIL.Image) tuples
        """
        import random

        indices = random.sample(range(len(self)), min(n, len(self)))
        return [self.get_image(i) for i in indices]

    def get_paths(self) -> list[Path]:
        """Return list of image paths."""
        return self._images.copy()

    def validate_all(self) -> tuple[int, list[str]]:
        """
        Validate all images can be loaded.

        Returns:
            Tuple of (valid_count, list_of_failed_filenames)
        """
        valid_count = 0
        failed = []

        for path in self._images:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_count += 1
            except Exception as e:
                failed.append(f"{path.name}: {e}")

        return valid_count, failed
