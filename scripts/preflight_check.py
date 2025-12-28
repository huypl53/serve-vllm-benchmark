#!/usr/bin/env python3
"""Preflight checks before running benchmarks."""

import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        return False, f"Python 3.10+ required, found {version.major}.{version.minor}"
    return True, f"Python {version.major}.{version.minor}"


def check_docker():
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "Docker not installed"

        # Check if Docker daemon is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "Docker daemon not running"

        return True, "Docker is running"
    except FileNotFoundError:
        return False, "Docker not installed"
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out"


def check_nvidia_driver():
    """Check NVIDIA driver."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "NVIDIA driver not found"

        driver_version = result.stdout.strip()
        return True, f"NVIDIA driver {driver_version}"
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timed out"


def check_gpu_memory():
    """Check available GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "Could not query GPU memory"

        line = result.stdout.strip().split("\n")[0]
        parts = line.split(", ")
        total = parts[0].strip()
        free = parts[1].strip()
        name = parts[2].strip()

        return True, f"{name}: {free} free / {total} total"
    except Exception as e:
        return False, f"GPU memory check failed: {e}"


def check_nvidia_container_toolkit():
    """Check if NVIDIA Container Toolkit is configured."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "nvidia" in result.stdout.lower():
            return True, "NVIDIA Container Toolkit configured"
        return False, "NVIDIA Container Toolkit not found in Docker runtimes"
    except Exception as e:
        return False, f"Could not check: {e}"


def check_huggingface_token():
    """Check for HuggingFace token."""
    # Check environment variables
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True, "Token found in environment"

    # Check cache files
    token_paths = [
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
    ]

    for path in token_paths:
        if path.exists():
            return True, f"Token found in {path}"

    return False, "No HuggingFace token found (some models may require it)"


def check_images_folder(folder_path: str | None = None):
    """Check images folder."""
    folder = Path(folder_path) if folder_path else Path("./images")

    if not folder.exists():
        return False, f"Images folder not found: {folder}"

    # Count images
    extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"]
    count = sum(1 for f in folder.iterdir() if f.suffix.lower() in extensions)

    if count == 0:
        return False, f"No images found in {folder}"

    return True, f"Found {count} images in {folder}"


def check_dependencies():
    """Check Python dependencies."""
    required = ["openai", "huggingface_hub", "PIL", "yaml", "rich", "tenacity"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            # Map package names to import names
            import_map = {"PIL": "pillow", "yaml": "pyyaml"}
            missing.append(import_map.get(pkg, pkg))

    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All dependencies installed"


def run_preflight_checks(images_folder: str | None = None, verbose: bool = True):
    """
    Run all preflight checks.

    Args:
        images_folder: Path to images folder
        verbose: Print results

    Returns:
        Tuple of (all_passed, list_of_results)
    """
    checks = [
        ("Python version", check_python_version),
        ("Docker", check_docker),
        ("NVIDIA driver", check_nvidia_driver),
        ("GPU memory", check_gpu_memory),
        ("NVIDIA Container Toolkit", check_nvidia_container_toolkit),
        ("HuggingFace token", check_huggingface_token),
        ("Python dependencies", check_dependencies),
    ]

    if images_folder:
        checks.append(("Images folder", lambda: check_images_folder(images_folder)))

    results = []
    all_passed = True

    for name, check_fn in checks:
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Check failed: {e}"

        results.append((name, passed, message))

        if not passed and name not in ["HuggingFace token", "NVIDIA Container Toolkit"]:
            # These are warnings, not failures
            all_passed = False

        if verbose:
            status = "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"
            print(f"{status} {name}: {message}")

    return all_passed, results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run preflight checks")
    parser.add_argument("--images", help="Path to images folder")
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()

    print("\nRunning preflight checks...\n")

    all_passed, results = run_preflight_checks(
        images_folder=args.images,
        verbose=not args.quiet,
    )

    print()
    if all_passed:
        print("\033[92mAll checks passed! Ready to run benchmarks.\033[0m")
        return 0
    else:
        print("\033[91mSome checks failed. Please fix the issues above.\033[0m")
        return 1


if __name__ == "__main__":
    sys.exit(main())
