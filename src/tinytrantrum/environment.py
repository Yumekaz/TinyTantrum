import platform
import sys


def report() -> dict[str, str | bool]:
    """Return the environment facts recorded with every experiment."""
    import torch

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
