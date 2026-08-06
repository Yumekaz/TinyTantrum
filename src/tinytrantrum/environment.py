import platform
import sys


def report() -> dict[str, object]:
    """Return the environment facts recorded with every experiment."""
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device": "cuda" if cuda_available else "cpu",
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
        "default_dtype": str(torch.get_default_dtype()),
        "autocast": "not used",
    }
