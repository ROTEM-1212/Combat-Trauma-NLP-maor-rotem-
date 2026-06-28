from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Returns the best available torch.device for use in model training/inference.
    Prints a one-line status message so the active device is always visible.

    Usage in any script::

        from gpu_check import get_device
        device = get_device()
        model = model.to(device)

    :return: torch.device ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[device] GPU detected — {gpu_name} (CUDA {torch.version.cuda})")
        return torch.device("cuda")
    print("[device] No GPU found — running on CPU.")
    return torch.device("cpu")


if __name__ == "__main__":
    torch_device = get_device()
    print(f"torch.device : {torch_device}")
