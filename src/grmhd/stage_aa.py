"""Pure Stage AA GPU-gate classification contracts."""

from __future__ import annotations


GPU_FAILURE_LAYERS = (
    "HOST_DRIVER",
    "WSL_GPU_BRIDGE",
    "CUDA_RUNTIME",
    "PYTORCH_ENVIRONMENT",
    "RESOLVED",
    "UNKNOWN",
)


def classify_gpu_failure(
    *, dev_dxg_exists: bool, wsl_nvidia_smi_pass: bool,
    torch_cuda_available: bool, torch_device_count: int,
    host_gpu_visible: bool | None,
) -> tuple[str, bool]:
    """Return the predeclared failure layer and scientific-gate result.

    ``host_gpu_visible=None`` means the Windows host could not be queried.  In
    that case a missing ``/dev/dxg`` is still direct evidence of a missing WSL
    GPU bridge, but it is not evidence that the host driver itself is healthy.
    """

    if int(torch_device_count) < 0:
        raise ValueError("torch device count cannot be negative")
    gate = bool(
        wsl_nvidia_smi_pass and torch_cuda_available and int(torch_device_count) >= 1
    )
    if gate:
        return "RESOLVED", True
    if host_gpu_visible is False:
        return "HOST_DRIVER", False
    if not dev_dxg_exists:
        return "WSL_GPU_BRIDGE", False
    if wsl_nvidia_smi_pass and not torch_cuda_available:
        return "PYTORCH_ENVIRONMENT", False
    if not wsl_nvidia_smi_pass:
        return "CUDA_RUNTIME", False
    return "UNKNOWN", False
