"""Pure Stage AB resource-chain classification contracts."""

from __future__ import annotations


RECOVERY_CLASSES = (
    "R1_HOST_GPU_NOT_VISIBLE",
    "R2_WSL_GPU_BRIDGE_BROKEN",
    "R3_WSL_INTEROP_BROKEN",
    "R4_WSL_INSTANCE_STALE_OR_CORRUPTED",
    "R5_PYTORCH_CUDA_ENVIRONMENT_BROKEN",
    "R6_MULTIPLE_RESOURCE_LAYER_FAILURES",
    "R7_RESOLVED",
    "R8_UNKNOWN",
)


def classify_resource_chain(
    *,
    interop_pass: bool,
    dev_dxg_exists: bool,
    wsl_nvidia_smi_pass: bool,
    torch_cuda_available: bool,
    torch_device_count: int,
    host_gpu_visible: bool | None,
    instance_stale_or_corrupted: bool = False,
) -> tuple[str, tuple[str, ...], bool]:
    """Classify the host-to-PyTorch chain without inventing host evidence."""

    if int(torch_device_count) < 0:
        raise ValueError("torch device count cannot be negative")

    gate = bool(
        dev_dxg_exists
        and wsl_nvidia_smi_pass
        and torch_cuda_available
        and int(torch_device_count) >= 1
    )
    if gate:
        return "R7_RESOLVED", (), True

    failures: list[str] = []
    if host_gpu_visible is False:
        failures.append("R1_HOST_GPU_NOT_VISIBLE")
    if not dev_dxg_exists:
        failures.append("R2_WSL_GPU_BRIDGE_BROKEN")
    if not interop_pass:
        failures.append("R3_WSL_INTEROP_BROKEN")
    if instance_stale_or_corrupted:
        failures.append("R4_WSL_INSTANCE_STALE_OR_CORRUPTED")
    if dev_dxg_exists and wsl_nvidia_smi_pass and not torch_cuda_available:
        failures.append("R5_PYTORCH_CUDA_ENVIRONMENT_BROKEN")

    ordered = tuple(dict.fromkeys(failures))
    if len(ordered) > 1:
        return "R6_MULTIPLE_RESOURCE_LAYER_FAILURES", ordered, False
    if len(ordered) == 1:
        return ordered[0], ordered, False
    return "R8_UNKNOWN", (), False
