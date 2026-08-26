import pytest

from grmhd.stage_ab import classify_resource_chain


def test_stage_ab_resolved_chain():
    assert classify_resource_chain(
        interop_pass=True,
        dev_dxg_exists=True,
        wsl_nvidia_smi_pass=True,
        torch_cuda_available=True,
        torch_device_count=1,
        host_gpu_visible=True,
    ) == ("R7_RESOLVED", (), True)


def test_stage_ab_current_multiple_failure_contract():
    classification, components, gate = classify_resource_chain(
        interop_pass=False,
        dev_dxg_exists=False,
        wsl_nvidia_smi_pass=False,
        torch_cuda_available=False,
        torch_device_count=0,
        host_gpu_visible=None,
    )
    assert classification == "R6_MULTIPLE_RESOURCE_LAYER_FAILURES"
    assert components == (
        "R2_WSL_GPU_BRIDGE_BROKEN",
        "R3_WSL_INTEROP_BROKEN",
    )
    assert gate is False


def test_stage_ab_does_not_infer_host_failure_from_missing_dxg():
    classification, components, _ = classify_resource_chain(
        interop_pass=True,
        dev_dxg_exists=False,
        wsl_nvidia_smi_pass=False,
        torch_cuda_available=False,
        torch_device_count=0,
        host_gpu_visible=None,
    )
    assert classification == "R2_WSL_GPU_BRIDGE_BROKEN"
    assert "R1_HOST_GPU_NOT_VISIBLE" not in components


def test_stage_ab_pytorch_failure_requires_working_lower_layers():
    classification, components, _ = classify_resource_chain(
        interop_pass=True,
        dev_dxg_exists=True,
        wsl_nvidia_smi_pass=True,
        torch_cuda_available=False,
        torch_device_count=0,
        host_gpu_visible=True,
    )
    assert classification == "R5_PYTORCH_CUDA_ENVIRONMENT_BROKEN"
    assert components == ("R5_PYTORCH_CUDA_ENVIRONMENT_BROKEN",)


def test_stage_ab_rejects_negative_device_count():
    with pytest.raises(ValueError):
        classify_resource_chain(
            interop_pass=False,
            dev_dxg_exists=False,
            wsl_nvidia_smi_pass=False,
            torch_cuda_available=False,
            torch_device_count=-1,
            host_gpu_visible=None,
        )
