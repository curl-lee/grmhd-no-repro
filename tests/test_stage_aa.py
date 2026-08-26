import pytest

from grmhd.stage_aa import classify_gpu_failure


def test_stage_aa_gate_resolved():
    assert classify_gpu_failure(
        dev_dxg_exists=True, wsl_nvidia_smi_pass=True,
        torch_cuda_available=True, torch_device_count=1,
        host_gpu_visible=True,
    ) == ("RESOLVED", True)


def test_stage_aa_missing_dxg_is_wsl_bridge_when_host_unknown():
    assert classify_gpu_failure(
        dev_dxg_exists=False, wsl_nvidia_smi_pass=False,
        torch_cuda_available=False, torch_device_count=0,
        host_gpu_visible=None,
    ) == ("WSL_GPU_BRIDGE", False)


def test_stage_aa_host_driver_failure_precedes_bridge():
    assert classify_gpu_failure(
        dev_dxg_exists=False, wsl_nvidia_smi_pass=False,
        torch_cuda_available=False, torch_device_count=0,
        host_gpu_visible=False,
    ) == ("HOST_DRIVER", False)


def test_stage_aa_pytorch_layer_requires_wsl_smi_pass():
    assert classify_gpu_failure(
        dev_dxg_exists=True, wsl_nvidia_smi_pass=True,
        torch_cuda_available=False, torch_device_count=0,
        host_gpu_visible=True,
    ) == ("PYTORCH_ENVIRONMENT", False)


def test_stage_aa_rejects_negative_device_count():
    with pytest.raises(ValueError):
        classify_gpu_failure(
            dev_dxg_exists=True, wsl_nvidia_smi_pass=False,
            torch_cuda_available=False, torch_device_count=-1,
            host_gpu_visible=None,
        )
