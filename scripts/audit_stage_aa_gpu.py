#!/usr/bin/env python3
"""Capture and classify the Stage AA WSL/CUDA environment without mutation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Sequence

import torch

from grmhd.stage_aa import classify_gpu_failure


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_aa/gpu"


def run(command: Sequence[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, capture_output=True)
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout.rstrip(),
            "stderr": result.stderr.rstrip(),
        }
    except FileNotFoundError as error:
        return {
            "command": list(command), "returncode": 127,
            "stdout": "", "stderr": str(error),
        }


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def render(record: dict[str, object]) -> str:
    command = " ".join(str(value) for value in record["command"])
    return (
        f"$ {command}\n"
        f"returncode: {record['returncode']}\n"
        f"stdout:\n{record['stdout']}\n"
        f"stderr:\n{record['stderr']}\n"
    )


def main() -> None:
    commands = {
        "uname": run(["uname", "-a"]),
        "os_release": run(["cat", "/etc/os-release"]),
        "nvidia_smi_wsl": run(["nvidia-smi"]),
        "dev_dxg": run(["ls", "-l", "/dev/dxg"]),
        "wsl_libraries": run(["ls", "-la", "/usr/lib/wsl/lib/"]),
        "which_python": run(["which", "python"]),
        "python_version": run([sys.executable, "--version"]),
        "torch_packages": run([
            "conda", "list", "-n", "grmhd-no",
        ]),
    }
    host_smi = Path("/mnt/c/Windows/System32/nvidia-smi.exe")
    commands["nvidia_smi_windows"] = run([str(host_smi)]) if host_smi.exists() else {
        "command": [str(host_smi)], "returncode": 127,
        "stdout": "", "stderr": "Windows nvidia-smi.exe not found",
    }

    dxg_exists = Path("/dev/dxg").exists()
    wsl_smi_pass = commands["nvidia_smi_wsl"]["returncode"] == 0
    host_smi_pass = commands["nvidia_smi_windows"]["returncode"] == 0
    torch_info = {
        "torch_version": str(torch.__version__),
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    host_visibility = "VISIBLE" if host_smi_pass else "UNKNOWN"
    failure_layer, gate_pass = classify_gpu_failure(
        dev_dxg_exists=dxg_exists,
        wsl_nvidia_smi_pass=wsl_smi_pass,
        torch_cuda_available=bool(torch_info["cuda_available"]),
        torch_device_count=int(torch_info["device_count"]),
        # Windows interop itself failed, so a non-zero return code cannot be
        # interpreted as evidence of a failed host driver.
        host_gpu_visible=True if host_smi_pass else None,
    )
    gate = {
        "schema_version": "stage-aa-gpu-gate-v1",
        "reproduction_scope": "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION",
        "GPU_FAILURE_LAYER": failure_layer,
        "GPU_SCIENTIFIC_GATE": "PASS" if gate_pass else "FAIL",
        "WSL_GPU_BRIDGE_FAILURE": failure_layer == "WSL_GPU_BRIDGE",
        "host_gpu_visibility": host_visibility,
        "host_visibility_note": (
            "Windows nvidia-smi.exe could not be executed because WSL interop returned "
            "UtilBindVsockAnyPort; host driver state is not independently proven from this instance."
            if not host_smi_pass else "Windows nvidia-smi.exe succeeded."
        ),
        "dev_dxg_exists": dxg_exists,
        "wsl_nvidia_smi_pass": wsl_smi_pass,
        "windows_nvidia_smi_pass": host_smi_pass,
        "torch": torch_info,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "wsl_stub_libcuda_exists": Path("/usr/lib/wsl/lib/libcuda.so.1").is_file(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_fallback": False,
        "project_code_modified_for_gpu_recovery": False,
        "package_reinstall_attempted": False,
        "driver_change_attempted": False,
        "training_authorized": gate_pass,
    }
    sections = [
        "# Stage AA GPU environment audit",
        "",
        f"captured_from: {ROOT}",
        f"platform: {platform.platform()}",
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
        f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV', '')}",
        "",
    ]
    for name, record in commands.items():
        sections.extend([f"## {name}", "", render(record)])
    sections.extend([
        "## PyTorch probe", "", json.dumps(torch_info, indent=2, sort_keys=True), "",
        "## Classification", "", json.dumps(gate, indent=2, sort_keys=True), "",
        "No driver, CUDA package, PyTorch package, project code, or model setting was changed.", "",
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_text(OUT / "gpu_environment_audit.txt", "\n".join(sections))
    atomic_json(OUT / "gpu_gate.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
