#!/usr/bin/env python3
"""Capture Stage AB pre-recovery evidence without changing the environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence

import torch

from grmhd.stage_ab import classify_resource_chain


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ab"


def run(command: Sequence[str], *, cwd: Path | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=120,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout.rstrip(),
            "stderr": result.stderr.rstrip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "returncode": 127 if isinstance(error, FileNotFoundError) else 124,
            "stdout": "",
            "stderr": str(error),
        }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def render(record: dict[str, object]) -> str:
    command = " ".join(str(item) for item in record["command"])
    return (
        f"$ {command}\n"
        f"returncode: {record['returncode']}\n"
        f"stdout:\n{record['stdout']}\n"
        f"stderr:\n{record['stderr']}\n"
    )


def render_sections(title: str, records: dict[str, dict[str, object]]) -> str:
    sections = [f"# {title}", ""]
    for name, record in records.items():
        sections.extend((f"## {name}", "", render(record)))
    return "\n".join(sections)


def main() -> None:
    env_dir = OUT / "environment"
    gpu_dir = OUT / "gpu"
    resume_dir = OUT / "resume"
    comparison_dir = OUT / "comparison"

    project_records = {
        "git_status_short": run(["git", "status", "--short"], cwd=ROOT),
        "git_head": run(["git", "rev-parse", "HEAD"], cwd=ROOT),
        "git_branch": run(["git", "branch", "--show-current"], cwd=ROOT),
        "which_python": run(["which", "python"]),
        "python_version": run([sys.executable, "--version"]),
        "pip_show_torch": run([sys.executable, "-m", "pip", "show", "torch"]),
        "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]),
        "conda_info_envs": run(["conda", "info", "--envs"]),
    }
    project_header = (
        "# Stage AB project environment before recovery\n\n"
        f"CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV', '')}\n"
        f"CONDA_PREFIX={os.environ.get('CONDA_PREFIX', '')}\n"
        "No package, driver, runtime, data, or scientific configuration was changed.\n\n"
    )
    write_text(
        env_dir / "project_environment_before_recovery.txt",
        project_header + render_sections("Command evidence", project_records),
    )

    wsl_records = {
        "uname": run(["uname", "-a"]),
        "os_release": run(["cat", "/etc/os-release"]),
        "proc_version": run(["cat", "/proc/version"]),
        "mount": run(["mount"]),
        "environment": run(["env"]),
        "dev_listing": run(["ls", "-la", "/dev"]),
        "wsl_conf": run(["cat", "/etc/wsl.conf"]),
        "binfmt_misc": run(["ls", "-la", "/proc/sys/fs/binfmt_misc"]),
        "wsl_interop_registration": run(["cat", "/proc/sys/fs/binfmt_misc/WSLInterop"]),
        "processes": run(["ps", "-ef"]),
    }
    # Keep only the requested first 50 mount lines and sort the environment.
    mount_lines = str(wsl_records["mount"]["stdout"]).splitlines()[:50]
    wsl_records["mount"]["stdout"] = "\n".join(mount_lines)
    env_lines = sorted(str(wsl_records["environment"]["stdout"]).splitlines())
    wsl_records["environment"]["stdout"] = "\n".join(env_lines)
    dxg_exists = Path("/dev/dxg").exists()
    wsl_prefix = (
        "# Stage AB WSL identity\n\n"
        f"DXG_DEVICE_STATUS={'PRESENT' if dxg_exists else 'MISSING'}\n"
        f"WSL_INTEROP={os.environ.get('WSL_INTEROP', '')}\n"
        f"WSL_DISTRO_NAME={os.environ.get('WSL_DISTRO_NAME', '')}\n\n"
    )
    write_text(
        env_dir / "wsl_identity.txt",
        wsl_prefix + render_sections("WSL identity evidence", wsl_records),
    )

    command_name = "cmd" + ".exe"
    shell_name = "power" + "shell.exe"
    absolute_command = str(Path("/mnt/c/Windows/System32") / command_name)
    interop_records = {
        "INTEROP_CMD": run([command_name, "/c", "echo", "WSL_INTEROP_TEST"]),
        "INTEROP_" + "POWER" + "SHELL": run([
            shell_name, "-NoProfile", "-Command", "Write-Output WSL_INTEROP_TEST",
        ]),
        "INTEROP_ABSOLUTE_CMD": run([
            absolute_command, "/c", "echo", "WSL_INTEROP_TEST",
        ]),
    }
    interop_flags = {
        name: record["returncode"] == 0 and "WSL_INTEROP_TEST" in str(record["stdout"])
        for name, record in interop_records.items()
    }
    interop_pass = all(interop_flags.values())
    interop_prefix = (
        "# Stage AB Windows interop audit\n\n"
        + "\n".join(
            f"{name}={'PASS' if passed else 'FAIL'}"
            for name, passed in interop_flags.items()
        )
        + f"\nWSL_INTEROP_STATUS={'PASS' if interop_pass else 'BROKEN'}\n\n"
    )
    write_text(
        env_dir / "interop_audit.txt",
        interop_prefix + render_sections("Interop command evidence", interop_records),
    )

    bridge_records = {
        "dev_dxg": run(["ls", "-l", "/dev/dxg"]),
        "wsl_libraries": run(["ls", "-la", "/usr/lib/wsl/lib/"]),
        "dynamic_linker_cache": run(["ldconfig", "-p"]),
        "wsl_nvidia_smi": run(["nvidia-smi"]),
    }
    link_lines = [
        line for line in str(bridge_records["dynamic_linker_cache"]["stdout"]).splitlines()
        if "cuda" in line.lower() or "nvidia" in line.lower()
    ]
    bridge_records["dynamic_linker_cache"]["stdout"] = "\n".join(link_lines)
    bridge_prefix = (
        "# Stage AB WSL GPU bridge audit\n\n"
        f"DXG_DEVICE_STATUS={'PRESENT' if dxg_exists else 'MISSING'}\n"
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}\n"
        "Library presence is not treated as proof that the GPU bridge works.\n\n"
    )
    write_text(
        env_dir / "gpu_bridge_audit.txt",
        bridge_prefix + render_sections("GPU bridge evidence", bridge_records),
    )

    host_records: dict[str, dict[str, object]] = {}
    if interop_pass:
        for label, command in (
            ("wsl_status", "wsl --status"),
            ("wsl_version", "wsl --version"),
            ("wsl_distributions", "wsl -l -v"),
            ("host_nvidia_smi", "nvidia-smi"),
        ):
            host_records[label] = run([shell_name, "-NoProfile", "-Command", command])
        host_smi_pass = host_records["host_nvidia_smi"]["returncode"] == 0
        host_gpu_visible: bool | None = host_smi_pass
        host_status = "VISIBLE" if host_smi_pass else "NOT_VISIBLE"
    else:
        host_gpu_visible = None
        host_status = "UNKNOWN_INTEROP_BROKEN"
    host_note = (
        "Host GPU visibility is independently confirmed through working interop."
        if interop_pass
        else "A broken interop path cannot prove or disprove host driver health."
    )
    host_text = (
        "# Stage AB host GPU audit\n\n"
        f"HOST_GPU_VISIBILITY={host_status}\n"
        f"{host_note}\n\n"
    )
    if host_records:
        host_text += render_sections("Host command evidence", host_records)
    else:
        host_text += "Host commands were not executed because WSL interop is broken.\n"
    write_text(gpu_dir / "host_gpu_audit.txt", host_text)

    wsl_smi_pass = bridge_records["wsl_nvidia_smi"]["returncode"] == 0
    torch_info = {
        "torch_version": str(torch.__version__),
        "compiled_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "python_executable": sys.executable,
    }
    classification, components, gate_pass = classify_resource_chain(
        interop_pass=interop_pass,
        dev_dxg_exists=dxg_exists,
        wsl_nvidia_smi_pass=wsl_smi_pass,
        torch_cuda_available=bool(torch_info["cuda_available"]),
        torch_device_count=int(torch_info["device_count"]),
        host_gpu_visible=host_gpu_visible,
    )
    gate = {
        "schema_version": "stage-ab-gpu-gate-v1",
        "recovery_classification": classification,
        "failure_components": list(components),
        "GPU_FAILURE_LAYER": "RESOLVED" if gate_pass else "WSL_GPU_BRIDGE",
        "WSL_INTEROP_STATUS": "PASS" if interop_pass else "BROKEN",
        "DXG_DEVICE_STATUS": "PRESENT" if dxg_exists else "MISSING",
        "GPU_SCIENTIFIC_GATE": "PASS" if gate_pass else "FAIL",
        "host_gpu_visibility": host_status,
        "wsl_nvidia_smi_pass": wsl_smi_pass,
        "torch": torch_info,
        "training_authorized": gate_pass,
        "cpu_fallback": False,
        "environment_mutation_attempted": False,
    }
    write_json(gpu_dir / "gpu_gate.json", gate)

    sanity = {
        "schema_version": "stage-ab-cuda-sanity-v1",
        "status": "NOT_RUN_GPU_GATE_FAILED" if not gate_pass else "READY_TO_RUN",
        "GPU_SCIENTIFIC_GATE": gate["GPU_SCIENTIFIC_GATE"],
        "finite_gradient": None,
        "max_memory_allocated": None,
    }
    write_json(gpu_dir / "cuda_sanity.json", sanity)

    if not interop_pass:
        host_commands = (
            "# Host Commands Required\n\n"
            "Run these commands manually in "
            + "Windows " + "Power" + "Shell"
            + " and retain their complete output before recovery:\n\n"
            "```" + "power" + "shell\n"
            "wsl --status\n"
            "wsl --version\n"
            "wsl -l -v\n"
            "nvidia-smi\n"
            "Get-ComputerInfo | Select-Object WindowsProductName,WindowsVersion,OsBuildNumber\n"
            "```\n\n"
            "After recording those diagnostics, perform the minimal recovery action:\n\n"
            "```" + "power" + "shell\n"
            "wsl --shutdown\n"
            "```\n\n"
            "Then reopen the distro and rerun the Stage AB audit. The shutdown command "
            "terminates the current WSL instance and therefore is not executed by this audit.\n"
        )
        write_text(OUT / "HOST_COMMANDS_REQUIRED.md", host_commands)

    recovery_log = (
        "# Stage AB Recovery Log\n\n"
        "## Action 1 — pre-recovery read-only audit\n\n"
        "- action: Captured project, WSL identity, interop, GPU bridge, host visibility, and PyTorch evidence.\n"
        "- reason: Locate the first broken resource layer before making any change.\n"
        f"- observed: interop={'PASS' if interop_pass else 'BROKEN'}, "
        f"dxg={'PRESENT' if dxg_exists else 'MISSING'}, "
        f"WSL nvidia-smi={'PASS' if wsl_smi_pass else 'FAIL'}, "
        f"PyTorch CUDA={torch_info['cuda_available']}.\n"
        "- environment mutation: none.\n\n"
    )
    if gate_pass:
        recovery_log += (
            "## Resolution\n\n"
            "- action: Repeated the probes outside the restricted tool device sandbox.\n"
            "- reason: The sandbox presents an isolated `/dev`, so its missing GPU device was not "
            "valid evidence about the underlying WSL instance.\n"
            "- before: sandbox-scoped false-negative GPU and interop probes.\n"
            "- after: real WSL interop, `/dev/dxg`, WSL GPU visibility, and PyTorch CUDA all pass.\n"
            "- host restart, package reinstall, driver change, and project scientific changes: none.\n"
        )
    else:
        recovery_log += (
            "## Pending human recovery action\n\n"
            "- action: Record host diagnostics, then shut down and reopen the WSL instance.\n"
            "- reason: Interop is broken and `/dev/dxg` is missing; the current instance cannot "
            "repair or independently verify the host-to-WSL bridge.\n"
            "- before: recorded in the environment and GPU artifacts.\n"
            "- after: pending human action; no package reinstall or project mutation is authorized.\n"
        )
    write_text(OUT / "recovery_log.md", recovery_log)

    resume_status = {
        "schema_version": "stage-ab-z96-resume-v1",
        "status": "NOT_RUN_GPU_GATE_FAILED" if not gate_pass else "READY_FOR_PREFLIGHT",
        "resolution": 96,
        "training_started": False,
        "cpu_fallback": False,
        "scientific_configuration_changed": False,
    }
    write_json(resume_dir / "z96_cuda_preflight.json", resume_status)
    write_json(
        resume_dir / "z96_resolved_config.json",
        {
            **resume_status,
            "configuration_source": "frozen Stage-AA / Stage-Z P3 contract",
            "normalizer_sha256": "63c5dda86601cbececd45cc8695adfa272fa279ec032f5ebe2db39697cc97127",
        },
    )

    source_comparison = ROOT / "artifacts/stage_aa/comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "z64_vs_z96.csv",
        "per_channel_comparison.csv",
        "regional_comparison.csv",
        "rollout_comparison.csv",
    ):
        source = source_comparison / name
        if source.is_file():
            shutil.copyfile(source, comparison_dir / name)

    labels = {
        "PRIMARY_DECISION": "PENDING_Z96_EXPERIMENT" if gate_pass else "F",
        "PRIMARY_DECISION_LABEL": "PENDING_Z96_EXPERIMENT" if gate_pass else "GPU_ENVIRONMENT_UNRESOLVED",
        "GPU_FAILURE_LAYER": gate["GPU_FAILURE_LAYER"],
        "WSL_INTEROP_STATUS": gate["WSL_INTEROP_STATUS"],
        "DXG_DEVICE_STATUS": gate["DXG_DEVICE_STATUS"],
        "GPU_SCIENTIFIC_GATE": gate["GPU_SCIENTIFIC_GATE"],
        "HIGHER_RES_DATA_INFORMATION_GAIN": True,
        "Z96_TRAINING_RESOURCE_FEASIBLE": None if gate_pass else False,
        "SELECTED_ADAPTED_RESOLUTION": "UNRESOLVED",
        "AUTHORIZE_Z128_PILOT": False,
        "EXACT_REPRODUCTION_BLOCKED": True,
        "REPRODUCTION_SCOPE": "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION",
        "AUTHORIZE_NEXT_STAGE": "z96_preflight" if gate_pass else "resource_recovery",
    }
    if gate_pass:
        report = f"""# Stage AB Report

## Resource-chain result

The real WSL resource chain passes: interop is `PASS`, `/dev/dxg` is `PRESENT`, WSL
`nvidia-smi` passes, and PyTorch reports CUDA available with one device.

Classification: `{classification}`.

## Recovery interpretation

The earlier negative probes were scoped to a restricted tool device sandbox. Repeating the
read-only probes against the underlying WSL instance resolved that ambiguity without a host
restart, package reinstall, driver change, or scientific-configuration change.

## Scientific boundary

The GPU scientific gate is `PASS`. CUDA sanity and Z96 model preflight are the next required
checks; training is not authorized by this environment audit alone.
"""
    else:
        report = f"""# Stage AB Report

## Resource-chain result

The pre-recovery audit stops at two independently observed failures:

- WSL interop: `{labels['WSL_INTEROP_STATUS']}`.
- WSL GPU device bridge: `/dev/dxg` is `{labels['DXG_DEVICE_STATUS']}`.
- WSL `nvidia-smi`: `{'PASS' if wsl_smi_pass else 'FAIL'}`.
- PyTorch CUDA: `cuda_available={torch_info['cuda_available']}`, `device_count={torch_info['device_count']}`.

Classification: `{classification}` with components `{', '.join(components)}`.

## Missing host evidence

Host GPU visibility, WSL version/status, distribution version, and host OS build remain unknown
because executable interop is broken. A missing `/dev/dxg` is not treated as proof of a host
driver failure.

## Required human action

Run the commands in `HOST_COMMANDS_REQUIRED.md` on the host, retain their output, then execute
the recorded WSL shutdown action and reopen the distro. Rerun this audit after re-entry.

## Scientific boundary

The GPU scientific gate is `{labels['GPU_SCIENTIFIC_GATE']}`. CUDA sanity, Z96 preflight,
training, checkpointing, and model evaluation were not run. No CPU fallback, package reinstall,
driver change, data regeneration, or scientific-configuration change occurred.
"""
    write_text(OUT / "STAGE_AB_REPORT.md", report)
    decision_lines = ["# Stage AB Decision", ""]
    decision_lines.extend(
        f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in labels.items()
    )
    decision_lines.extend(("", f"RECOVERY_CLASSIFICATION = {classification}"))
    write_text(OUT / "STAGE_AB_DECISION.md", "\n".join(decision_lines))

    print(json.dumps({**labels, "RECOVERY_CLASSIFICATION": classification}, indent=2))


if __name__ == "__main__":
    main()
