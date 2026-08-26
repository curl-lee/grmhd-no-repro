#!/usr/bin/env python
"""Capture the reproducibility-critical software and GPU environment."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_info(name: str) -> dict[str, str | None]:
    try:
        module = __import__(name)
    except Exception as exc:  # environment capture must survive optional deps
        return {"version": None, "file": None, "error": repr(exc)}
    return {
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/environment_wsl.json"))
    parser.add_argument(
        "--neuraloperator-dir", type=Path, default=Path("external/neuraloperator")
    )
    args = parser.parse_args()

    neuraloperator_dir = args.neuraloperator_dir.resolve()
    neuraloperator_commit = command_output(
        [
            "git",
            "-c",
            f"safe.directory={neuraloperator_dir.as_posix()}",
            "-C",
            str(neuraloperator_dir),
            "rev-parse",
            "HEAD",
        ]
    )

    torch_info = package_info("torch")
    try:
        import torch

        torch_info.update(
            {
                "cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_count": torch.cuda.device_count(),
                "gpus": [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                    }
                    for i in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception:
        pass

    project_dir = Path.cwd().resolve()
    os_release: dict[str, str] = {}
    os_release_path = Path("/etc/os-release")
    if os_release_path.exists():
        for line in os_release_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    payload = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "uname_s": command_output(["uname", "-s"]),
            "uname_a": command_output(["uname", "-a"]),
            "hostname": platform.node(),
            "username": getpass.getuser(),
            "project_path": str(project_dir),
            "is_wsl": "microsoft" in platform.release().lower(),
            "wsl_distro_name": os.environ.get("WSL_DISTRO_NAME"),
            "os_release": os_release,
        },
        "git": {
            "root": command_output(["git", "rev-parse", "--show-toplevel"]),
            "branch": command_output(["git", "branch", "--show-current"]),
            "commit": command_output(["git", "rev-parse", "HEAD"]),
        },
        "packages": {
            "torch": torch_info,
            "neuralop": package_info("neuralop"),
            "numpy": package_info("numpy"),
            "h5py": package_info("h5py"),
            "scipy": package_info("scipy"),
            "torch_harmonics": package_info("torch_harmonics"),
            "tensorly": package_info("tensorly"),
            "yaml": package_info("yaml"),
            "matplotlib": package_info("matplotlib"),
        },
        "neuraloperator": {
            "source_dir": str(neuraloperator_dir),
            "commit": neuraloperator_commit,
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "nvidia_driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
