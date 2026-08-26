from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/train_paper_reduced.py"


def load_entry_module():
    spec = spec_from_file_location("train_paper_reduced", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_entry_help_exposes_required_cli():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for option in (
        "--config",
        "--experiment-name",
        "--epochs",
        "--max-train-batches",
        "--max-validation-batches",
        "--smoke",
        "--resume",
        "--initial-state",
        "--pair-order",
        "--output-dir",
    ):
        assert option in result.stdout


def test_stage_f_smoke_limit_rejects_more_than_two_epochs():
    module = load_entry_module()
    with pytest.raises(ValueError, match="capped at two epochs"):
        module.validate_run_limits(
            smoke=True,
            epochs=3,
            max_train_batches=2,
            max_validation_batches=1,
        )


def test_resource_scaled_entry_rejects_more_than_thirty_epochs():
    module = load_entry_module()
    with pytest.raises(ValueError, match="30-epoch"):
        module.validate_run_limits(
            smoke=False,
            epochs=31,
            max_train_batches=None,
            max_validation_batches=None,
        )
