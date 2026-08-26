from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/compare_paper_smokes.py"


def load_module():
    spec = spec_from_file_location("compare_paper_smokes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(mode: str, state_hash: str = "same"):
    disabled = [] if mode == "full" else ["h1", "roi"]
    return {
        "mode": mode,
        "status": "passed_with_h1_warning" if mode == "full" else "passed",
        "parameter_count": 10,
        "initial_state_hash": state_hash,
        "protocol_checksum": "protocol",
        "epochs": 2,
        "train_batches": 4,
        "runtime": {
            "wall_seconds": 1.0,
            "total_wall_seconds": 2.0,
            "samples_per_second": 4.0,
            "device": "cpu",
            "peak_allocated_mib": 0.0,
            "peak_reserved_mib": 0.0,
            "clipping_fraction": 1.0,
        },
        "one_step": {
            "oracle_aware": {
                "metrics": {
                    name: {"global_relative_l2": 1.0}
                    for name in ("E_norm", "E_model_oracle", "E_model_raw", "E_oracle_raw")
                }
            }
        },
        "rollout3": {
            "finite": True,
            "rho_press_positive": True,
            "no_double_transform": True,
        },
        "checkpoint_reload": {
            name: {
                key: True
                for key in (
                    "strict_model_reload",
                    "optimizer_reload",
                    "scheduler_reload",
                    "prediction_parity",
                    "metadata_validated",
                )
            }
            for name in ("best", "last")
        },
        "h1_monitoring": {
            "value_ratios": [20.0] if mode == "full" else None,
            "gradient_ratios": [8.0] if mode == "full" else None,
        },
        "disabled_components": disabled,
    }


def test_comparison_requires_controlled_initial_identity_and_disclaims_science(tmp_path):
    module = load_module()
    payload = module.build_comparison(metric("full"), metric("plain"))
    assert payload["status"] == "passed"
    assert payload["scientific_comparison"] is False
    assert payload["controlled_identity"]["initial_state_hash_equal"] is True
    assert payload["runs"][1]["enabled_loss_components"] == ["plain_l2"]
    assert payload["runs"][1]["disabled_loss_components"] == ["h1", "roi"]
    module.write_comparison(payload, tmp_path)
    assert (tmp_path / "smoke_comparison.csv").is_file()
    assert "does not establish scientific" in (
        tmp_path / "smoke_comparison.md"
    ).read_text(encoding="utf-8")


def test_comparison_fails_when_initial_state_differs():
    module = load_module()
    payload = module.build_comparison(
        metric("full", state_hash="full"), metric("plain", state_hash="plain")
    )
    assert payload["status"] == "failed"
