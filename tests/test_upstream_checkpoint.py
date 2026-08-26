from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest
import torch

from grmhd.upstream_adapters import (
    PINNED_NEURALOP_COMMIT,
    UpstreamFNOConfig,
    build_upstream_fno,
    build_upstream_optimizer,
    load_upstream_training_bundle,
    save_upstream_training_bundle,
)


def test_upstream_checkpoint_round_trip_and_commit_rejection(tmp_path):
    config = UpstreamFNOConfig(in_channels=8)
    model = build_upstream_fno(config)
    optimizer = build_upstream_optimizer(model)
    metadata = {
        "model_config": config.as_dict(), "prediction_mode": "state",
        "target_mode": "state", "normalizer_checksum": "n",
        "target_stats_checksum": "t", "hdf5_checksum": "h",
        "train_indices": [0, 1], "fold": "synthetic", "shells": None,
        "loss_weights": {"l2": 1.0}, "rollout_k": 1,
        "checkpoint_selection_metric": "target",
    }
    save_dir = tmp_path / "checkpoint"
    save_upstream_training_bundle(
        save_dir, "best", model=model, optimizer=optimizer,
        scheduler=None, epoch=3, metadata=metadata,
    )
    reloaded = build_upstream_fno(config)
    reloaded_optimizer = build_upstream_optimizer(reloaded)
    reloaded, reloaded_optimizer, _, epoch, loaded_metadata = load_upstream_training_bundle(
        save_dir, "best", model=reloaded, optimizer=reloaded_optimizer,
    )
    assert epoch == 3
    assert loaded_metadata["neuraloperator_commit"] == PINNED_NEURALOP_COMMIT
    for expected, actual in zip(model.parameters(), reloaded.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    sidecar = save_dir / "best_grmhd_metadata.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["neuraloperator_commit"] = "wrong"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="commit mismatch"):
        load_upstream_training_bundle(save_dir, "best", model=build_upstream_fno(config))


def test_pinned_upstream_worktree_is_clean():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(root / "external/neuraloperator"), "status", "--short"],
        check=True, capture_output=True, text=True,
    )
    assert result.stdout == ""
