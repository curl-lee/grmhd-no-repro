"""Stage R normalized-residual target and identity-reconstruction contract."""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any, Mapping

import torch

from . import CHANNELS
from .paper_losses import PlainL2Loss


PREDICTION_MODE = "normalized_residual"
RESIDUAL_SCALE = 1.0
FROZEN_HISTORY = {
    "Stage K": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    "Stage L": "3. MIXED_OVERALL",
    "Stage M": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
    "Stage N": "P3 ready; mixed operator-response failure",
    "Stage O": "C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    "Stage P": "6. MIXED_CLOSED_LOOP_FAILURE",
    "Stage Q": "D. NO_VALID_ANCHOR_CANDIDATE",
}


def _validate_state(name: str, value: torch.Tensor) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 5 or value.shape[1] != len(CHANNELS):
        raise ValueError(f"{name} must have shape (B,8,Nphi,Ntheta,Nr)")


def normalized_residual_target(
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
) -> torch.Tensor:
    """Return the unscaled P3 normalized temporal residual target."""

    _validate_state("normalized_input", normalized_input)
    _validate_state("normalized_target", normalized_target)
    if normalized_input.shape != normalized_target.shape:
        raise ValueError("Stage R input and target shapes differ")
    return normalized_target - normalized_input


def reconstruct_normalized_state(
    normalized_input: torch.Tensor,
    predicted_residual: torch.Tensor,
) -> torch.Tensor:
    """Apply the sole Stage R output contract: an exact identity state skip."""

    _validate_state("normalized_input", normalized_input)
    _validate_state("predicted_residual", predicted_residual)
    if normalized_input.shape != predicted_residual.shape:
        raise ValueError("Stage R input and predicted residual shapes differ")
    return normalized_input + predicted_residual


def residual_plain_l2(
    predicted_residual: torch.Tensor,
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    loss: PlainL2Loss | None = None,
) -> torch.Tensor:
    """Evaluate strict unit-channel Plain L2 in normalized residual space."""

    target_residual = normalized_residual_target(normalized_input, normalized_target)
    return (loss or PlainL2Loss())(predicted_residual, target_residual)


def loss_equivalence(
    predicted_residual: torch.Tensor,
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    loss: PlainL2Loss | None = None,
) -> dict[str, Any]:
    """Numerically compare residual-space and reconstructed-state Plain L2."""

    plain = loss or PlainL2Loss()
    residual_value = residual_plain_l2(
        predicted_residual,
        normalized_input,
        normalized_target,
        loss=plain,
    )
    reconstructed = reconstruct_normalized_state(normalized_input, predicted_residual)
    state_value = plain(reconstructed, normalized_target)
    absolute_difference = torch.abs(residual_value - state_value)
    scale = torch.maximum(
        torch.maximum(torch.abs(residual_value), torch.abs(state_value)),
        residual_value.new_tensor(1.0),
    )
    tolerance = 8.0 * torch.finfo(residual_value.dtype).eps * scale
    return {
        "residual_loss": residual_value,
        "reconstructed_state_loss": state_value,
        "absolute_difference": absolute_difference,
        "tolerance": tolerance,
        "passed": bool((absolute_difference <= tolerance).detach().cpu()),
    }


def validate_contract_config(values: Mapping[str, Any]) -> None:
    prediction = values.get("prediction")
    required = {
        "mode": PREDICTION_MODE,
        "state_skip": "identity",
        "residual_scale": RESIDUAL_SCALE,
        "learnable_scale": False,
        "clipping": False,
    }
    if prediction != required:
        raise ValueError("Stage R prediction contract changed")
    if values.get("model", {}).get("prediction_mode") != PREDICTION_MODE:
        raise ValueError("Stage R model output semantics changed")
    loss = values.get("loss", {})
    if loss.get("mode") != "strict_plain_l2" or loss.get("target") != PREDICTION_MODE:
        raise ValueError("Stage R loss target contract changed")


def contract_source_has_forbidden_operation() -> bool:
    source = "\n".join(
        (
            inspect.getsource(normalized_residual_target),
            inspect.getsource(reconstruct_normalized_state),
        )
    )
    return any(
        token in source
        for token in (
            "clip(",
            "clamp(",
            "nan_to_num",
            "sigmoid(",
            "tanh(",
            "detach(",
        )
    )


def tensor_state_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def expected_rollout_transform_counts(steps: int) -> dict[str, int]:
    value = int(steps)
    if not 1 <= value <= 100:
        raise ValueError("Stage R rollout must contain between 1 and 100 steps")
    return {"prediction_decode": value, "next_input_encode": value}
