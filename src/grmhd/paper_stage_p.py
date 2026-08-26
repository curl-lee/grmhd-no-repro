"""Read-only contracts for the Stage P closed-loop attribution audit."""

from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from . import CHANNELS
from .paper_preprocessing import PAPER_TRANSFORMS
from .paper_stage_n import MINIMAL_INVERSE_CLAMP, NO_SOFTCLIP, PrototypePreprocessor


STAGE_P_CLASSIFICATION = "post_hoc_closed_loop_failure_attribution"
SELECTED_GT_STEPS = (1, 2, 3, 5, 10, 19)
SELECTED_NO_GT_STEPS = (25, 50, 75, 100)
TARGET_CHANNELS = ("Bcc2", "Bcc3", "vel3")
TRAIN_QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
DECODER_EPSILONS = (1.0e-5, 1.0e-4)
LOCAL_GAIN_EPSILONS = (1.0e-3, 1.0e-2)
DRIVER_PRIMARY_THRESHOLD = 0.5
DRIVER_WEAK_THRESHOLD = 0.2
FROZEN_HISTORY = {
    "Stage K": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    "Stage L": "3. MIXED_OVERALL",
    "Stage M": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
    "Stage N": "P3 ready; mixed operator-response failure",
    "Stage O": "C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
}


def load_stage_o_model_only(
    checkpoint_dir: str | Path,
    *,
    config,
    model: torch.nn.Module,
    expected_config_checksum: str,
) -> tuple[torch.nn.Module, int, Mapping[str, Any]]:
    """Strictly load only the frozen model; never construct training state."""

    from neuralop.layers.spectral_convolution import SpectralConv

    from .paper_checkpoint import validate_paper_checkpoint_metadata

    directory = Path(checkpoint_dir)
    metadata = json.loads(
        (directory / "paper_grmhd_metadata.json").read_text(encoding="utf-8")
    )
    validate_paper_checkpoint_metadata(
        metadata, config, expected_config_checksum=expected_config_checksum
    )
    manifest = torch.load(
        directory / "manifest.pt", map_location="cpu", weights_only=True
    )
    if manifest.get("optimizer") is None or manifest.get("scheduler") is None:
        raise ValueError("Frozen Stage O manifest lost its training-state references")
    epoch = int(manifest["epoch"])
    if epoch != int(metadata["epoch"]):
        raise ValueError("Frozen Stage O manifest/sidecar epoch mismatch")
    with torch.serialization.safe_globals([torch._C._nn.gelu, SpectralConv]):
        state = torch.load(
            directory / str(manifest["model"]),
            map_location="cpu",
            weights_only=True,
        )
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Frozen Stage O model did not load strictly")
    model.eval()
    return model, epoch, metadata


def relative_l2(left: np.ndarray, right: np.ndarray, *, epsilon: float = 1e-30) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    denominator = max(float(np.linalg.norm(np.asarray(right, dtype=np.float64))), epsilon)
    return float(np.linalg.norm(delta) / denominator)


def distribution_summary(values: np.ndarray) -> dict[str, float | bool]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(array)
    if not finite.all():
        raise FloatingPointError("Stage P distribution contains NaN/Inf")
    quantiles = np.quantile(array, TRAIN_QUANTILES)
    return {
        "finite": True,
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "absolute_maximum": float(np.max(np.abs(array))),
        "q001": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "q999": float(quantiles[4]),
        "robust_span_q999_q001": float(quantiles[4] - quantiles[0]),
    }


def ood_diagnostics(values: np.ndarray, envelope: Mapping[str, float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q_low = float(envelope["q001"])
    q_high = float(envelope["q999"])
    minimum = float(envelope["minimum"])
    maximum = float(envelope["maximum"])
    outside_q = (array < q_low) | (array > q_high)
    outside_range = (array < minimum) | (array > maximum)
    excess = np.maximum(np.maximum(q_low - array, array - q_high), 0.0)
    range_excess = np.maximum(np.maximum(minimum - array, array - maximum), 0.0)
    span = max(q_high - q_low, 1e-30)
    return {
        "fraction_outside_train_q001_q999": float(outside_q.mean()),
        "fraction_outside_train_min_max": float(outside_range.mean()),
        "maximum_normalized_excess": float(excess.max()),
        "maximum_normalized_excess_over_robust_span": float(excess.max() / span),
        "maximum_minmax_excess": float(range_excess.max()),
    }


def _decode_z_and_derivative(
    preprocessor: PrototypePreprocessor, values: np.ndarray, channel: int
) -> tuple[np.ndarray, np.ndarray]:
    item = np.asarray(values, dtype=np.float64)
    policy = preprocessor.spec.channel_policies[channel]
    if policy == NO_SOFTCLIP:
        return item, np.ones_like(item)
    limit = (
        preprocessor.minimal_limit
        if policy == MINIMAL_INVERSE_CLAMP
        else preprocessor.inverse_clamp_fraction * preprocessor.gamma
    )
    clipped = np.clip(item, -limit, limit)
    z = preprocessor.gamma * np.arctanh(clipped / preprocessor.gamma)
    interior = np.abs(item) < limit
    derivative = np.zeros_like(item)
    derivative[interior] = 1.0 / (
        1.0 - np.square(item[interior] / preprocessor.gamma)
    )
    return z, derivative


def decoder_derivative_numpy(
    preprocessor: PrototypePreprocessor,
    normalized: np.ndarray,
    *,
    channel_axis: int = 0,
) -> np.ndarray:
    """Analytic elementwise derivative of the frozen P3 decoder."""

    array = np.asarray(normalized)
    axis = channel_axis % array.ndim
    if array.shape[axis] != len(CHANNELS):
        raise ValueError("Stage P decoder derivative requires eight channels")
    output = np.empty(array.shape, dtype=np.float64)
    for channel, _ in enumerate(PAPER_TRANSFORMS):
        selection = [slice(None)] * array.ndim
        selection[axis] = channel
        key = tuple(selection)
        output[key] = decoder_derivative_channel(
            preprocessor, array[key], channel=channel, source_dtype=array.dtype
        )
    if not np.isfinite(output).all():
        raise FloatingPointError("Stage P decoder derivative contains NaN/Inf")
    return output


def decoder_derivative_channel(
    preprocessor: PrototypePreprocessor,
    normalized: np.ndarray,
    *,
    channel: int,
    source_dtype: np.dtype[Any] = np.dtype(np.float32),
) -> np.ndarray:
    """Memory-bounded analytic P3 decoder derivative for one channel."""

    if not 0 <= int(channel) < len(CHANNELS):
        raise ValueError("Stage P decoder derivative channel is invalid")
    item = np.asarray(normalized, dtype=np.float64)
    z, dz_dy = _decode_z_and_derivative(preprocessor, item, int(channel))
    scale = float(preprocessor.base.scale[channel])
    transformed = z * scale + float(preprocessor.base.median[channel])
    epsilon = float(preprocessor.base.epsilon[channel])
    dtype = np.dtype(source_dtype)
    physical_limit = float(np.finfo(dtype if dtype.kind == "f" else np.float32).max)
    kind = PAPER_TRANSFORMS[channel]
    if kind == "positive_log":
        exponent_limit = math.log10(physical_limit + epsilon)
        active = transformed < exponent_limit
        derivative = np.zeros_like(transformed)
        derivative[active] = (
            math.log(10.0)
            * np.power(10.0, transformed[active])
            * scale
            * dz_dy[active]
        )
    elif kind == "signed_log":
        exponent_limit = math.log10(physical_limit) - math.log10(epsilon)
        active = np.abs(transformed) < exponent_limit
        derivative = np.zeros_like(transformed)
        derivative[active] = (
            math.log(10.0)
            * epsilon
            * np.power(10.0, np.abs(transformed[active]))
            * scale
            * dz_dy[active]
        )
    else:
        active = np.abs(transformed) < physical_limit
        derivative = np.where(active, scale * dz_dy, 0.0)
    if not np.isfinite(derivative).all():
        raise FloatingPointError("Stage P channel decoder derivative contains NaN/Inf")
    return derivative


def finite_difference_decoder_derivative(
    preprocessor: PrototypePreprocessor,
    normalized: np.ndarray,
    *,
    epsilon: float,
    channel_axis: int = 0,
) -> np.ndarray:
    if epsilon not in DECODER_EPSILONS:
        raise ValueError("Stage P decoder epsilon is frozen to 1e-5 or 1e-4")
    array = np.asarray(normalized, dtype=np.float64)
    plus = preprocessor.decode_numpy(array + epsilon, channel_axis=channel_axis)
    minus = preprocessor.decode_numpy(array - epsilon, channel_axis=channel_axis)
    return (plus.astype(np.float64) - minus.astype(np.float64)) / (2.0 * epsilon)


def roundtrip_only(
    preprocessor: PrototypePreprocessor,
    normalized: np.ndarray,
    *,
    applications: int,
    channel_axis: int = 0,
) -> np.ndarray:
    if applications not in (1, 2):
        raise ValueError("Stage P H may be applied only once or twice")
    state = np.asarray(normalized).copy()
    for _ in range(applications):
        physical = preprocessor.decode_numpy(state, channel_axis=channel_axis)
        state = preprocessor.encode_numpy(physical, channel_axis=channel_axis)
    return state


def project_train_envelope(
    normalized: torch.Tensor,
    envelope: Mapping[str, Mapping[str, float]],
    channels: Iterable[str],
) -> torch.Tensor:
    selected = tuple(channels)
    if not set(selected) <= set(CHANNELS):
        raise ValueError("Unknown Stage P projection channel")
    output = normalized.clone()
    for name in selected:
        channel = CHANNELS.index(name)
        output[:, channel].clamp_(
            min=float(envelope[name]["q001"]), max=float(envelope[name]["q999"])
        )
    return output


def reset_channels(
    prediction: torch.Tensor, target: torch.Tensor, channels: Iterable[str]
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[1] != 8:
        raise ValueError("Stage P reset states must have matching (B,8,...) shapes")
    selected = tuple(channels)
    if not set(selected) <= set(CHANNELS):
        raise ValueError("Unknown Stage P reset channel")
    output = prediction.clone()
    for name in selected:
        output[:, CHANNELS.index(name)] = target[:, CHANNELS.index(name)]
    return output


def first_failure(events: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for record in events:
        if bool(record.get(key, False)):
            return int(record["step"])
    return None


def driver_support(reductions: Sequence[float]) -> str:
    values = [float(value) for value in reductions if np.isfinite(value)]
    primary = sum(value >= DRIVER_PRIMARY_THRESHOLD for value in values)
    weak = sum(DRIVER_WEAK_THRESHOLD <= value < DRIVER_PRIMARY_THRESHOLD for value in values)
    if primary >= 4:
        return "supported_primary_driver"
    if primary + weak >= 4:
        return "weak_driver"
    return "not_supported"


def mechanism_label(*, supported: int, weak: int = 0) -> str:
    if supported >= 2:
        return "supported"
    if supported == 1 or weak >= 2:
        return "weakly_supported"
    if supported == 0 and weak == 0:
        return "not_supported"
    return "inconclusive"


def final_mechanism_decision(mechanisms: Mapping[str, str]) -> str:
    supported = [name for name, status in mechanisms.items() if status == "supported"]
    mapping = {
        "M1": "1. NORMALIZED_MODEL_OVERSHOOT_DOMINATED",
        "M2": "2. P3_INVERSE_TAIL_AMPLIFICATION_DOMINATED",
        "M3": "3. ROUNDTRIP_FEEDBACK_DOMINATED",
        "M4": "4. RECURRENT_OPERATOR_GAIN_DOMINATED",
        "M5": "5. CHANNEL_COUPLING_DOMINATED",
    }
    if len(supported) == 1:
        return mapping[supported[0]]
    if len(supported) >= 2:
        return "6. MIXED_CLOSED_LOOP_FAILURE"
    return "7. INCONCLUSIVE_OR_ENGINEERING_FAILURE"


def normalized_direct_step(
    model: torch.nn.Module, normalized: torch.Tensor, shells: torch.Tensor
) -> torch.Tensor:
    """One diagnostic step with no decoder or encoder call."""

    with torch.no_grad():
        return model(x=torch.cat((normalized, shells), dim=1))


def teacher_forced_predictions(
    model: torch.nn.Module,
    oracle_inputs: Sequence[torch.Tensor],
    shells: torch.Tensor,
) -> list[torch.Tensor]:
    """Predict each oracle input independently; no prediction is fed back."""

    outputs = []
    with torch.no_grad():
        for state in oracle_inputs:
            outputs.append(model(x=torch.cat((state, shells), dim=1)))
    return outputs


def directional_gain(
    mapping,
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Finite-difference directional gain without parameter backward."""

    if epsilon not in LOCAL_GAIN_EPSILONS:
        raise ValueError("Stage P local-gain epsilon is frozen to 1e-3 or 1e-2")
    if state.shape != direction.shape:
        raise ValueError("Stage P local-gain state/direction shapes differ")
    direction_norm = torch.linalg.vector_norm(direction.to(dtype=torch.float64))
    if float(direction_norm) == 0.0:
        raise ValueError("Stage P local-gain direction is zero")
    with torch.no_grad():
        baseline = mapping(state)
        perturbed = mapping(state + epsilon * direction)
    response = (perturbed - baseline) / epsilon
    gain = torch.linalg.vector_norm(response.to(dtype=torch.float64)) / direction_norm
    return {
        "gain": float(gain.detach().cpu()),
        "baseline": baseline.detach(),
        "response": response.detach(),
        "used_backward": False,
    }
