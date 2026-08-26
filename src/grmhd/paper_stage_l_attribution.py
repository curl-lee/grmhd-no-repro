"""Deterministic post-hoc structure metrics for Stage L collapse attribution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


AXES = ("phi", "theta", "r")
SPECTRAL_BANDS = {
    "low_k": (None, 0.125),
    "mid_k": (0.125, 0.25),
    "high_k": (0.25, None),
}


def _field(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 3:
        raise ValueError("Stage L field metrics require exactly (phi,theta,r)")
    if not np.isfinite(result).all():
        raise FloatingPointError("Stage L field contains NaN/Inf")
    return result


def safe_retention(
    numerator: float, denominator: float, *, epsilon: float
) -> tuple[float | None, bool]:
    """Return a ratio and explicit undefined flag without inventing a value."""

    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise FloatingPointError("Stage L retention inputs must be finite")
    if abs(denominator) <= float(epsilon):
        return None, True
    return numerator / denominator, False


def decompose_metric(
    raw: float, oracle: float, model: float, *, epsilon: float
) -> dict[str, Any]:
    pre, pre_undefined = safe_retention(oracle, raw, epsilon=epsilon)
    model_ratio, model_undefined = safe_retention(model, oracle, epsilon=epsilon)
    total, total_undefined = safe_retention(model, raw, epsilon=epsilon)
    return {
        "raw": float(raw),
        "oracle": float(oracle),
        "model": float(model),
        "preprocessing_retention": pre,
        "model_retention": model_ratio,
        "total_retention": total,
        "preprocessing_undefined": pre_undefined,
        "model_undefined": model_undefined,
        "total_undefined": total_undefined,
    }


def relative_l2(left: np.ndarray, right: np.ndarray, *, epsilon: float) -> float | None:
    left = _field(left)
    right = _field(right)
    denominator = float(np.linalg.norm(right.ravel()))
    if denominator <= epsilon:
        return None
    return float(np.linalg.norm((left - right).ravel()) / denominator)


def pair_metrics(
    prediction: np.ndarray, reference: np.ndarray, *, epsilon: float
) -> dict[str, float | None]:
    prediction = _field(prediction).ravel()
    reference = _field(reference).ravel()
    pred_centered = prediction - prediction.mean()
    ref_centered = reference - reference.mean()
    pearson_denominator = float(
        np.linalg.norm(pred_centered) * np.linalg.norm(ref_centered)
    )
    cosine_denominator = float(np.linalg.norm(prediction) * np.linalg.norm(reference))
    return {
        "pearson_correlation": (
            None
            if pearson_denominator <= epsilon
            else float(np.dot(pred_centered, ref_centered) / pearson_denominator)
        ),
        "cosine_similarity": (
            None
            if cosine_denominator <= epsilon
            else float(np.dot(prediction, reference) / cosine_denominator)
        ),
        "sign_agreement": float(np.mean(np.signbit(prediction) == np.signbit(reference))),
        "relative_l2": relative_l2(
            prediction.reshape((-1, 1, 1)),
            reference.reshape((-1, 1, 1)),
            epsilon=epsilon,
        ),
    }


def basic_field_metrics(
    values: np.ndarray,
    *,
    region_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    values = _field(values)
    flat = values.ravel()
    quantiles = np.quantile(flat, [0.01, 0.10, 0.50, 0.90, 0.99])
    differences = [np.diff(values, axis=axis) for axis in range(3)]
    variation = [float(np.mean(np.abs(item))) for item in differences]
    gradient_energy = [float(np.mean(np.square(item))) for item in differences]
    rms = float(np.sqrt(np.mean(np.square(flat))))
    near_zero_threshold = max(1.0e-30, 1.0e-6 * rms)
    output = {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat, ddof=0)),
        "variance": float(np.var(flat, ddof=0)),
        "minimum": float(np.min(flat)),
        "maximum": float(np.max(flat)),
        "q01": float(quantiles[0]),
        "q10": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q90": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "dynamic_span_q99_q01": float(quantiles[4] - quantiles[0]),
        "dynamic_range_max_min": float(np.max(flat) - np.min(flat)),
        "rms": rms,
        "near_zero_threshold": near_zero_threshold,
        "near_zero_occupancy": float(np.mean(np.abs(flat) <= near_zero_threshold)),
        "positive_fraction": float(np.mean(flat > 0)),
        "negative_fraction": float(np.mean(flat < 0)),
        "zero_fraction": float(np.mean(flat == 0)),
        "sign_balance": float(np.mean(flat > 0) - np.mean(flat < 0)),
        "total_variation": float(np.mean(variation)),
        "gradient_energy": float(np.sum(gradient_energy)),
        "phi_variation": variation[0],
        "theta_variation": variation[1],
        "r_variation": variation[2],
        "phi_gradient_energy": gradient_energy[0],
        "theta_gradient_energy": gradient_energy[1],
        "r_gradient_energy": gradient_energy[2],
    }
    if region_masks:
        for name, mask in region_masks.items():
            selected = values[np.asarray(mask, dtype=bool)]
            if selected.size == 0:
                raise ValueError(f"Stage L region is empty: {name}")
            output[f"{name}_variance"] = float(np.var(selected, ddof=0))
            output[f"{name}_std"] = float(np.std(selected, ddof=0))
    return output


def make_region_masks(
    *, theta: np.ndarray, shell_index: np.ndarray, spatial_shape: tuple[int, int, int]
) -> dict[str, np.ndarray]:
    nphi, ntheta, nr = spatial_shape
    theta = np.asarray(theta, dtype=np.float64)
    shell_index = np.asarray(shell_index, dtype=np.int64)
    if theta.shape != (ntheta,) or shell_index.shape != (nr,):
        raise ValueError("Stage L region coordinates do not match the field shape")
    radial = np.broadcast_to(shell_index.reshape(1, 1, nr), spatial_shape)
    polar_1d = (theta <= np.pi / 6.0) | (theta >= 5.0 * np.pi / 6.0)
    polar = np.broadcast_to(polar_1d.reshape(1, ntheta, 1), spatial_shape)
    return {
        "center_region": radial <= 1,
        "polar_region": polar,
        "outer_shell_region": radial >= 6,
    }


def shell_metrics(
    values: np.ndarray, shell_index: np.ndarray, *, n_shells: int = 8
) -> list[dict[str, Any]]:
    values = _field(values)
    shell_index = np.asarray(shell_index, dtype=np.int64)
    if shell_index.shape != (values.shape[-1],):
        raise ValueError("Stage L shell index must align only with the radial axis")
    rows = []
    for shell in range(n_shells):
        radial_selection = shell_index == shell
        if not np.any(radial_selection):
            raise ValueError(f"Frozen shell {shell + 1} has no radial cells")
        selected = values[..., radial_selection]
        metrics = basic_field_metrics(selected)
        high_k = spectrum_metrics(selected, axis="combined", demean=True)["high_k_energy"]
        rows.append(
            {
                "shell": shell + 1,
                "shell_group": (
                    "inner" if shell < 2 else "middle" if shell < 6 else "outer"
                ),
                "radial_cell_count": int(radial_selection.sum()),
                "voxel_count": int(selected.size),
                "mean": metrics["mean"],
                "std": metrics["std"],
                "variance": metrics["variance"],
                "q01": metrics["q01"],
                "q50": metrics["q50"],
                "q99": metrics["q99"],
                "dynamic_span_q99_q01": metrics["dynamic_span_q99_q01"],
                "total_variation": metrics["total_variation"],
                "high_k_energy": high_k,
            }
        )
    return rows


def radial_profile(values: np.ndarray) -> dict[str, Any]:
    values = _field(values)
    profile = np.mean(values, axis=(0, 1))
    return {
        "profile": profile.tolist(),
        "length": int(profile.size),
        "variance": float(np.var(profile, ddof=0)),
        "std": float(np.std(profile, ddof=0)),
        "dynamic_span": float(np.max(profile) - np.min(profile)),
    }


def _frequency_grid(shape: tuple[int, int, int], axis: str) -> np.ndarray:
    frequencies = [np.abs(np.fft.fftfreq(size)) for size in shape]
    if axis == "combined":
        grids = np.meshgrid(*frequencies, indexing="ij")
        return np.maximum.reduce(grids)
    if axis not in AXES:
        raise ValueError(f"Unknown Stage L spectral axis: {axis}")
    index = AXES.index(axis)
    reshape = [1, 1, 1]
    reshape[index] = shape[index]
    return np.broadcast_to(frequencies[index].reshape(reshape), shape)


def spectrum_metrics(
    values: np.ndarray, *, axis: str, demean: bool
) -> dict[str, Any]:
    values = _field(values)
    work = values - values.mean() if demean else values.copy()
    if axis == "combined":
        transformed = np.fft.fftn(work, axes=(0, 1, 2), norm="ortho")
    else:
        transformed = np.fft.fft(work, axis=AXES.index(axis), norm="ortho")
    energy = np.square(np.abs(transformed))
    frequency = _frequency_grid(values.shape, axis)
    spatial_energy = float(np.sum(np.square(work)))
    spectral_energy = float(np.sum(energy))
    parseval_relative_error = abs(spatial_energy - spectral_energy) / max(
        spatial_energy, 1.0e-300
    )
    low = frequency <= 0.125
    mid = (frequency > 0.125) & (frequency <= 0.25)
    high = frequency > 0.25
    if np.any((low.astype(int) + mid.astype(int) + high.astype(int)) != 1):
        raise RuntimeError("Stage L spectral bands overlap or leave gaps")
    nonzero = frequency > 0
    nonzero_energy = float(np.sum(energy[nonzero]))
    centroid = (
        None
        if nonzero_energy <= 1.0e-300
        else float(np.sum(frequency[nonzero] * energy[nonzero]) / nonzero_energy)
    )
    return {
        "axis": axis,
        "demeaned": bool(demean),
        "window": "none",
        "normalization": "ortho",
        "frequency_metric": (
            "max_abs_cycles_per_index" if axis == "combined" else "abs_cycles_per_index"
        ),
        "low_k_energy": float(np.sum(energy[low])),
        "mid_k_energy": float(np.sum(energy[mid])),
        "high_k_energy": float(np.sum(energy[high])),
        "total_energy": spectral_energy,
        "total_nonzero_mode_energy": nonzero_energy,
        "spectral_centroid": centroid,
        "parseval_spatial_energy": spatial_energy,
        "parseval_spectral_energy": spectral_energy,
        "parseval_relative_error": float(parseval_relative_error),
        "bin_count_low": int(np.count_nonzero(low)),
        "bin_count_mid": int(np.count_nonzero(mid)),
        "bin_count_high": int(np.count_nonzero(high)),
    }


def all_spectrum_metrics(values: np.ndarray) -> list[dict[str, Any]]:
    return [
        spectrum_metrics(values, axis=axis, demean=demean)
        for axis in (*AXES, "combined")
        for demean in (False, True)
    ]


def require_oracle(*, ground_truth_available: bool) -> None:
    if not ground_truth_available:
        raise ValueError("Stage L no-GT analysis cannot request an oracle")


@dataclass(frozen=True)
class ChannelAttribution:
    choice: str
    preprocessing_severe_count: int
    model_severe_count: int
    reason: str


def classify_channel(
    *,
    preprocessing_severe: Mapping[str, bool],
    model_severe: Mapping[str, bool],
    oracle_detector_triggered: bool,
    localno_detector_triggered: bool,
    oracle_degraded: bool = False,
    engineering_failure: bool = False,
) -> ChannelAttribution:
    required = {"global_variance", "shell_radial_variance", "high_k_spectral_energy"}
    if set(preprocessing_severe) != required or set(model_severe) != required:
        raise ValueError("Stage L attribution requires exactly three core evidence classes")
    pre_count = sum(bool(value) for value in preprocessing_severe.values())
    model_count = sum(bool(value) for value in model_severe.values())
    if engineering_failure:
        return ChannelAttribution(
            "E. INCONCLUSIVE_OR_ENGINEERING_FAILURE",
            pre_count,
            model_count,
            "predefined engineering/provenance failure",
        )
    if (pre_count >= 2 and model_count >= 2) or (
        oracle_degraded and pre_count >= 1 and model_count >= 1
    ):
        return ChannelAttribution(
            "C. MIXED_PREPROCESSING_AND_MODEL",
            pre_count,
            model_count,
            "both preprocessing and LocalNO add material structural degradation",
        )
    if pre_count >= 2 and model_count < 2 and oracle_detector_triggered:
        return ChannelAttribution(
            "A. PREPROCESSING_DOMINATED",
            pre_count,
            model_count,
            "preprocessing is severe in at least two classes and oracle triggers detector",
        )
    if (
        model_count >= 2
        and pre_count < 2
        and not oracle_detector_triggered
        and localno_detector_triggered
    ):
        return ChannelAttribution(
            "B. MODEL_DOMINATED",
            pre_count,
            model_count,
            "LocalNO is severe in at least two classes while oracle does not collapse",
        )
    if pre_count < 2 and model_count < 2 and localno_detector_triggered:
        return ChannelAttribution(
            "D. DETECTOR_SPECIFIC_MISMATCH",
            pre_count,
            model_count,
            "detector triggers without two-class structural severe loss",
        )
    return ChannelAttribution(
        "E. INCONCLUSIVE_OR_ENGINEERING_FAILURE",
        pre_count,
        model_count,
        "frozen evidence does not satisfy a stable attribution rule",
    )


def overall_attribution(decisions: Mapping[str, str]) -> str:
    if set(decisions) != {"Bcc3", "vel3"}:
        raise ValueError("Overall Stage L attribution requires Bcc3 and vel3")
    values = set(decisions.values())
    if any(value.startswith("E.") for value in values):
        return "5. INCONCLUSIVE"
    if values == {"A. PREPROCESSING_DOMINATED"}:
        return "1. PREPROCESSING_DOMINATED_OVERALL"
    if values == {"B. MODEL_DOMINATED"}:
        return "2. MODEL_DOMINATED_OVERALL"
    if any(value.startswith("C.") for value in values) or values == {
        "A. PREPROCESSING_DOMINATED",
        "B. MODEL_DOMINATED",
    }:
        return "3. MIXED_OVERALL"
    if values == {"D. DETECTOR_SPECIFIC_MISMATCH"}:
        return "4. DETECTOR_REQUIRES_SEPARATE_REVIEW"
    return "5. INCONCLUSIVE"
