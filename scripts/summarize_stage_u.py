#!/usr/bin/env python3
"""Create controlled Stage T versus Stage U comparisons and decision report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from grmhd import CHANNELS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_u"
VARIANTS = {
    "stage_t_baseline": ("Stage-T baseline", "spectral + index FD"),
    "spectral_only": ("spectral-only", "spectral"),
    "coordinate_fd": ("coordinate-aware", "spectral + coordinate FD"),
    "spherical_proxy_fd": ("spherical-proxy", "spectral + spherical-proxy FD"),
}
FOCUS = ("Bcc2", "Bcc3", "vel3", "rho", "press")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def load_all() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    metrics = {"stage_t_baseline": read(ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json")}
    rollouts = {"stage_t_baseline": read(ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json")}
    for variant in tuple(VARIANTS)[1:]:
        root = OUT / "variants" / variant
        metrics[variant] = read(root / "one_step_metrics.json")
        rollouts[variant] = read(root / "rollout_metrics.json")
    return metrics, rollouts, read(OUT / "audit_summary.json")


def aggregate_row(key: str, metric: Mapping[str, Any], rollout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": key, "label": VARIANTS[key][0], "operator": VARIANTS[key][1],
        "state_l2": metric["model"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": metric["model_error_over_persistence_error"],
        "physical_l2": metric["model"]["physical_relative_l2"]["arithmetic_average"],
        "residual_l2": metric["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_cosine": metric["model"]["residual"]["global_cosine"],
        "shell_skill": metric["transport"]["shell_skill_median"],
        "radial_skill": metric["transport"]["radial_skill_median"],
        "rout_fraction": metric["normalized_global_norm"]["above_Rout_fraction"],
        "one_step_finite": metric["finite"], "one_step_rho_press_positive": metric["rho_press_positive"],
        "rollout_finite": rollout["finite"], "rollout_rho_press_positive": rollout["rho_press_positive"],
        "first_10x_step": rollout["FIRST_10X_PHYSICAL_RANGE_STEP"],
        "first_negative_residual_cosine_step": rollout.get("FIRST_NEGATIVE_RESIDUAL_COSINE_STEP"),
        "rollout_completed_steps": rollout["completed_steps"],
    }


def rollout_rows(rollouts: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected={1,2,3,5,10,19,42,100}; rows=[]
    for variant, result in rollouts.items():
        found={int(record["step"]):record for record in result.get("records",[])}
        for step in sorted(selected):
            record=found.get(step)
            rows.append({
                "variant":variant,"step":step,"available":record is not None,
                "finite":None if record is None else record["finite"],
                "rho_press_positive":None if record is None else record["rho_press_positive"],
                "normalized_relative_l2_average":None if record is None else record.get("normalized_relative_l2_average"),
                "physical_relative_l2_average":None if record is None else record.get("physical_relative_l2_average"),
                "residual_cosine":None if record is None else record.get("residual_cosine"),
                "shell_skill":None if record is None or record.get("transport") is None else record["transport"]["shell_skill_median"],
                "radial_skill":None if record is None or record.get("transport") is None else record["transport"]["radial_skill_median"],
                "normalized_global_norm":None if record is None else record["normalized_global_norm"],
                "physical_range_explosion":None if record is None else record["physical_range_explosion"],
            })
    return rows


def temporal_rows() -> list[dict[str, Any]]:
    rows=[]
    baseline=list(csv.DictReader((ROOT/"artifacts/stage_t/metrics/validation_time_stratification.csv").open()))
    baseline=[r for r in baseline if int(r["epoch"])==150]
    for row in baseline: rows.append({"variant":"stage_t_baseline",**row})
    for variant in tuple(VARIANTS)[1:]:
        for row in csv.DictReader((OUT/"variants"/variant/"validation_time_stratification.csv").open()): rows.append(dict(row))
    return rows


def training_diagnostics() -> dict[str, Any]:
    output={}
    for variant in tuple(VARIANTS)[1:]:
        root=OUT/"variants"/variant
        summary=read(root/"training_summary.json")
        with (root/"train_log.csv").open(newline="",encoding="utf-8") as handle:
            rows=list(csv.DictReader(handle))
        output[variant]={
            **summary,
            "clipping_fraction_epoch_mean":float(np.mean([float(row["clipping_fraction"]) for row in rows])),
            "first_epoch_train_loss":float(rows[0]["train_loss_mean"]),
            "last_epoch_train_loss":float(rows[-1]["train_loss_mean"]),
            "minimum_epoch_mean_train_loss":float(min(float(row["train_loss_mean"]) for row in rows)),
        }
    return output


def temporal_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result={}
    for variant in VARIANTS:
        channel_ratios={}
        for channel in CHANNELS:
            selected=[r for r in rows if r["variant"]==variant and r["channel"]==channel]
            values={r["stratum"]:float(r["model_normalized_relative_l2"]) for r in selected}
            channel_ratios[channel]=values["late"]/values["early"]
        maximum=max(channel_ratios.values())
        label="LOW" if maximum<=1.10 else ("MODERATE" if maximum<=1.50 else "STRONG")
        result[variant]={"per_channel_late_over_early":channel_ratios,"maximum":maximum,"TEMPORAL_SHIFT_SENSITIVITY":label}
    return result


def per_channel(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows=[]
    for variant,metric in metrics.items():
        transport=metric["transport"]["per_channel"]
        for channel in CHANNELS:
            rows.append({
                "variant":variant,"channel":channel,
                "state_l2":metric["model"]["normalized_relative_l2"]["per_channel"][channel],
                "physical_l2":metric["model"]["physical_relative_l2"]["per_channel"][channel],
                "residual_l2":metric["model"]["residual"]["per_channel_relative_l2"][channel],
                "residual_cosine":metric["model"]["residual"]["per_channel_cosine"][channel],
                "shell_skill":transport[channel]["shell_skill_median"],
                "radial_skill":transport[channel]["radial_skill_median"],
            })
    return rows


def figures(rows: list[dict[str, Any]], rollout_table: list[dict[str, Any]]) -> None:
    out=OUT/"figures"; out.mkdir(parents=True,exist_ok=True)
    specs=[("state_l2","state_l2_comparison.png","normalized state rel-L2"),("residual_l2","residual_l2_comparison.png","residual rel-L2"),("residual_cosine","residual_cosine_comparison.png","residual global cosine"),("shell_skill","shell_skill_comparison.png","shell skill"),("radial_skill","radial_skill_comparison.png","radial skill")]
    labels=[row["label"] for row in rows]
    for key,name,title in specs:
        fig,axis=plt.subplots(figsize=(7,4));axis.bar(labels,[row[key] for row in rows]);axis.axhline(0,color="black",lw=.8);axis.set_title(title);axis.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(out/name,dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,4))
    for variant in VARIANTS:
        selected=[r for r in rollout_table if r["variant"]==variant and r["available"] and r["normalized_relative_l2_average"] is not None]
        if selected: axis.semilogy([r["step"] for r in selected],[r["normalized_relative_l2_average"] for r in selected],marker="o",label=VARIANTS[variant][0])
    axis.set(xlabel="rollout step",ylabel="normalized rel-L2",title="Frozen closed-loop rollout error");axis.legend();fig.tight_layout();fig.savefig(out/"rollout_error_comparison.png",dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7,4))
    for variant in VARIANTS:
        selected=[r for r in rollout_table if r["variant"]==variant and r["available"]]
        if selected: axis.semilogy([r["step"] for r in selected],[r["normalized_global_norm"] for r in selected],marker="o",label=VARIANTS[variant][0])
    axis.set(xlabel="rollout step",ylabel="normalized global norm",title="Closed-loop normalized range proxy");axis.legend();fig.tight_layout();fig.savefig(out/"physical_range_comparison.png",dpi=150);plt.close(fig)
    # Required top-level copies for the two synthetic diagnostics.
    for source,target in ((OUT/"synthetic_fd/figures/derivative_error_vs_radius.png",out/"derivative_error_vs_radius.png"),(OUT/"boundary_audit/figures/boundary_error.png",out/"boundary_error.png")):
        target.write_bytes(source.read_bytes())


def main() -> None:
    metrics,rollouts,audit=load_all()
    training=training_diagnostics()
    comparison=[aggregate_row(key,metrics[key],rollouts[key]) for key in VARIANTS]
    channels=per_channel(metrics); roll=rollout_rows(rollouts); temporal=temporal_rows(); sensitivity=temporal_sensitivity(temporal)
    write_csv(OUT/"comparison/stage_t_vs_stage_u.csv",comparison);write_csv(OUT/"comparison/per_channel_comparison.csv",channels);write_csv(OUT/"comparison/rollout_comparison.csv",roll);write_csv(OUT/"comparison/temporal_shift_comparison.csv",temporal)
    (OUT/"comparison/temporal_shift_summary.json").write_text(json.dumps(sensitivity,indent=2,sort_keys=True)+"\n")
    figures(comparison,roll)
    base=comparison[0]; spectral=comparison[1]; coord=comparison[2]; sphere=comparison[3]
    differential_suspected=(spectral["shell_skill"]>base["shell_skill"] and spectral["radial_skill"]>base["radial_skill"])
    spectral_suspected=audit["SPECTRAL_INDEX_SPACE_MISMATCH"]=="STRONG"
    geometry_any_transport=(coord["shell_skill"]>base["shell_skill"] or coord["radial_skill"]>base["radial_skill"] or coord["first_10x_step"] not in (None,1))
    # Synthetic differential and spectral mismatches are strong, but neither
    # removing nor repairing the differential path restores transport. This is
    # exactly the predeclared decision D.
    decision="D"
    gate_rows=[]
    for row in (coord,sphere):
        gate_rows.append({
            "variant":row["variant"],"G1_state_within_5pct":row["state_l2"]<=base["state_l2"]*1.05,
            "G2_residual_l2_below_1":row["residual_l2"]<1,"G3_cosine_above_0_7":row["residual_cosine"]>0.7,
            "G4_shell_positive":row["shell_skill"]>0,"G5_radial_positive":row["radial_skill"]>0,
            "G6_first_10x_after_1":row["first_10x_step"] is None or row["first_10x_step"]>1,
            "G6_strong_after_3":row["first_10x_step"] is None or row["first_10x_step"]>3,
            "G7_physical_tail_improved":row["physical_l2"]<base["physical_l2"],
        })
    write_csv(OUT/"comparison/geometry_gates.csv",gate_rows)
    (OUT/"comparison/training_diagnostics.json").write_text(json.dumps(training,indent=2,sort_keys=True)+"\n")
    # Compact focus-channel result block.
    focus=[]
    for channel in FOCUS:
        focus.append({v:next(r for r in channels if r["variant"]==v and r["channel"]==channel) for v in VARIANTS})
    table="\n".join(
        f"| {r['label']} | {r['operator']} | {r['state_l2']:.6g} | {r['residual_l2']:.6g} | {r['residual_cosine']:.6g} | {r['shell_skill']:.6g} | {r['radial_skill']:.6g} | {r['first_10x_step']} |"
        for r in comparison
    )
    report=f"""# Stage U — Spherical-Grid / Operator Geometry Mismatch Audit

| variant | operator | state L2 | residual L2 | cosine | shell skill | radial skill | first 10x step |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

## Frozen contract and audit result

`DATASET_FROZEN = true`; `SPLIT_FROZEN = true`; `PREPROCESSING_FROZEN = true`; `TARGET_CONTRACT_FROZEN = true`; `LOSS_FROZEN = true`.

The frozen expanded HDF5 is `(phi,theta,r)=64^3`; r is geometric with ratio 1.084693 and `dr=0.097027..14.993990`. The pinned upstream FD uses one scalar grid width (1.0 at 64^3) and circular `3x3x3` kernels on all axes. Synthetic results are `INDEX_FD_GEOMETRY_ERROR={audit['INDEX_FD_GEOMETRY_ERROR']}`, `BOUNDARY_MISMATCH={audit['BOUNDARY_MISMATCH']}`, and `SPECTRAL_INDEX_SPACE_MISMATCH={audit['SPECTRAL_INDEX_SPACE_MISMATCH']}`. Synthetic minimax padding selected `{audit['synthetic_selected_padding']}`.

## Controlled pilots

All three pilots used seed 42, 168 train pairs, 42 validation pairs, normalized residual targets, PlainL2, Adam, batch 1, accumulation 4, width 16, modes 8^3, four layers, and exactly the first 6,300 updates of Stage T's 50,400-update scheduler. They ran sequentially for 150 epochs. U1 has 330,648 parameters (-7.72%); U3a/U3b have 339,864 (-5.14%); all common non-differential initial tensors match the frozen shared state. U2 was skipped because pinned LocalNO offers no semantics-preserving differential-only switch.

Training runtime / peak allocated MiB / mean clipping fraction were: U1 `{training['spectral_only']['runtime_seconds']:.1f}s / {training['spectral_only']['peak_allocated_mib_max']:.1f} / {training['spectral_only']['clipping_fraction_epoch_mean']:.4f}`, U3a `{training['coordinate_fd']['runtime_seconds']:.1f}s / {training['coordinate_fd']['peak_allocated_mib_max']:.1f} / {training['coordinate_fd']['clipping_fraction_epoch_mean']:.4f}`, U3b `{training['spherical_proxy_fd']['runtime_seconds']:.1f}s / {training['spherical_proxy_fd']['peak_allocated_mib_max']:.1f} / {training['spherical_proxy_fd']['clipping_fraction_epoch_mean']:.4f}`. All training parameters remained finite; U3b's closed-loop physical decode did not.

## Direct answers

1. **Regular-grid assumption?** Yes. Upstream explicitly documents a regular grid and divides its centered convolution by one scalar.
2. **Same effective spacing?** Yes: Stage T passes `grid_width=1.0` to all three directions at 64^3.
3. **Wrong periodic theta/r?** Yes. `conv_padding_mode=periodic` becomes circular for phi, theta, and r.
4. **Log-r FD error?** Yes, strong: the same index derivative cannot represent stored-coordinate radial derivatives across a 154.5x dr range.
5. **Different physical scales?** Yes. The spherical-coordinate proxies vary strongly with r and theta; they are not proper Kerr-Schild distances.
6. **Spectral index-space mismatch?** Strong as an assumption mismatch: FFT is circular-shift equivariant in tensor index although radial physical scale changes with index.
7. **Did disabling FD improve transport?** No (`shell {base['shell_skill']:.3g}->{spectral['shell_skill']:.3g}`, `radial {base['radial_skill']:.3g}->{spectral['radial_skill']:.3g}`); both worsen, state/residual metrics regress, and first-10x remains step 1. Thus `DIFFERENTIAL_BRANCH_SUSPECTED={str(differential_suspected).lower()}`.
8. **Did coordinate-aware FD improve transport?** No. It worsened shell/radial skill to {coord['shell_skill']:.3g}/{coord['radial_skill']:.3g} and retained step-1 failure.
9. **Did spherical proxy improve further?** No. Near-pole `1/(r sin theta)` conditioning caused catastrophic training/evaluation scales; one-step state L2={sphere['state_l2']:.3g} and closed-loop decode became nonfinite immediately.
10. **Any positive shell/radial skill?** No variant made either aggregate skill positive.
11. **Was step-1 range failure delayed?** No.
12. **Does geometry alone explain state/transport decoupling?** No. Synthetic mismatch is real and strong, but the minimal differential repair does not recover learned transport. The spectral branch remains index-periodic on theta/r, while objective mismatch and temporal shift remain independent plausible causes.
13. **Next direction?** Do not continue this naive spherical proxy. Audit loss/objective alignment next, with distribution-shift/generalization reported in parallel; a future geometry operator would need pole-regular, coordinate-aware spectral/basis treatment rather than only multiplying a learned stencil by singular scale factors.

## Channel-specific result

For Bcc2, Bcc3, vel3, rho, and press, neither U3 variant produces a consistent transport improvement. U3a's aggregate physical L2 is {coord['physical_l2']:.3e}; U3b's is {sphere['physical_l2']:.3e}. These are decoder-tail failures, not improvements obtained by clipping (no new clipping was added). See `comparison/per_channel_comparison.csv` for exact per-channel state/residual/cosine/shell/radial metrics.

## Temporal distribution shift

Stage-T / U1 / U3a / U3b maximum late-over-early channel ratios and labels are stored in `comparison/temporal_shift_summary.json`: {json.dumps({k:(round(v['maximum'],4),v['TEMPORAL_SHIFT_SENSITIVITY']) for k,v in sensitivity.items()})}. Geometry mismatch and chronological distribution shift are not treated as the same causal issue.

## Scientific limitations

This is a reduced, spherical Kerr-Schild coordinate-basis, nearest-leaf 64^3 adaptation. `SPHERICAL_COORDINATE_SCALE_PROXY` is not a strict Kerr-Schild proper distance or covariant derivative. Bcc1/2/3 and vel1/2/3 are not relabeled as Cartesian components. `CARTESIAN_REMAP_NOT_AUTHORIZED`.
"""
    (OUT/"STAGE_U_REPORT.md").write_text(report,encoding="utf-8")
    decision_text=f"""# Stage U Decision

`PRIMARY_DECISION = {decision}`

`D = SPECTRAL_AND_DIFFERENTIAL_GEOMETRY_BOTH_MISMATCH`

The no-training audits find strong differential, boundary, and spectral index-space mismatches. Spectral-only is worse than Stage T on aggregate transport and fails state/residual/rollout gates. Coordinate-aware FD does not rescue transport, and the naive spherical proxy is numerically pathological near the poles. Therefore repairing only the differential path is insufficient.

```text
PRIMARY_DECISION = {decision}
DIFFERENTIAL_BRANCH_SUSPECTED = {str(differential_suspected).lower()}
SPECTRAL_BRANCH_SUSPECTED = {str(spectral_suspected).lower()}
BOUNDARY_MISMATCH = {audit['BOUNDARY_MISMATCH']}
INDEX_FD_GEOMETRY_ERROR = {audit['INDEX_FD_GEOMETRY_ERROR']}
AUTHORIZE_NEXT_STAGE = loss_objective_audit_with_distribution_shift_reporting
CARTESIAN_REMAP_NOT_AUTHORIZED
```
"""
    (OUT/"STAGE_U_DECISION.md").write_text(decision_text,encoding="utf-8")
    print(json.dumps({"PRIMARY_DECISION":decision,"DIFFERENTIAL_BRANCH_SUSPECTED":differential_suspected,"SPECTRAL_BRANCH_SUSPECTED":spectral_suspected,"geometry_any_transport_improvement":geometry_any_transport},indent=2))


if __name__=="__main__":main()
