#!/usr/bin/env python3
"""Run one sequential 150-epoch Stage U geometry-only GPU pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_s_training import (
    accumulation_groups,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)
from grmhd.stage_u_training import (
    VARIANTS,
    build_stage_u_model,
    load_coords,
    sha256_json,
    variant_parameter_difference,
)

from train_stage_s import StageSBatchPath, gradient_norm
from train_stage_t import (
    atomic_csv,
    atomic_json,
    checkpoint_payload,
    load_rows,
    save_checkpoint,
    verify_frozen_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def load_stage_u(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    stage_u = yaml.safe_load(path.read_text(encoding="utf-8"))
    stage_t_path = ROOT / stage_u["frozen_stage_t_config"]
    if sha256_file(stage_t_path) != stage_u["frozen_stage_t_config_sha256"]:
        raise ValueError("Frozen Stage T config changed")
    stage_t, stage_s, freeze, train_pairs, _, _ = verify_frozen_contract(stage_t_path)
    audit = json.loads((ROOT / "artifacts/stage_u/audit_summary.json").read_text())
    if not audit["AUTHORIZE_U3"]:
        raise RuntimeError("No-training audit did not authorize geometry-aware variants")
    freeze.update({
        "TARGET_CONTRACT_FROZEN": True,
        "LOSS_FROZEN": True,
        "STAGE_T_CONFIG_FROZEN": True,
        "MODEL_FROZEN_EXCEPT_GEOMETRY": True,
    })
    return stage_u, stage_s, freeze, train_pairs


def prepare(
    variant: str,
    stage_s: Mapping[str, Any],
    output: Path,
) -> tuple[torch.device, str, torch.nn.Module, torch.optim.Optimizer, PlainL2Loss, StageSBatchPath, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage U requires CUDA and refuses silent CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"Expected RTX 5070, found {device_name!r}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(stage_s["runtime"]["seed"])
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    dataset = ROOT / stage_s["data"]["dataset"]
    coords = load_coords(dataset)
    padding = json.loads((ROOT / "artifacts/stage_u/audit_summary.json").read_text())["synthetic_selected_padding"]
    initial = torch.load(ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True)
    if tensor_state_sha256(initial) != stage_s["frozen_pairing"]["initial_tensor_state_sha256"]:
        raise ValueError("Frozen common initial state changed")
    model, initialization = build_stage_u_model(
        stage_s, variant, coords=coords, initial_state=initial, padding=padding
    )
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    loss = PlainL2Loss().to(device)
    data = StageSBatchPath(dataset, ROOT / stage_s["preprocessing"]["artifact"], device)
    output.mkdir(parents=True, exist_ok=True)
    return device, device_name, model, optimizer, loss, data, initialization


def train_group(
    *, group: list[int], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loss_fn: PlainL2Loss, data: StageSBatchPath, parameters: list[torch.nn.Parameter],
    clip: float, update: int, total_updates: int, warmup_updates: int,
    base_lr: float, min_lr: float,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for source in group:
        batch=data.predict(model,int(source))
        loss=loss_fn(batch["predicted_residual"],batch["residual_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage U loss became nonfinite")
        (loss/len(group)).backward(); losses.append(float(loss.detach().cpu()))
    before=gradient_norm(parameters)
    if not np.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage U gradient became nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(parameters,clip)
    after=gradient_norm(parameters)
    lr=warmup_cosine_learning_rate(
        update,total_updates=total_updates,warmup_updates=warmup_updates,
        base_learning_rate=base_lr,min_learning_rate=min_lr,
    )
    for group_spec in optimizer.param_groups: group_spec["lr"]=lr
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise FloatingPointError("Stage U optimizer produced nonfinite parameters")
    return {"loss":float(np.mean(losses)),"before":before,"after":after,"clipped":before>clip,"lr":lr}


def run(variant: str, config: Path, root: Path, resume: bool) -> None:
    stage_u,stage_s,freeze,train_pairs=load_stage_u(config)
    output=root/"variants"/variant
    device,device_name,model,optimizer,loss_fn,data,initialization=prepare(variant,stage_s,output)
    pilot=stage_u["pilots"]
    epochs=int(pilot["epochs"]); accumulation=int(pilot["gradient_accumulation"])
    orders=natural_epoch_orders(train_pairs,epochs=epochs,seed=int(pilot["seed"]))
    updates_per_epoch=len(accumulation_groups(orders[0],accumulation=accumulation))
    scheduler_total_updates=updates_per_epoch*int(pilot["scheduler_total_epochs"])
    warmup_updates=updates_per_epoch*int(pilot["scheduler_warmup_epochs"])
    pilot_updates=updates_per_epoch*epochs
    flat=[value for order in orders for value in order]
    contract={
        **freeze,"schema_version":"stage-u-pilot-v1","variant":variant,
        "operator":{
            "spectral_only":"spectral",
            "coordinate_fd":"spectral_plus_coordinate_fd",
            "spherical_proxy_fd":"spectral_plus_spherical_proxy_fd",
        }[variant],
        "initialization":initialization,
        "parameter_difference":variant_parameter_difference(trainable_parameter_count(model)),
        "epochs":epochs,"train_pairs":len(train_pairs),"validation_pairs":42,
        "microbatches_per_epoch":len(train_pairs),"updates_per_epoch":updates_per_epoch,
        "pilot_optimizer_updates":pilot_updates,
        "scheduler_total_updates":scheduler_total_updates,"warmup_updates":warmup_updates,
        "scheduler_semantics":"same Stage T 1200-epoch schedule truncated after epoch 150",
        "pair_order_sha256":order_sha256(flat),"epoch_order_sha256":[order_sha256(x) for x in orders],
        "seed":int(pilot["seed"]),"device":device_name,"torch":str(torch.__version__),
        "torch_cuda":torch.version.cuda,"python":sys.version,"platform":platform.platform(),
        "loss":"PlainL2Loss_on_normalized_residual","mixed_precision":False,
        "project_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "validation_not_used_for_training_or_stopping":True,
        "CARTESIAN_REMAP_NOT_AUTHORIZED":True,
    }
    contract_hash=sha256_json(contract); atomic_json(output/"resolved_config.json",contract)
    state={"completed_epoch":0,"optimizer_updates":0,"microbatches":0,"runtime_seconds":0.0}
    rows=load_rows(output/"train_log.csv") if resume else []
    resume_path=output/"resume.pt"
    if resume:
        payload=torch.load(resume_path,map_location="cpu",weights_only=True)
        if payload["metadata"]["contract_sha256"]!=contract_hash: raise ValueError("Stage U resume contract changed")
        model.load_state_dict(payload["model_state_dict"],strict=True);optimizer.load_state_dict(payload["optimizer_state_dict"]);state=dict(payload["training_state"])
        completed = int(state["completed_epoch"])
        if len(rows) < completed:
            raise ValueError("Stage U resume log is shorter than its checkpoint")
        # A disconnected process may have atomically written later epoch rows
        # after the most recent 10-epoch resume checkpoint.  Those rows do not
        # have matching optimizer state and must be replayed from the checkpoint.
        rows = rows[:completed]
    params=[p for p in model.parameters() if p.requires_grad]; clip=float(stage_s["optimizer"]["gradient_clip_norm"])
    model.train()
    for epoch_index in range(int(state["completed_epoch"]),epochs):
        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device);started=time.perf_counter()
        records=[]
        groups=accumulation_groups(orders[epoch_index],accumulation=accumulation)
        for group in groups:
            update=int(state["optimizer_updates"])+1
            record=train_group(group=group,model=model,optimizer=optimizer,loss_fn=loss_fn,data=data,parameters=params,clip=clip,update=update,total_updates=scheduler_total_updates,warmup_updates=warmup_updates,base_lr=float(stage_s["optimizer"]["learning_rate"]),min_lr=float(stage_s["scheduler"]["min_learning_rate"]))
            state["optimizer_updates"]=update;state["microbatches"]+=len(group);records.append(record)
        torch.cuda.synchronize(device);elapsed=time.perf_counter()-started;epoch=epoch_index+1
        state["completed_epoch"]=epoch;state["runtime_seconds"]+=elapsed
        row={"epoch":epoch,"optimizer_updates":state["optimizer_updates"],"microbatches_seen":state["microbatches"],"microbatches_this_epoch":len(orders[epoch_index]),"optimizer_updates_this_epoch":len(groups),"train_loss_mean":float(np.mean([x["loss"] for x in records])),"train_loss_min":float(np.min([x["loss"] for x in records])),"train_loss_max":float(np.max([x["loss"] for x in records])),"learning_rate_start":records[0]["lr"],"learning_rate_end":records[-1]["lr"],"gradient_norm_before_clip_mean":float(np.mean([x["before"] for x in records])),"gradient_norm_before_clip_max":float(np.max([x["before"] for x in records])),"gradient_norm_after_clip_mean":float(np.mean([x["after"] for x in records])),"gradient_norm_after_clip_max":float(np.max([x["after"] for x in records])),"clipping_fraction":float(np.mean([x["clipped"] for x in records])),"nonfinite_count":0,"epoch_runtime_seconds":elapsed,"cumulative_runtime_seconds":state["runtime_seconds"],"gpu_peak_allocated_mib":torch.cuda.max_memory_allocated(device)/2**20,"gpu_peak_reserved_mib":torch.cuda.max_memory_reserved(device)/2**20}
        rows.append(row);atomic_csv(output/"train_log.csv",rows)
        metadata={**contract,"contract_sha256":contract_hash,"total_updates":scheduler_total_updates,"warmup_updates":warmup_updates,"epoch_train_loss":row["train_loss_mean"],"epoch_learning_rate":row["learning_rate_end"]}
        payload=checkpoint_payload(model=model,optimizer=optimizer,training_state=state,metadata=metadata)
        if epoch in {int(x) for x in pilot["checkpoint_epochs"]}: save_checkpoint(output/"checkpoints"/f"epoch_{epoch:04d}.pt",payload)
        if epoch%10==0 or epoch in {int(x) for x in pilot["checkpoint_epochs"]}: save_checkpoint(resume_path,payload)
        print(json.dumps({"variant":variant,"epoch":epoch,"updates":state["optimizer_updates"],"loss":row["train_loss_mean"],"lr":row["learning_rate_end"],"clip":row["clipping_fraction"],"seconds":elapsed}),flush=True)
    if state["optimizer_updates"]!=pilot_updates: raise RuntimeError("Stage U pilot update count changed")
    final_hash=tensor_state_sha256(model.state_dict())
    atomic_json(output/"training_summary.json",{"status":"passed","all_finite":True,**state,"final_model_state_sha256":final_hash,"parameter_count":trainable_parameter_count(model),"peak_allocated_mib_max":max(float(r["gpu_peak_allocated_mib"]) for r in rows),"peak_reserved_mib_max":max(float(r["gpu_peak_reserved_mib"]) for r in rows)})
    data.close()


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--variant",choices=VARIANTS,required=True);parser.add_argument("--config",type=Path,default=Path("configs/stage_u/geometry_ablation.yaml"));parser.add_argument("--root",type=Path,default=Path("artifacts/stage_u"));parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    run(args.variant,args.config,args.root,args.resume)


if __name__=="__main__":main()
