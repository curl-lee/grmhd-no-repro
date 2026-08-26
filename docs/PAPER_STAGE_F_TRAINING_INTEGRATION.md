# Stage F training integration

## Upstream boundary

The model remains the pinned `neuralop.models.FNO`.  Each training batch is
executed through `neuralop.training.Trainer.train_one_batch`; the local adapter
only propagates the epoch, preserves gradients within an explicitly configured
accumulation group, and exposes the processed prediction/context for structured
logging.  It does not copy or modify the upstream FNO, H1 loss, or Trainer
implementation.

`PaperDataProcessor` supplies `x=model_input` and `y=normalized_target` to the
upstream batch path.  After FNO forward it replaces the generic sample with the
explicit Stage F fields and `PaperLossContext`.  `PaperTrainerLossAdapter`
dispatches to exactly one loss:

- Full: `PaperCompositeLoss`, including the frozen Stage E terms;
- Plain: `PlainL2Loss`, with H1, ROI, bounds training penalty, envelope, and
  dissipation explicitly listed as disabled rather than logged as zero-valued
  components.

The no-optimizer real-batch test and upstream Trainer parity test cover both
modes.  They compare normalized prediction, scalar and component losses,
gradient norm, epoch/ROI ramp, and protocol metadata.  No test in that gate
calls `optimizer.step()`.

## Epoch and accumulation

At every epoch boundary the local adapter calls `set_epoch(epoch)` on the data
processor and loss adapter and `on_epoch_start(epoch)` on the upstream Trainer.
The context therefore evaluates the fixed ramp `min(1, epoch/375)`.  A small
optimizer proxy only gates the upstream batch-start `zero_grad`; optimizer step,
gradient clipping, and scheduler step remain explicit in the local run loop so
gradient accumulation is observable and testable.

## Checkpoint contract

`best.pt/` and `last.pt/` are neuraloperator training-state bundle directories.
They contain the strict model state plus upstream optimizer, scheduler, and
manifest files.  The existing local upstream wrapper adds a JSON sidecar; Stage
F validates that sidecar before loading model state.

The sidecar binds model/loss mode, project and upstream commits, source config,
HDF5, preprocessing, Stage D and Stage E checksums, split indices, selected
radial semantics, epoch/ramp, optimizer, scheduler, accumulation, seed, thermal
adaptation, coordinate adaptation, and a deterministic validation prediction
hash.  Tests cover strict model prediction parity, optimizer/scheduler restore,
epoch/ramp restore, stale checksum rejection, and bidirectional Full/Plain
cross-load rejection.

All checkpoint outputs from Stage F smoke are engineering artifacts and are
explicitly marked `scientific_checkpoint: false`.
