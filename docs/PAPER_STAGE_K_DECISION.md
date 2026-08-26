# Stage K Decision

## C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE

Stage K remains classified as `adapted_method_reproduction` using a 3D
differential LocalNO. DISCO integral is disabled. This is not 3D DISCO and not
an exact paper reproduction.

The construction, CUDA, smoke, training, checkpoint, validation, and transform
engineering gates passed:

- real 64-cube preflight passed on RTX 5070 without an optimizer;
- two-epoch smoke completed 158 microbatches and 40 updates;
- formal training completed 2,370 microbatches and 600 updates;
- best epoch 22 and last epoch 30 both strictly reload model, Adam, scheduler,
  epoch, initial identity, pair order, architecture, Plain loss, and provenance;
- no training NaN/Inf occurred and the trained state differs from initialization;
- 19-step GT and 100-step no-GT rollout are finite with positive rho/press;
- physical autoregression counters are exactly 100 prediction decodes and 100
  next-input encodes;
- no selected state exceeds the frozen `Rout` range gate.

Performance and stability do not satisfy A or B. One-step normalized average
is 0.624225 versus persistence 0.189088 (3.3012x worse), and LocalNO is worse
than persistence at every selected GT rollout step. More importantly, the
unchanged frozen artifact detector marks Bcc3 field collapse at steps
25/50/75/100 and vel3 collapse at 75/100. Persistent collapse is an explicit C
condition, so C takes precedence over the otherwise-passing finite,
positivity, and range gates.

Bcc3 and vel3 carry a known preprocessing-oracle floor, which limits the
physical interpretation of these flags. That limitation is documented but is
not used to alter or waive the predeclared Stage K rule after seeing results.

This decision does not imply that neural operators generally fail, that
differential LocalNO is equivalent to DISCO LocalNO, or that finite rollout is
physical validation. It only states that this frozen 30-epoch reduced100,
spherical Kerr--Schild, `press`-adapted differential LocalNO workflow trained
successfully but did not produce an artifact-stable rollout under the existing
detector.

No hyperparameter search, validation tuning, loss change, longer training,
LocalNO Full-adapted run, or paper ablation is authorized by this result.
