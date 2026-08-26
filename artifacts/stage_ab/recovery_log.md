# Stage AB Recovery Log

## Action 1 — pre-recovery read-only audit

- action: Captured project, WSL identity, interop, GPU bridge, host visibility, and PyTorch evidence.
- reason: Locate the first broken resource layer before making any change.
- observed: interop=PASS, dxg=PRESENT, WSL nvidia-smi=PASS, PyTorch CUDA=True.
- environment mutation: none.

## Resolution

- action: Repeated the probes outside the restricted tool device sandbox.
- reason: The sandbox presents an isolated `/dev`, so its missing GPU device was not valid evidence about the underlying WSL instance.
- before: sandbox-scoped false-negative GPU and interop probes.
- after: real WSL interop, `/dev/dxg`, WSL GPU visibility, and PyTorch CUDA all pass.
- host restart, package reinstall, driver change, and project scientific changes: none.
