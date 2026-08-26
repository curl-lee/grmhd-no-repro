# Candidate coordinate embeddings (design only; no training)

These are controlled design candidates, not selected by validation.

## C1 — continuous spherical features

Append five channels: normalized `log(r)`, `sin(theta)`, `cos(theta)`, `sin(phi)`, `cos(phi)`. The periodic pair avoids a raw-phi seam. These are coordinate-position features only and do not transform B or velocity components.

## C2 — paper-analogue positional embedding

An exact Kerr–Schild Cartesian position map is not authorized because the current simulation's spin/mapping convention is absent. A clearly labelled `ADAPTED_POSITIONAL_ANALOGUE` could use normalized Euclidean spherical-position proxies `(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`; it must not be called exact Cartesian KS.

## C3 — shell plus continuous hybrid

Retain the existing eight physical-r shell channels and append the five C1 channels. This keeps the frozen coarse central-focus signal while resolving position within a shell.

No candidate is trained or ranked in Stage X. Because field basis and operator provenance are stronger blockers, these designs are not the authorized next stage.
