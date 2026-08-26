# Boundary and Normalization Audit

- `PHI_BOUNDARY = PERIODIC`; source phi index uses modulo wrapping.
- `THETA_BOUNDARY = TRUNCATED_RENORMALIZED`; no circular, replicate, reflect, or speculative pole-parity continuation is used.
- `R_BOUNDARY = TRUNCATED_RENORMALIZED`; no radial wrapping is used.
- `POLE_TOPOLOGY_EXACT = false` because coordinate-basis vector parity provenance is incomplete.
- candidate offsets remain exactly `[-3,3]^3`, shape `[7,7,7]`.
- local radius remains exactly `R_i=3 median(valid +/-1 embedding distances)`.

Production normalization fails: `ZERO_Z_COUNT = 4` at
`[basis, theta_index, r_index] = [[4, 0, 0], [4, 0, 63], [4, 63, 0], [4, 63, 63]]`. These are basis 4 at the
four combined theta/r corners. No radius, K, boundary, or epsilon fallback was
changed. Training is prohibited by the frozen Stage AE gate.
