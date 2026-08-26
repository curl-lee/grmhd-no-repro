# Mixed transform implementation notes

The implementation uses an explicit even extension followed by `torch.fft.fft`
for orthonormal DCT-II and reconstructs the conjugate even spectrum for IDCT.
This avoids an external DCT dependency and is checked against roundtrip,
Parseval, localization, shift, and boundary-response contracts.

Phi alone uses orthonormal `rfft/irfft`.  Requested modes `(8,8,8)` therefore
store `(5,8,8)` complex modal weights.  This is a reversible axis permutation
of the Stage-T initial `(8,8,5)` spectral tensor and keeps parameter count
exactly unchanged.  Theta and log-r use the lowest eight cosine modes.

The DCT axes represent even non-periodic continuation at their faces.  They do
not identify lower and upper boundaries and are not spherical harmonics.  No
coordinate values or metric factors enter the learned convolution.
