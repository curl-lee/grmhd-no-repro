"""Auditable mixed Fourier/cosine operators for Stage W.

The spatial tensor contract is ``(phi, theta, r)``.  The implementation uses
an orthonormal real FFT along periodic phi and orthonormal DCT-II transforms
along non-periodic theta and stored log-r index.  It is a computational basis,
not a spherical-harmonic or Kerr--Schild covariant operator.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


def _real_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"Stage W DCT requires float32 or float64, found {x.dtype}")
    return x.dtype


def orthonormal_dct_ii(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Apply an orthonormal DCT-II using an explicitly tested even FFT extension."""

    _real_dtype(x)
    moved = x.movedim(dim, -1)
    n = moved.shape[-1]
    if n < 2:
        raise ValueError("DCT axis must contain at least two samples")
    extension = torch.cat((moved, moved.flip(-1)), dim=-1)
    spectrum = torch.fft.fft(extension, dim=-1)
    modes = torch.arange(n, dtype=moved.dtype, device=moved.device)
    phase = torch.exp(-1j * math.pi * modes / (2 * n))
    cosine_sum = 0.5 * (spectrum[..., :n] * phase).real
    scale = torch.full(
        (n,), math.sqrt(2.0 / n), dtype=moved.dtype, device=moved.device
    )
    scale[0] = math.sqrt(1.0 / n)
    return (cosine_sum * scale).movedim(-1, dim)


def orthonormal_idct_ii(coefficients: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Invert :func:`orthonormal_dct_ii` with the conjugate even spectrum."""

    _real_dtype(coefficients)
    moved = coefficients.movedim(dim, -1)
    n = moved.shape[-1]
    if n < 2:
        raise ValueError("IDCT axis must contain at least two samples")
    modes = torch.arange(n, dtype=moved.dtype, device=moved.device)
    scale = torch.full(
        (n,), math.sqrt(2.0 / n), dtype=moved.dtype, device=moved.device
    )
    scale[0] = math.sqrt(1.0 / n)
    cosine_sum = moved / scale
    positive = 2.0 * cosine_sum * torch.exp(1j * math.pi * modes / (2 * n))
    complex_dtype = torch.complex128 if moved.dtype == torch.float64 else torch.complex64
    spectrum = torch.zeros(
        (*moved.shape[:-1], 2 * n), dtype=complex_dtype, device=moved.device
    )
    spectrum[..., :n] = positive
    spectrum[..., n] = 0
    spectrum[..., n + 1 :] = positive[..., 1:].flip(-1).conj()
    extension = torch.fft.ifft(spectrum, dim=-1).real
    return extension[..., :n].movedim(-1, dim)


def mixed_basis_transform(x: torch.Tensor) -> torch.Tensor:
    """Return ``rFFT_phi x DCT-II_theta x DCT-II_r`` coefficients."""

    if x.ndim < 3:
        raise ValueError("Mixed transform requires phi, theta, and r axes")
    transformed = orthonormal_dct_ii(x, dim=-1)
    transformed = orthonormal_dct_ii(transformed, dim=-2)
    return torch.fft.rfft(transformed, dim=-3, norm="ortho")


def mixed_basis_inverse(coefficients: torch.Tensor, *, n_phi: int) -> torch.Tensor:
    """Invert a full mixed-basis coefficient tensor."""

    transformed = torch.fft.irfft(coefficients, n=int(n_phi), dim=-3, norm="ortho")
    transformed = orthonormal_idct_ii(transformed, dim=-2)
    return orthonormal_idct_ii(transformed, dim=-1)


def mixed_basis_energy(coefficients: torch.Tensor, *, n_phi: int) -> torch.Tensor:
    """Energy of an orthonormal phi-rFFT representation with redundancy weights."""

    weights = torch.full(
        (coefficients.shape[-3],), 2.0,
        dtype=coefficients.real.dtype, device=coefficients.device,
    )
    weights[0] = 1.0
    if int(n_phi) % 2 == 0 and coefficients.shape[-3] == int(n_phi) // 2 + 1:
        weights[-1] = 1.0
    shape = [1] * coefficients.ndim
    shape[-3] = weights.numel()
    return (coefficients.abs().square() * weights.reshape(shape)).sum()


def dct_lowpass_1d(signal: torch.Tensor, keep: int) -> torch.Tensor:
    coefficients = orthonormal_dct_ii(signal, dim=-1)
    truncated = torch.zeros_like(coefficients)
    truncated[..., : int(keep)] = coefficients[..., : int(keep)]
    return orthonormal_idct_ii(truncated, dim=-1)


class MixedBasisSpectralConv3d(nn.Module):
    """Stage-T-isomorphic spectral convolution with a mixed non-Euclidean basis.

    ``n_modes=(8,8,8)`` stores ``(5,8,8)`` complex coefficients because only
    phi uses a real FFT.  This exactly matches the Stage-T dense complex weight
    count ``(8,8,5)`` while retaining eight requested modes per logical axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Sequence[int],
        *,
        resolution_scaling_factor=None,
        max_n_modes=None,
        rank: float = 1.0,
        fixed_rank_modes=False,
        implementation: str = "factorized",
        separable: bool = False,
        factorization=None,
        decomposition_kwargs=None,
        bias: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        if len(n_modes) != 3:
            raise ValueError("MixedBasisSpectralConv3d requires exactly three spatial axes")
        if resolution_scaling_factor is not None:
            raise NotImplementedError("Stage W freezes resolution scaling to None")
        if separable:
            raise NotImplementedError("Stage W freezes dense channel mixing")
        if factorization not in (None, "Dense") or rank != 1.0 or fixed_rank_modes:
            raise NotImplementedError("Stage W freezes an unfactorized dense spectral weight")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.requested_n_modes = tuple(int(value) for value in n_modes)
        self.n_modes = self.requested_n_modes
        if max_n_modes is not None and tuple(int(v) for v in max_n_modes) != self.requested_n_modes:
            raise NotImplementedError("Stage W does not use incremental mode growth")
        self.max_n_modes = self.requested_n_modes
        self.retained_shape = (
            self.requested_n_modes[0] // 2 + 1,
            self.requested_n_modes[1],
            self.requested_n_modes[2],
        )
        self.rank = float(rank)
        self.factorization = factorization
        self.implementation = str(implementation)
        self.separable = False
        self.fft_norm = "ortho"
        init_std = math.sqrt(2.0 / (self.in_channels + self.out_channels))
        real = torch.randn(
            self.in_channels, self.out_channels, *self.retained_shape
        )
        imaginary = torch.randn_like(real)
        self.weight = nn.Parameter(init_std * torch.complex(real, imaginary))
        self.bias = (
            nn.Parameter(
                init_std * torch.randn(self.out_channels, 1, 1, 1)
            )
            if bias else None
        )

    def transform(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        if output_shape is not None and tuple(output_shape) != tuple(x.shape[-3:]):
            raise NotImplementedError("Stage W freezes input and output resolution")
        return x

    def forward(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        if output_shape is not None and tuple(output_shape) != tuple(x.shape[-3:]):
            raise NotImplementedError("Stage W freezes input and output resolution")
        if not x.is_floating_point() or x.is_complex():
            raise TypeError("MixedBasisSpectralConv3d requires real floating activations")
        coefficients = mixed_basis_transform(x)
        phi_keep = min(coefficients.shape[-3], self.retained_shape[0])
        theta_keep = min(coefficients.shape[-2], self.retained_shape[1])
        radial_keep = min(coefficients.shape[-1], self.retained_shape[2])
        retained = torch.einsum(
            "bipqr,iopqr->bopqr",
            coefficients[:, :, :phi_keep, :theta_keep, :radial_keep],
            self.weight[:, :, :phi_keep, :theta_keep, :radial_keep],
        )
        # irfft requires real-valued zero-frequency coefficients.  The pinned
        # upstream layer applies the analogous constraint on its real-FFT axis.
        zero_mode = torch.complex(retained[:, :, :1].real, torch.zeros_like(retained[:, :, :1].real))
        retained = torch.cat((zero_mode, retained[:, :, 1:]), dim=2)
        output_coefficients = torch.zeros(
            (x.shape[0], self.out_channels, *coefficients.shape[-3:]),
            dtype=coefficients.dtype,
            device=x.device,
        )
        output_coefficients[:, :, :phi_keep, :theta_keep, :radial_keep] = retained
        output = mixed_basis_inverse(output_coefficients, n_phi=x.shape[-3])
        if self.bias is not None:
            output = output + self.bias
        return output


class AxisBoundaryFiniteDifferenceConvolution3d(nn.Module):
    """Exact Stage-T 3x3x3 stencil with axis-specific boundary padding only."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        groups: int = 1,
        phi_padding: str = "circular",
        theta_padding: str = "replicate",
        radial_padding: str = "replicate",
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Finite-difference kernel size must be odd")
        allowed = {"circular", "replicate", "reflect", "constant"}
        if {phi_padding, theta_padding, radial_padding} - allowed:
            raise ValueError("Unsupported Stage W padding mode")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_dim = 3
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)
        self.pad_size = self.kernel_size // 2
        self.phi_padding = phi_padding
        self.theta_padding = theta_padding
        self.radial_padding = radial_padding
        self.conv = nn.Conv3d(
            self.in_channels, self.out_channels, self.kernel_size,
            padding=0, bias=False, groups=self.groups,
        )
        self.weight = self.conv.weight

    @staticmethod
    def _pad_axis(x: torch.Tensor, axis: int, amount: int, mode: str) -> torch.Tensor:
        padding = [0, 0, 0, 0, 0, 0]
        start = {-1: 0, -2: 2, -3: 4}[axis]
        padding[start] = padding[start + 1] = int(amount)
        if mode == "constant":
            return F.pad(x, padding, mode=mode, value=0.0)
        return F.pad(x, padding, mode=mode)

    def forward(self, x: torch.Tensor, grid_width: float | torch.Tensor) -> torch.Tensor:
        padded = self._pad_axis(x, -3, self.pad_size, self.phi_padding)
        padded = self._pad_axis(padded, -2, self.pad_size, self.theta_padding)
        padded = self._pad_axis(padded, -1, self.pad_size, self.radial_padding)
        convolved = F.conv3d(padded, self.weight, groups=self.groups)
        coefficient_sum = self.weight.sum(dim=(2, 3, 4), keepdim=True)
        centered = F.conv3d(x, coefficient_sum, groups=self.groups)
        return (convolved - centered) / grid_width
