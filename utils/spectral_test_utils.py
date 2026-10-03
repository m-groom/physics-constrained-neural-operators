import math

import torch


def create_periodic_grid(grid_shape, domain_lengths, device=None, dtype=torch.float64):
    """Build a periodic tensor-product grid with points x_i = i L / N."""
    coords = []

    for N, L in zip(grid_shape, domain_lengths, strict=False):
        coords.append(torch.arange(N, device=device, dtype=dtype) * (L / N))

    return torch.meshgrid(*coords, indexing="ij")


def build_wavenumbers(grid_shape, device=None, dtype=torch.float64):
    """Build broadcastable Fourier wavenumber tensors for a periodic grid."""
    d = len(grid_shape)
    wavenumbers = []

    for axis, N in enumerate(grid_shape):
        dx = 1.0 / N
        k = 2 * math.pi * torch.fft.fftfreq(N, d=dx)

        shape = [1] * d
        shape[axis] = N

        if device is not None:
            k = k.to(device)

        if dtype is not None:
            k = k.to(dtype)

        wavenumbers.append(k.reshape(shape))

    return wavenumbers


def evaluate_error(estimated, reference):
    """Compute max absolute error and RMSE between two tensors."""
    diff = estimated - reference
    max_error = diff.abs().max()
    rmse = torch.sqrt(torch.mean(diff**2))

    return max_error.item(), rmse.item()
