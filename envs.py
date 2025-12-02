"""
Environment definitions for the hemispheric specialization simulations.

Two environments are implemented:

1. SpatialEnvironment: multiscale spatial structure with a local anomaly
   embedded in a globally smoothed random field.

2. PlanarEnvironment: structure-free control with i.i.d. noise, mean-centred
   and variance-normalised, with no spatial gradients or anomalies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from config import SimulationConfig


def _gaussian_kernel_1d(sigma: float, radius: int | None = None) -> NDArray[np.float64]:
    """
    Construct a 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.
    radius : int or None
        Radius of the kernel (number of elements on each side of the centre).
        If None, radius is set to int(3 * sigma).

    Returns
    -------
    kernel : np.ndarray
        Normalised 1D kernel.
    """
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def _gaussian_blur_2d(grid: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """
    Apply separable 2D Gaussian blur to a 2D array using reflective padding.

    No external dependencies beyond NumPy are required.
    """
    kernel = _gaussian_kernel_1d(sigma)
    radius = (len(kernel) - 1) // 2

    # Convolve along rows
    padded = np.pad(grid, ((0, 0), (radius, radius)), mode="reflect")
    blurred_rows = np.empty_like(grid)
    for i in range(grid.shape[0]):
        row = padded[i]
        conv = np.convolve(row, kernel, mode="valid")
        blurred_rows[i] = conv

    # Convolve along columns
    padded = np.pad(blurred_rows, ((radius, radius), (0, 0)), mode="reflect")
    blurred = np.empty_like(grid)
    for j in range(grid.shape[1]):
        col = padded[:, j]
        conv = np.convolve(col, kernel, mode="valid")
        blurred[:, j] = conv

    return blurred


class SpatialEnvironment:
    """
    Multiscale spatial environment with a global structure and a local anomaly.

    Each trial consists of:
    - Base noise sampled from N(0, 1)
    - Gaussian smoothing with sigma = config.spatial_sigma
    - A 3 x 3 anomaly patch with intensity offset sampled from U(1.5, 2.0)
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_trial(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate a single spatial trial (20 x 20 grid)."""
        n = self.config.grid_size

        # Base noise
        grid = rng.normal(loc=0.0, scale=1.0, size=(n, n))

        # Global spatial structure
        grid = _gaussian_blur_2d(grid, sigma=self.config.spatial_sigma)

        # Localised anomaly: 3 x 3 patch with positive offset
        patch = self.config.anomaly_size
        # Choose centre such that patch fits fully inside the grid
        half = patch // 2
        i_center = rng.integers(low=half, high=n - half)
        j_center = rng.integers(low=half, high=n - half)
        offset = rng.uniform(1.5, 2.0)

        i_start = i_center - half
        i_end = i_center + half + 1
        j_start = j_center - half
        j_end = j_center + half + 1

        grid[i_start:i_end, j_start:j_end] += offset

        return grid


class PlanarEnvironment:
    """
    Structure-free planar control environment.

    Each trial consists of:
    - IID noise sampled from N(0, 1) on a 20 x 20 grid
    - Mean-centering and variance normalisation
    - No smoothing, no anomalies, and no spatial gradients
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate_trial(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Generate a single planar trial (20 x 20 grid, structure-free)."""
        n = self.config.grid_size

        grid = rng.normal(loc=0.0, scale=1.0, size=(n, n))

        # Mean-centre and variance-normalise
        mean = grid.mean()
        std = grid.std()
        if std <= 0.0:
            std = 1.0
        grid = (grid - mean) / std

        return grid
