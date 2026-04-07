"""Geometry utilities for ear-canal discretization and builders."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from .elements import ViscothermalDuct


@dataclass
class EarCanalBuilder:
    """Build an equivalent ear-canal element by viscothermal segmentation."""

    n_segments: int = 40
    radius_scale: float = 1.0
    c0: float = 340.0
    rho0: float = 1.2

    def __post_init__(self) -> None:
        if self.n_segments < 2:
            raise ValueError("n_segments must be >= 2")
        if self.radius_scale <= 0.0:
            raise ValueError("radius_scale must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")

    def _placeholder_profile(self, length: float = 24e-3) -> tuple[np.ndarray, np.ndarray]:
        """Plausible temporary profile: smooth narrowing toward the eardrum."""
        x = np.linspace(0.0, length, self.n_segments + 1)
        t = x / length
        # Approximate adult-canal-like trend: ~4.0 mm entrance -> ~3.0 mm deep end.
        r_mm = 4.0 - 1.0 * (t**0.8)
        r = 1e-3 * r_mm * self.radius_scale
        warnings.warn(
            "EarCanalBuilder is using a temporary placeholder ear-canal profile. "
            "Replace with measured or Stinson-derived geometry for validation work.",
            RuntimeWarning,
            stacklevel=2,
        )
        return x, r

    def build(
        self,
        x: np.ndarray | None = None,
        radius: np.ndarray | None = None,
        *,
        return_segments: bool = False,
    ):
        """Return sum of N ViscothermalDuct segments for the provided profile.

        Parameters:
            x: Axial coordinates [m], size N+1.
            radius: Radius profile [m], same size as x.
            return_segments: If True, also return the list of segments.
        """
        if (x is None) ^ (radius is None):
            raise ValueError("x and radius must be both provided or both omitted")

        if x is None and radius is None:
            x, radius = self._placeholder_profile()
        else:
            x = np.asarray(x, dtype=np.float64).ravel()
            radius = np.asarray(radius, dtype=np.float64).ravel()

        if x.size != radius.size:
            raise ValueError("x and radius must have the same size")
        if x.size < 3:
            raise ValueError("profile must contain at least 3 points")
        if np.any(np.diff(x) <= 0.0):
            raise ValueError("x must be strictly increasing")
        if np.any(radius <= 0.0):
            raise ValueError("radius must be strictly positive")

        # Resample profile to requested segment count for controlled discretization.
        x_uniform = np.linspace(x[0], x[-1], self.n_segments + 1)
        r_uniform = np.interp(x_uniform, x, radius) * self.radius_scale
        r_mid = 0.5 * (r_uniform[:-1] + r_uniform[1:])
        lengths = np.diff(x_uniform)

        segments = [
            ViscothermalDuct(radius=float(r), length=float(L), c0=self.c0, rho0=self.rho0)
            for r, L in zip(r_mid, lengths)
        ]
        canal = sum(segments)
        if return_segments:
            return canal, segments
        return canal
