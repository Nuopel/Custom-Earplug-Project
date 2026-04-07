"""Minimal acoustic parameter container used across acoustmm examples.

This mirrors the small self-contained part of the historical
``toolkitsd.acoustic.pressure_gen.AcousticParameters`` API without importing
the rest of that legacy module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcousticParameters:
    """Frequency-dependent ambient acoustic parameters.

    The class intentionally stays small: it only provides the attributes that
    the current acoustmm package and showcase scripts use directly.
    """

    frequencies: np.ndarray | None = None
    c0: float = 343.0
    rho0: float = 1.225
    time_convention: str = "jwt"

    def __post_init__(self) -> None:
        if self.frequencies is None:
            frequencies = None
        else:
            frequencies = np.asarray(self.frequencies, dtype=np.float64).ravel()
            if frequencies.size == 0:
                raise ValueError("frequencies must not be empty")
            if np.any(frequencies <= 0.0):
                raise ValueError("frequencies must be strictly positive")
        if self.c0 <= 0.0:
            raise ValueError("c0 must be strictly positive")
        if self.rho0 <= 0.0:
            raise ValueError("rho0 must be strictly positive")
        if self.time_convention not in {"jwt", "neg_jwt"}:
            raise ValueError("time_convention must be 'jwt' or 'neg_jwt'")
        object.__setattr__(self, "frequencies", frequencies)

    @property
    def omega(self) -> np.ndarray:
        if self.frequencies is None:
            raise ValueError("frequencies must be defined to compute omega")
        return 2.0 * np.pi * self.frequencies

    @property
    def z0(self) -> float:
        return self.rho0 * self.c0

    @property
    def wavenumbers(self) -> np.ndarray:
        if self.frequencies is None:
            raise ValueError("frequencies must be defined to compute wavenumbers")
        return self.omega / self.c0

    @property
    def time_sign(self) -> int:
        return -1 if self.time_convention == "jwt" else 1
