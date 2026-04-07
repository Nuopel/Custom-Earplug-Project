"""
Validation and shaping utilities shared across porous computations.

Angles are plane-referenced by convention: 0° = grazing, 90° = normal.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_frequency_array(frequencies: ArrayLike) -> NDArray[np.float64]:
    """Return validated 1D frequency array (Hz)."""
    freq = np.asarray(frequencies, dtype=float).ravel()
    if freq.size == 0:
        raise ValueError("frequencies must be non-empty")
    if np.any(freq <= 0):
        raise ValueError("frequencies must be strictly positive")
    return freq


def as_angle_array(incidence_angle_deg: ArrayLike) -> NDArray[np.float64]:
    """Return validated incidence angles in degrees (0–90), plane-referenced."""
    angles = np.asarray(incidence_angle_deg, dtype=float).ravel()
    if angles.size == 0:
        raise ValueError("incidence_angle_deg must be non-empty")
    if np.any((angles < 0) | (angles > 90)):
        raise ValueError(
            "You might be looking inside the Porous material? incidence_angle_deg must be within [0, 90] degrees")
    return angles


def column_vector(values: ArrayLike) -> np.ndarray:
    """Reshape 1D numeric array into column form for broadcasting."""
    arr = np.asarray(values)
    return arr.reshape(arr.size, 1)
