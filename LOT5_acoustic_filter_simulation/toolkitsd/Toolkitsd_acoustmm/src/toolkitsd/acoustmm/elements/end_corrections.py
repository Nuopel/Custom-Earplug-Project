"""End-correction helpers for necks, ports, and resonator junctions."""

from __future__ import annotations

import numpy as np

def neck_to_cavity_end_correction(radius_neck: float, radius_cavity: float) -> float:
    """Return the end correction for a neck opening into a larger cavity."""
    if radius_neck <= 0.0 or radius_cavity <= 0.0:
        raise ValueError("radius_neck and radius_cavity must be positive")

    ratio = radius_neck / radius_cavity
    return 0.82 * (1.0 - 1.35 * ratio + 0.31 * ratio**3) * radius_neck



def neck_to_waveguide_end_correction(radius_neck: float, radius_waveguide: float) -> float:
    """Return the end correction for a neck opening into a main circular waveguide."""
    if radius_neck <= 0.0 or radius_waveguide <= 0.0:
        raise ValueError("radius_neck and radius_waveguide must be positive")

    ratio = radius_neck / radius_waveguide
    return 0.82 * (
        1.0
        - 0.235 * ratio
        - 1.32 * ratio**2
        + 1.54 * ratio**3
        - 0.86 * ratio**4
    ) * radius_neck



def total_neck_end_correction(
    radius_neck: float,
    radius_cavity: float,
    radius_waveguide: float | None,
    *,
    outside_flanged: bool = False,
) -> float:
    """Return the total end correction for a neck coupled to a cavity and either a waveguide or free space."""
    cavity_side = neck_to_cavity_end_correction(radius_neck, radius_cavity)
    if radius_waveguide is None:
        outside_side = neck_to_outside_end_correction(radius_neck, flanged=outside_flanged)
    else:
        outside_side = neck_to_waveguide_end_correction(radius_neck, radius_waveguide)
    return cavity_side + outside_side

def neck_to_outside_end_correction(
    radius_neck: float,
    flanged: bool = False,
) -> float:
    """
    Return the radiation end correction for a neck opening to free space.

    Parameters
    ----------
    radius_neck : float
        Radius of the neck [m].
    flanged : bool
        If True, assume an infinite-flange (half-space) termination → 8/(3π)·a.
        If False (default), assume an unflanged termination → 0.6133·a
        (Levine & Schwinger 1948, low-frequency limit ka ≪ 1).
    """
    if radius_neck <= 0.0:
        raise ValueError("radius_neck must be positive")

    if flanged:
        return (8.0 / (3.0 * np.pi)) * radius_neck  # ≈ 0.8216 · a
    else:
        return 0.6133 * radius_neck  # Levine–Schwinger
