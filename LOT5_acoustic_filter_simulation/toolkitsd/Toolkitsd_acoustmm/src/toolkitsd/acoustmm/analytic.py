"""Analytical reference formulas used for validation and simple closed-form comparisons."""

from __future__ import annotations

import numpy as np


def _trapz_compat(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    raise AttributeError("NumPy has neither trapezoid nor trapz")


def tl_simple_expansion_analytic(
    frequencies: np.ndarray,
    chamber_length: float,
    area_ratio: float,
    *,
    c0: float = 340.0,
) -> np.ndarray:
    """Analytical TL of a simple expansion chamber from Munjal."""
    freqs = np.asarray(frequencies, dtype=np.float64).ravel()
    if np.any(freqs <= 0.0):
        raise ValueError("frequencies must be strictly positive")
    if chamber_length <= 0.0:
        raise ValueError("chamber_length must be positive")
    if area_ratio <= 0.0:
        raise ValueError("area_ratio must be positive")
    if c0 <= 0.0:
        raise ValueError("c0 must be positive")

    k0 = 2.0 * np.pi * freqs / c0
    m = float(area_ratio)
    return 10.0 * np.log10(1.0 + 0.25 * (m - 1.0 / m) ** 2 * np.sin(k0 * chamber_length) ** 2)


def calculate_zp_parois_simple(
    omega: np.ndarray,
    k0: np.ndarray,
    theta: float,
    E: float,
    h: float,
    nu: float,
    mu: float,
) -> np.ndarray:
    """Plate surface impedance for the simple wall model used in legacy `pytmm` examples."""
    w = np.asarray(omega, dtype=np.float64).ravel()
    k = np.asarray(k0, dtype=np.float64).ravel()
    if w.shape != k.shape:
        raise ValueError("omega and k0 must have the same shape")
    if np.any(w <= 0.0):
        raise ValueError("omega must be strictly positive")
    if E <= 0.0 or h <= 0.0 or mu <= 0.0:
        raise ValueError("E, h and mu must be positive")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must be in (-1, 0.5)")

    D = (E * h**3) / (12.0 * (1.0 - nu**2))
    kp = np.power(w, 0.25) * np.sqrt(mu / D)
    return (D / (1j * w)) * ((k**4) * np.sin(theta) ** 4 - (kp**2) * np.power(w, 1.5))


def tl_paroi_analytic(
    omega: np.ndarray,
    theta: float,
    rho0: float,
    c0: float,
    mu: float,
    E: float,
    h: float,
    nu: float,
) -> np.ndarray:
    """Analytical transmission loss for a single simple plate partition."""
    w = np.asarray(omega, dtype=np.float64).ravel()
    if np.any(w <= 0.0):
        raise ValueError("omega must be strictly positive")
    if rho0 <= 0.0 or c0 <= 0.0:
        raise ValueError("rho0 and c0 must be positive")
    if E <= 0.0 or h <= 0.0 or mu <= 0.0:
        raise ValueError("E, h and mu must be positive")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu must be in (-1, 0.5)")

    D = E * h**3 / (12.0 * (1.0 - nu**2))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    denominator = np.abs(
        -w**2 * mu
        + D * (w / c0) ** 4 * sin_theta**4
        + (2.0 * rho0 * w * c0) / (1j * cos_theta)
    ) ** 2
    tau = (w**2 * (rho0 * c0) ** 2) * 4.0 / (cos_theta**2 * denominator)
    return 10.0 * np.log10(1.0 / tau)


__all__ = [
    "tl_simple_expansion_analytic",
    "calculate_zp_parois_simple",
    "tl_paroi_analytic",
    "integrate_3d_diffuse",
]


def integrate_3d_diffuse(
    tl_function,
    frequencies: np.ndarray,
    theta_lim: float = np.pi / 2.0 - 1e-9,
    n_eval: int = 50,
) -> np.ndarray:
    """Legacy-compatible diffuse-field integration of TL over incidence angle."""
    freqs = np.asarray(frequencies, dtype=np.float64).ravel()
    if np.any(freqs <= 0.0):
        raise ValueError("frequencies must be strictly positive")
    if not (0.0 < theta_lim < np.pi / 2.0):
        raise ValueError("theta_lim must be in (0, pi/2)")
    if n_eval < 2:
        raise ValueError("n_eval must be >= 2")

    theta_values = np.linspace(1e-9, theta_lim, n_eval)
    tl_integrated = np.zeros_like(freqs, dtype=np.float64)

    for i, freq in enumerate(freqs):
        tl_values = np.asarray(tl_function(freq, theta_values), dtype=np.float64)
        if tl_values.shape != theta_values.shape:
            raise ValueError("tl_function(freq, theta_values) must return shape (n_eval,)")
        integrand = tl_values * np.cos(theta_values) * np.sin(theta_values)
        tl_integrated[i] = 2.0 * _trapz_compat(integrand, theta_values) / (np.sin(theta_lim) ** 2)

    return tl_integrated
