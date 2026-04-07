"""Infinite-extent barrier/layer models."""

from __future__ import annotations

import numpy as np

from ..analytic import calculate_zp_parois_simple
from .base import InfiniteLayerModel


def _trapz_compat(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x, axis=axis)
    if hasattr(np, "trapz"):
        return np.trapz(y, x, axis=axis)
    raise AttributeError("NumPy has neither trapezoid nor trapz")


def _plate_specific_impedance(
    omega: np.ndarray,
    *,
    rho_plate: float,
    h: float,
    E: float,
    nu: float,
    theta: float,
    c0: float,
) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64).ravel()
    if np.any(omega <= 0.0):
        raise ValueError("omega must be strictly positive")
    k0 = omega / c0
    return calculate_zp_parois_simple(
        omega=omega,
        k0=k0,
        theta=theta,
        E=E,
        h=h,
        nu=nu,
        mu=rho_plate * h,
    )


class InfinitePlate(InfiniteLayerModel):
    """Infinite-extent plate model for transmission/reflection calculations."""

    def __init__(
        self,
        *,
        rho_plate: float,
        h: float,
        E: float,
        nu: float,
        theta: float = 0.0,
        c0: float = 340.0,
    ) -> None:
        if rho_plate <= 0.0:
            raise ValueError("rho_plate must be positive")
        if h <= 0.0:
            raise ValueError("h must be positive")
        if E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")
        if c0 <= 0.0:
            raise ValueError("c0 must be positive")
        self.rho_plate = float(rho_plate)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.theta = float(theta)
        self.c0 = float(c0)

    def specific_impedance(self, omega: np.ndarray, *, theta: float | None = None) -> np.ndarray:
        return _plate_specific_impedance(
            omega,
            rho_plate=self.rho_plate,
            h=self.h,
            E=self.E,
            nu=self.nu,
            theta=self.theta if theta is None else float(theta),
            c0=self.c0,
        )

    def transmission_coefficient(
        self,
        omega: np.ndarray,
        *,
        theta: float = 0.0,
        Z_c: float | complex | np.ndarray,
    ) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        Z_c_arr = np.asarray(Z_c, dtype=np.complex128)
        if Z_c_arr.ndim == 0:
            Z_c_arr = np.full(omega.shape, Z_c_arr, dtype=np.complex128)
        Z_c_arr = np.broadcast_to(Z_c_arr, omega.shape)

        ctheta = np.cos(theta)
        if np.isclose(ctheta, 0.0):
            ctheta = np.copysign(1e-12, ctheta if ctheta != 0.0 else 1.0)

        z_w = self.specific_impedance(omega, theta=theta)
        return 2.0 / (2.0 + (z_w * ctheta) / Z_c_arr)

    def reflection_coefficient(
        self,
        omega: np.ndarray,
        *,
        theta: float = 0.0,
        Z_c: float | complex | np.ndarray,
    ) -> np.ndarray:
        tau = self.transmission_coefficient(omega, theta=theta, Z_c=Z_c)
        return 1.0 - tau

    def TL(self, Z_c: float | complex | np.ndarray, omega: np.ndarray) -> np.ndarray:
        tau = self.transmission_coefficient(omega, theta=self.theta, Z_c=Z_c)
        return 20.0 * np.log10(1.0 / np.abs(tau))

    def TL_diffuse(
        self,
        Z_c: float | complex | np.ndarray,
        omega: np.ndarray,
        *,
        theta_lim: float = np.pi / 2.0 - 1e-9,
        n_eval: int = 50,
    ) -> np.ndarray:
        if not (0.0 < theta_lim < np.pi / 2.0):
            raise ValueError("theta_lim must be in (0, pi/2)")
        if n_eval < 2:
            raise ValueError("n_eval must be >= 2")

        omega = np.asarray(omega, dtype=np.float64).ravel()
        theta_values = np.linspace(1e-9, theta_lim, n_eval)
        tl_theta = np.zeros((n_eval, omega.size), dtype=np.float64)

        for i, theta in enumerate(theta_values):
            tau = self.transmission_coefficient(omega, theta=float(theta), Z_c=Z_c)
            tl_theta[i, :] = 20.0 * np.log10(1.0 / np.abs(tau))

        weights = (np.cos(theta_values) * np.sin(theta_values))[:, None]
        return 2.0 * _trapz_compat(tl_theta * weights, theta_values, axis=0) / (np.sin(theta_lim) ** 2)


__all__ = ["InfinitePlate", "_plate_specific_impedance"]
