"""Porous-layer TMM elements."""

from __future__ import annotations

import numpy as np

from toolkitsd.porous import JCAMaterial, JCAModel, MikiMaterial, MikiModel

from .base import AcousticElement


class JCALayer(AcousticElement):
    """JCA equivalent-fluid porous slab as a 2x2 TMM element."""

    def __init__(
        self,
        *,
        phi: float,
        sigma: float,
        alpha_inf: float,
        lambda_v: float,
        lambda_t: float,
        length: float,
        area: float,
        rho0: float = 1.213,
        c0: float = 342.2,
        name: str | None = None,
    ) -> None:
        if length <= 0.0:
            raise ValueError("length must be positive")
        if area <= 0.0:
            raise ValueError("area must be positive")

        self.length = float(length)
        self.area = float(area)
        self.material = JCAMaterial(
            sigma=float(sigma),
            thickness=float(length),
            phi=float(phi),
            lambda1=float(lambda_v),
            lambdap=float(lambda_t),
            tortu=float(alpha_inf),
            rho0=float(rho0),
            c0=float(c0),
            name=name,
        )
        self.model = JCAModel()

    def _k_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        freqs = omega / (2.0 * np.pi)
        props = self.model.properties(self.material, freqs)
        k = np.asarray(props.wavenumber, dtype=np.complex128).ravel()
        # Porous equivalent-fluid transfer in p/U uses an effective impedance scaled by porosity.
        z_char = np.asarray(props.impedance, dtype=np.complex128).ravel() / (self.area * self.material.phi)
        return k, z_char

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        k, z_char = self._k_zc(omega)
        kL = k * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((k.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = 1j * z_char * skL
        T[:, 1, 0] = 1j * skL / z_char
        T[:, 1, 1] = ckL
        return T


class MikiLayer(AcousticElement):
    """Miki equivalent-fluid porous slab as a 2x2 TMM element."""

    def __init__(
        self,
        *,
        sigma: float,
        length: float,
        area: float,
        rho0: float = 1.213,
        c0: float = 342.2,
        name: str | None = None,
    ) -> None:
        if length <= 0.0:
            raise ValueError("length must be positive")
        if area <= 0.0:
            raise ValueError("area must be positive")

        self.length = float(length)
        self.area = float(area)
        self.material = MikiMaterial(
            sigma=float(sigma),
            thickness=float(length),
            rho0=float(rho0),
            c0=float(c0),
            name=name,
        )
        self.model = MikiModel()

    def _k_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        freqs = omega / (2.0 * np.pi)
        props = self.model.properties(self.material, freqs)
        k = np.asarray(props.wavenumber, dtype=np.complex128).ravel()
        z_char = np.asarray(props.impedance, dtype=np.complex128).ravel() / self.area
        return k, z_char

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        k, z_char = self._k_zc(omega)
        kL = k * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((k.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = 1j * z_char * skL
        T[:, 1, 0] = 1j * skL / z_char
        T[:, 1, 1] = ckL
        return T
