"""Duct transfer-matrix elements."""

from __future__ import annotations

import warnings

import numpy as np
from ..mode_cutoffs import first_mode_round_duct
from .base import AcousticElement, SeriesImpedanceElement
from .loss_model import CircularLossModel, KirchhoffStinsonEquivalentFluidModel, KirchhoffStinsonEquivalentFluidModelRectangular, RectangularLossModel


class CylindricalDuct(AcousticElement):
    """Lossless cylindrical duct."""

    def __init__(
        self,
        radius: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        self.radius = float(radius)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.area = np.pi * self.radius**2
        self.Zc = self.rho0 * self.c0 / self.area
        self.first_mode_bc = "rigid"
        self.first_mode_id, self.first_mode_cutoff_hz = first_mode_round_duct(
            a=self.radius,
            c0=self.c0,
            bc=self.first_mode_bc,
        )

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        kL = (omega / self.c0) * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = 1j * self.Zc * skL
        T[:, 1, 0] = 1j * skL / self.Zc
        T[:, 1, 1] = ckL
        return T


class RectangularDuct(AcousticElement):
    """Lossless plane-wave rectangular duct."""

    def __init__(
        self,
        width: float,
        height: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
    ) -> None:
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        self.width = float(width)
        self.height = float(height)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.area = self.width * self.height
        self.Zc = self.rho0 * self.c0 / self.area

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        kL = (omega / self.c0) * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = 1j * self.Zc * skL
        T[:, 1, 0] = 1j * skL / self.Zc
        T[:, 1, 1] = ckL
        return T


class _ElasticSlabBase(AcousticElement):
    """Shared material relations for duct-confined longitudinal elastic slabs."""

    def __init__(
        self,
        radius: float,
        length: float,
        *,
        rho: float,
        young: float,
        poisson: float,
        loss_factor: float = 0.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if rho <= 0.0 or young <= 0.0:
            raise ValueError("rho and young must be positive")
        if not (-1.0 < poisson < 0.5):
            raise ValueError("poisson must be in (-1, 0.5)")
        if loss_factor < 0.0:
            raise ValueError("loss_factor must be >= 0")

        self.radius = float(radius)
        self.length = float(length)
        self.rho = float(rho)
        self.young = float(young)
        self.poisson = float(poisson)
        self.loss_factor = float(loss_factor)
        self.area = np.pi * self.radius**2

    @property
    def complex_young(self) -> complex:
        return self.young * (1.0 + 1j * self.loss_factor)

    @property
    def longitudinal_modulus(self) -> complex:
        return self.complex_young * (1.0 - self.poisson) / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))

    @property
    def longitudinal_speed(self) -> complex:
        return np.lib.scimath.sqrt(self.longitudinal_modulus / self.rho)

    @property
    def longitudinal_specific_impedance(self) -> complex:
        return self.rho * self.longitudinal_speed

    @property
    def longitudinal_acoustic_impedance(self) -> complex:
        return self.longitudinal_specific_impedance / self.area


class ElasticSlab(_ElasticSlabBase):
    """Exact 1D longitudinal elastic slab in a waveguide.

    This is a simple surrogate for a plug that fully fills the cross-section
    over a finite length. It is the exact slab propagator from the A5 note,
    expressed in the package state vector ``[p, U]``.
    """

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        j = 1j
        k_long = omega / self.longitudinal_speed
        zc_long = self.longitudinal_acoustic_impedance

        kL = k_long * self.length
        ckL = np.cos(kL)
        skL = np.sin(kL)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = ckL
        T[:, 0, 1] = j * zc_long * skL
        T[:, 1, 0] = j * skL / zc_long
        T[:, 1, 1] = ckL
        return T


class ElasticSlabThin(_ElasticSlabBase):
    """First-order thin-slab Taylor expansion from the A5 note.

    This keeps the mass term in ``T12`` and the compliance term in ``T21``.
    """

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        j = 1j
        z_series_specific = j * omega * self.rho * self.length * (1.0 + 1j * self.loss_factor)
        y_shunt_specific = j * omega * self.length / (self.rho * self.longitudinal_speed**2)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = z_series_specific / self.area
        T[:, 1, 0] = y_shunt_specific * self.area
        T[:, 1, 1] = 1.0
        return T


class ElasticSlabSeries(_ElasticSlabBase, SeriesImpedanceElement):
    """Mass-only series approximation from the A5 note."""

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        return 1j * omega * self.rho * self.length * (1.0 + 1j * self.loss_factor) / self.area

class ConicalDuct(AcousticElement):
    """Exact lossless conical duct using spherical-wave cone coordinates."""

    def __init__(
        self,
        r1: float,
        r2: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
    ) -> None:
        if r1 <= 0.0 or r2 <= 0.0:
            raise ValueError("r1 and r2 must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        if np.isclose(self.r1, self.r2, rtol=0.0, atol=1e-12):
            return CylindricalDuct(radius=self.r1, length=self.length, c0=self.c0, rho0=self.rho0).matrix(
                omega
            )

        # Cone slope alpha where r(x) = alpha * x in apex coordinates.
        alpha = (self.r2 - self.r1) / self.length
        if np.isclose(alpha, 0.0, rtol=0.0, atol=1e-18):
            return CylindricalDuct(radius=self.r1, length=self.length, c0=self.c0, rho0=self.rho0).matrix(
                omega
            )

        x1 = self.r1 / alpha
        x2 = self.r2 / alpha
        if np.isclose(x1, 0.0, atol=1e-15) or np.isclose(x2, 0.0, atol=1e-15):
            raise ValueError("Conical apex too close to section boundary; exact formulation is singular.")

        k = omega / self.c0
        j = 1j
        omega_solid = np.pi * alpha**2
        pref = omega_solid / (self.rho0 * self.c0)

        km1 = k * x1
        km2 = k * x2
        em1 = np.exp(-j * km1)
        ep1 = np.exp(j * km1)
        em2 = np.exp(-j * km2)
        ep2 = np.exp(j * km2)

        F1 = np.empty((omega.size, 2, 2), dtype=np.complex128)
        F2 = np.empty((omega.size, 2, 2), dtype=np.complex128)

        F1[:, 0, 0] = em1 / x1
        F1[:, 0, 1] = ep1 / x1
        F1[:, 1, 0] = pref * x1 * (1.0 + 1.0 / (j * km1)) * em1
        F1[:, 1, 1] = -pref * x1 * (1.0 - 1.0 / (j * km1)) * ep1

        F2[:, 0, 0] = em2 / x2
        F2[:, 0, 1] = ep2 / x2
        F2[:, 1, 0] = pref * x2 * (1.0 + 1.0 / (j * km2)) * em2
        F2[:, 1, 1] = -pref * x2 * (1.0 - 1.0 / (j * km2)) * ep2

        return np.einsum("nij,njk->nik", F1, np.linalg.inv(F2))


class ViscothermalConicalDuctDiscrete(AcousticElement):
    """Discrete viscothermal conical duct built from cascaded cylindrical slices.

    This is a geometrically conical element whose thermoviscous losses are
    modeled by discretizing the cone into ``n_sub`` short
    :class:`ViscothermalDuct` sections using midpoint radii.
    """

    def __init__(
        self,
        r1: float,
        r2: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
        *,
        n_sub: int = 32,
        eta0: float = 1.839e-5,
        P0: float = 101325.0,
        Pr: float = 0.71,
        loss_model: CircularLossModel | None = None,
    ) -> None:
        if r1 <= 0.0 or r2 <= 0.0:
            raise ValueError("r1 and r2 must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        if n_sub < 1:
            raise ValueError("n_sub must be >= 1")
        if eta0 <= 0.0 or P0 <= 0.0 or Pr <= 0.0:
            raise ValueError("eta0, P0, and Pr must be positive")

        self.r1 = float(r1)
        self.r2 = float(r2)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.n_sub = int(n_sub)
        self.eta0 = float(eta0)
        self.P0 = float(P0)
        self.Pr = float(Pr)
        self.loss_model = loss_model

    def _equivalent_system(self) -> AcousticElement:
        if np.isclose(self.r1, self.r2, rtol=0.0, atol=1e-12):
            return ViscothermalDuct(
                radius=self.r1,
                length=self.length,
                c0=self.c0,
                rho0=self.rho0,
                eta0=self.eta0,
                P0=self.P0,
                Pr=self.Pr,
                loss_model=self.loss_model,
            )

        radii = np.linspace(self.r1, self.r2, self.n_sub + 1, dtype=float)
        r_mid = 0.5 * (radii[:-1] + radii[1:])
        sub_length = self.length / self.n_sub

        segments = [
            ViscothermalDuct(
                radius=float(r),
                length=sub_length,
                c0=self.c0,
                rho0=self.rho0,
                eta0=self.eta0,
                P0=self.P0,
                Pr=self.Pr,
                loss_model=self.loss_model,
            )
            for r in r_mid
        ]
        return sum(segments)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        return self._equivalent_system().matrix(omega)


class ViscothermalDuct(AcousticElement):
    """Kirchhoff/Stinson viscothermal cylindrical duct model."""

    def __init__(
        self,
        radius: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
        *,
        eta0: float = 1.839e-5,
        P0: float = 101325.0,
        Pr: float = 0.71,
        loss_model: CircularLossModel | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        if eta0 <= 0.0 or P0 <= 0.0 or Pr <= 0.0:
            raise ValueError("eta0, P0, and Pr must be positive")

        self.radius = float(radius)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.eta0 = float(eta0)
        self.P0 = float(P0)
        self.Pr = float(Pr)
        self.area = np.pi * self.radius**2
        self.loss_model = loss_model or KirchhoffStinsonEquivalentFluidModel(
            radius=self.radius,
            area=self.area,
            c0=self.c0,
            rho0=self.rho0,
            P0=self.P0,
            eta0=self.eta0,
            Pr=self.Pr,
        )
        self.Zc_lossless = self.loss_model.zc_lossless

    def _gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.loss_model.gamma_zc(omega)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        gamma_vt, zc_vt = self._gamma_zc(omega)
        gL = gamma_vt * self.length
        cgL = np.cosh(gL)
        sgL = np.sinh(gL)

        T = np.zeros((gamma_vt.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = cgL
        T[:, 0, 1] = zc_vt * sgL  # ← plus de 1j
        T[:, 1, 0] = sgL / zc_vt  # ← plus de 1j
        T[:, 1, 1] = cgL
        return T


class ViscothermalRectangularDuct(AcousticElement):
    """Kirchhoff/Stinson viscothermal rectangular duct model."""

    def __init__(
        self,
        width: float,
        height: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
        *,
        eta0: float = 1.839e-5,
        P0: float = 101325.0,
        Pr: float = 0.71,
        n_modes: int = 24,
        loss_model: RectangularLossModel | None = None,
    ) -> None:
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        if eta0 <= 0.0 or P0 <= 0.0 or Pr <= 0.0:
            raise ValueError("eta0, P0, and Pr must be positive")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")

        self.width = float(width)
        self.height = float(height)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.eta0 = float(eta0)
        self.P0 = float(P0)
        self.Pr = float(Pr)
        self.n_modes = int(n_modes)
        self.area = self.width * self.height
        self.loss_model = loss_model or KirchhoffStinsonEquivalentFluidModelRectangular(
            width=self.width,
            height=self.height,
            area=self.area,
            c0=self.c0,
            rho0=self.rho0,
            P0=self.P0,
            eta0=self.eta0,
            Pr=self.Pr,
            n_modes=self.n_modes,
        )
        self.Zc_lossless = self.loss_model.zc_lossless

    def _gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.loss_model.gamma_zc(omega)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        gamma_vt, zc_vt = self._gamma_zc(omega)
        gL = gamma_vt * self.length
        cgL = np.cosh(gL)
        sgL = np.sinh(gL)

        T = np.zeros((gamma_vt.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = cgL
        T[:, 0, 1] = zc_vt * sgL
        T[:, 1, 0] = sgL / zc_vt
        T[:, 1, 1] = cgL
        return T


class BLIDuct(AcousticElement):
    """Simplified boundary-layer impedance (BLI) cylindrical duct model."""

    def __init__(
        self,
        radius: float,
        length: float,
        c0: float = 340.0,
        rho0: float = 1.2,
        *,
        eta0: float = 1.839e-5,
        P0: float = 101325.0,
        Pr: float = 0.71,
        correct_zc: bool = True,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        if eta0 <= 0.0 or P0 <= 0.0 or Pr <= 0.0:
            raise ValueError("eta0, P0, and Pr must be positive")

        self.radius = float(radius)
        self.length = float(length)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.eta0 = float(eta0)
        self.P0 = float(P0)
        self.Pr = float(Pr)
        self.correct_zc = bool(correct_zc)
        self.area = np.pi * self.radius**2
        self.Zc_lossless = self.rho0 * self.c0 / self.area
        self.gamma = self.rho0 * self.c0**2 / self.P0

        # Validity hint for the BLI asymptotic model (rule-of-thumb at 1 kHz).
        delta_v_1khz = np.sqrt(2.0 * self.eta0 / (self.rho0 * 2.0 * np.pi * 1000.0))
        if self.radius < 5.0 * delta_v_1khz:
            warnings.warn(
                "BLIDuct validity may be marginal for this radius "
                f"(r={self.radius*1e3:.3f} mm, r/delta_v@1kHz={self.radius/delta_v_1khz:.2f}). "
                "Consider ViscothermalDuct for sub-mm bores.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        a = self.radius
        k = omega / self.c0
        delta_v = np.sqrt(2.0 * self.eta0 / (self.rho0 * omega))
        delta_t = np.sqrt(2.0 * self.eta0 / (self.rho0 * self.Pr * omega))
        corr = 0.5 * (1.0 - 1.0j) * ((delta_v / a) + (self.gamma - 1.0) * (delta_t / a))

        # k_eff follows simplified BLI correction; convert to propagation constant Gamma.
        k_eff = k * (1.0 + corr)
        gamma_bli = 1j * k_eff
        gamma_bli = np.where(np.real(gamma_bli) < 0.0, -gamma_bli, gamma_bli)

        if self.correct_zc:
            zc_corr = 0.5 * (1.0 - 1.0j) * ((delta_v / a) - (self.gamma - 1.0) * (delta_t / a))
            zc_bli = self.Zc_lossless * (1.0 + zc_corr)
            zc_bli = np.where(np.real(zc_bli) < 0.0, -zc_bli, zc_bli)
        else:
            zc_bli = np.full(omega.shape, self.Zc_lossless + 0j, dtype=np.complex128)
        return gamma_bli, zc_bli

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        gamma_bli, zc_bli = self._gamma_zc(omega)
        gL = gamma_bli * self.length
        cgL = np.cosh(gL)
        sgL = np.sinh(gL)

        T = np.zeros((gamma_bli.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = cgL
        T[:, 0, 1] = zc_bli * sgL
        T[:, 1, 0] = sgL / zc_bli
        T[:, 1, 1] = cgL
        return T


class FlowDuct(AcousticElement):
    """Cylindrical duct with uniform mean-flow Mach correction."""

    def __init__(
        self,
        radius: float,
        length: float,
        mach: float = 0.0,
        c0: float = 340.0,
        rho0: float = 1.2,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if length <= 0.0:
            raise ValueError("length must be positive")
        if c0 <= 0.0 or rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        if abs(mach) >= 1.0:
            raise ValueError("mach must satisfy |mach| < 1")

        self.radius = float(radius)
        self.length = float(length)
        self.mach = float(mach)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.area = np.pi * self.radius**2
        self.Zc = self.rho0 * self.c0 / self.area

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        k = omega / self.c0
        k_plus = k / (1.0 + self.mach)
        k_minus = k / (1.0 - self.mach)

        e_plus = np.exp(1j * k_plus * self.length)
        e_minus = np.exp(-1j * k_minus * self.length)

        s = 0.5 * (e_plus + e_minus)
        d = 0.5 * (e_plus - e_minus)

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = s
        T[:, 0, 1] = self.Zc * d
        T[:, 1, 0] = d / self.Zc
        T[:, 1, 1] = s
        return T
