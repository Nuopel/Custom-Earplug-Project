"""Boundary impedances used as load terminations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import j1, jv, struve, yv


@dataclass(frozen=True)
class RigidWall:
    """Hard-wall boundary: infinite load impedance."""

    def Z(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")
        return np.full(omega.shape, np.inf + 0j, dtype=np.complex128)


@dataclass(frozen=True)
class MatchedLoad:
    """Anechoic load matched to duct characteristic impedance."""

    area: float
    c0: float = 340.0
    rho0: float = 1.2

    def __post_init__(self) -> None:
        if self.area <= 0.0:
            raise ValueError("area must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")

    def Z(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")
        Zc = self.rho0 * self.c0 / self.area
        return np.full(omega.shape, Zc + 0j, dtype=np.complex128)


@dataclass(frozen=True)
class RadiationImpedance:
    """Radiation impedance for circular tube termination."""

    radius: float
    mode: str = "unflanged"
    c0: float = 340.0
    rho0: float = 1.2

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0:
            raise ValueError("c0 and rho0 must be positive")
        mode = self.mode.lower()
        if mode not in ("unflanged", "unflanged_v2", "flanged"):
            raise ValueError("mode must be 'unflanged', 'unflanged_v2', or 'flanged'")
        object.__setattr__(self, "mode", mode)

    @property
    def area(self) -> float:
        return np.pi * self.radius**2

    def Z(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        k = omega / self.c0
        ka = k * self.radius
        zref = self.rho0 * self.c0 / self.area

        if self.mode == "unflanged":
            z_norm = (ka**2) / 4.0 + 1j * 0.6133 * ka
            return zref * z_norm
        if self.mode == "unflanged_v2":
            # Extended unflanged open-tube approximation with higher-order/log terms.
            term1 = 1j * ka * 0.6133
            term2 = -1j * (ka**3) * (0.036 - 0.034 * np.log(ka) + 0.0187 * (ka**2))
            term3 = (ka**2) / 4.0 + (ka**4) * (0.0127 + 0.082 * np.log(ka) - 0.023 * (ka**2))
            return zref * (term1 + term2 + term3)

        # Levine-Schwinger flanged expression.
        two_ka = 2.0 * ka
        z_norm = 1.0 - j1(two_ka) / ka + 1j * struve(1, two_ka) / ka
        return zref * z_norm


@dataclass(frozen=True)
class EardrumImpedance:
    """Shaw-like lumped first-pass middle-ear termination model."""

    R1: float = 1.0e8
    M1: float = 1.4e-3
    C1: float = 1.2e-12
    R_coch: float = 1.0e9

    def __post_init__(self) -> None:
        if self.R1 <= 0.0 or self.M1 <= 0.0 or self.C1 <= 0.0 or self.R_coch <= 0.0:
            raise ValueError("R1, M1, C1, and R_coch must be positive")

    def Z(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        j = 1j
        z_tm = self.R1 + j * omega * self.M1 + 1.0 / (j * omega * self.C1)
        return z_tm + self.R_coch


@dataclass(frozen=True)
class IEC711Coupler:
    """IEC 60318-4 / IEC 711 occluded-ear simulator impedance.

    The default model is a transfer-matrix implementation of the two-resonator
    coupler geometry used in the IEC 60318-4 literature. A legacy
    compliance-only mode is still available for simple first-pass studies.
    """

    model: str = "tmm"
    volume_m3: float = 2.0e-6
    c0: float = 343
    rho0: float = 1.2
    R_loss: float = 0.0
    mu: float = 1.82e-5
    gamma: float = 1.4
    Cp: float = 1.0e3
    lam: float = 24.80e-3
    hr1_cavity_model: str = "bessel"
    hr2_cavity_model: str = "lumped"
    R0: float = 3.77e-3
    L1: float = 3.12e-3
    L3: float = 4.75e-3
    L5: float = 4.69e-3
    a2: float = 2.53e-3
    b2: float = 2.35e-3
    h2: float = 0.16e-3
    r2: float = 6.30e-3
    R2: float = 9.01e-3
    d1: float = 1.91e-3
    r4: float = 4.66e-3
    alpha_deg: float = 95.33
    h4: float = 0.05e-3
    R4: float = 9.01e-3
    d2: float = 1.40e-3

    def __post_init__(self) -> None:
        model = self.model.lower()
        if model not in {"tmm", "lumped", "compliance"}:
            raise ValueError("model must be 'tmm', 'lumped', or 'compliance'")
        object.__setattr__(self, "model", model)
        for attr in ("hr1_cavity_model", "hr2_cavity_model"):
            value = getattr(self, attr).lower()
            if value not in {"bessel", "lumped"}:
                raise ValueError(f"{attr} must be 'bessel' or 'lumped'")
            object.__setattr__(self, attr, value)
        if self.volume_m3 <= 0.0:
            raise ValueError("volume_m3 must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0 or self.mu <= 0.0 or self.gamma <= 1.0 or self.Cp <= 0.0 or self.lam <= 0.0:
            raise ValueError("air properties must be positive, with gamma > 1")
        if self.R_loss < 0.0:
            raise ValueError("R_loss must be >= 0")
        for attr in (
            "R0",
            "L1",
            "L3",
            "L5",
            "a2",
            "b2",
            "h2",
            "r2",
            "R2",
            "d1",
            "r4",
            "h4",
            "R4",
            "d2",
        ):
            if getattr(self, attr) <= 0.0:
                raise ValueError(f"{attr} must be positive")
        if self.r2 >= self.R2:
            raise ValueError("r2 must be < R2")
        if self.r4 >= self.R4:
            raise ValueError("r4 must be < R4")
        if self.r4 <= self.R0:
            raise ValueError("r4 must be > R0 for the annular slit geometry")
        if self.alpha_deg <= 0.0:
            raise ValueError("alpha_deg must be positive")

    @property
    def compliance(self) -> float:
        return self.volume_m3 / (self.rho0 * self.c0**2)

    @property
    def _main_area(self) -> float:
        return np.pi * self.R0**2

    @property
    def _alpha_rad(self) -> float:
        return np.deg2rad(self.alpha_deg)

    def _end_corr_slit(self, h: float, b: float) -> float:
        beta = h / b
        eps = 1.0 + beta**2
        dl_h = (
            (1.0 / (3.0 * np.pi)) * (beta + (1.0 - eps**1.5) / beta**2)
            + (1.0 / np.pi)
            * (
                np.log(beta + np.sqrt(eps)) / beta
                + np.log((1.0 + np.sqrt(eps)) / beta)
            )
        )
        return dl_h * h

    def _kfield(self, k_bnd: np.ndarray, h: float) -> np.ndarray:
        arg = k_bnd * h / 2.0
        return 1.0 - np.tan(arg) / arg

    def _lrf_kz(self, k0: np.ndarray, h: float, area: float) -> tuple[np.ndarray, np.ndarray]:
        lh = self.lam / (self.rho0 * self.c0 * self.Cp)
        lv = self.mu / (self.rho0 * self.c0)
        kh = (1.0 - 1.0j) / np.sqrt(2.0) * np.sqrt(k0 / lh)
        kv = (1.0 - 1.0j) / np.sqrt(2.0) * np.sqrt(k0 / lv)
        Kh = self._kfield(kh, h)
        Kv = self._kfield(kv, h)
        Khp = self.gamma - (self.gamma - 1.0) * Kh
        kl = k0 * np.sqrt(Khp / Kv)
        Zl = (self.rho0 * self.c0 / area) / np.sqrt(Khp * Kv)
        return kl, Zl

    def _annular_cavity_impedance(
        self,
        k0: np.ndarray,
        r_inner: float,
        r_outer: float,
        area: float,
    ) -> np.ndarray:
        Bs = yv(1, k0 * r_outer) / jv(1, k0 * r_outer)
        return (
            1j
            * self.rho0
            * self.c0
            * (Bs * jv(0, k0 * r_inner) - yv(0, k0 * r_inner))
            / (area * (Bs * jv(1, k0 * r_inner) - yv(1, k0 * r_inner)))
        )

    def _hr1_impedance(self, omega: np.ndarray, k0: np.ndarray) -> np.ndarray:
        slit_area = self.b2 * self.h2
        kl2, Zl2 = self._lrf_kz(k0, self.h2, slit_area)
        dl2 = self._end_corr_slit(self.h2, self.b2)
        Z_slit2 = 1j * Zl2 * np.tan(kl2 * (self.a2 + 2.0 * dl2))

        if self.hr1_cavity_model == "bessel":
            cav_area = np.pi * (self.R2**2 - self.r2**2)
            Z_cav2 = self._annular_cavity_impedance(k0, self.r2, self.R2, cav_area)
        else:
            cav_volume = np.pi * (self.R2**2 - self.r2**2) * self.d1
            Z_cav2 = -1j * self.rho0 * self.c0**2 / (omega * cav_volume)
        return Z_slit2 + Z_cav2

    def _hr2_impedance(self, omega: np.ndarray, k0: np.ndarray) -> np.ndarray:
        r_mean4 = 0.5 * (self.R0 + self.r4)
        slit_area = 3.0 * self._alpha_rad * r_mean4 * self.h4
        kl4, Zl4 = self._lrf_kz(k0, self.h4, slit_area)

        p_in = 3.0 * self._alpha_rad * self.R0
        p_out = 3.0 * self._alpha_rad * self.r4
        R0_in = self.R0 - self._end_corr_slit(self.h4, p_in)
        r4_out = self.r4 + self._end_corr_slit(self.h4, p_out)

        As = yv(0, kl4 * r4_out) / jv(0, kl4 * r4_out)
        Z_slit4 = (
            1j
            * Zl4
            * (As * jv(0, kl4 * R0_in) - yv(0, kl4 * R0_in))
            / (As * jv(1, kl4 * R0_in) - yv(1, kl4 * R0_in))
        )

        if self.hr2_cavity_model == "bessel":
            cav_area = np.pi * (self.R4**2 - self.r4**2)
            Z_cav4 = self._annular_cavity_impedance(k0, self.r4, self.R4, cav_area)
        else:
            cav_volume = np.pi * (self.R4**2 - self.r4**2) * self.d2
            Z_cav4 = -1j * self.rho0 * self.c0**2 / (omega * cav_volume)
        return Z_slit4 + Z_cav4

    def _tm_tube(self, k0: np.ndarray, Z1: float, length: float) -> np.ndarray:
        T = np.zeros((2, 2, k0.size), dtype=np.complex128)
        c = np.cos(k0 * length)
        s = np.sin(k0 * length)
        T[0, 0] = c
        T[0, 1] = 1j * Z1 * s
        T[1, 0] = 1j * s / Z1
        T[1, 1] = c
        return T

    def _tm_shunt(self, impedance: np.ndarray) -> np.ndarray:
        T = np.zeros((2, 2, impedance.size), dtype=np.complex128)
        T[0, 0] = 1.0
        T[1, 1] = 1.0
        T[1, 0] = 1.0 / impedance
        return T

    def _matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.einsum("ijk,jlk->ilk", A, B)

    def _lumped_hr_parameters(self) -> dict[str, float]:
        slit_area2 = self.b2 * self.h2
        r_a2 = 12.0 * self.mu * self.a2 / (self.b2 * self.h2**3)
        m_a2 = 6.0 * self.rho0 * self.a2 / (5.0 * slit_area2)
        c_a2 = np.pi * (self.R2**2 - self.r2**2) * self.d1 / (self.rho0 * self.c0**2)

        alpha_r = self._alpha_rad
        a4_ = self.r4 - self.R0
        b4_ = 3.0 * alpha_r * (self.R0 + self.r4) / 2.0
        slit_area4 = b4_ * self.h4
        r_a4 = 12.0 * self.mu * a4_ / (b4_ * self.h4**3)
        m_a4 = 6.0 * self.rho0 * a4_ / (5.0 * slit_area4)
        c_a4 = np.pi * (self.R4**2 - self.r4**2) * self.d2 / (self.rho0 * self.c0**2)

        return {
            "r_a2": r_a2,
            "m_a2": m_a2,
            "c_a2": c_a2,
            "r_a4": r_a4,
            "m_a4": m_a4,
            "c_a4": c_a4,
        }

    def _cav_lumped(self, length: float) -> tuple[float, float]:
        m = self.rho0 * length / self._main_area
        c = self._main_area * length / (self.rho0 * self.c0**2)
        return m, c

    def _lumped_impedance(self, omega: np.ndarray) -> np.ndarray:
        m1, c1 = self._cav_lumped(self.L1)
        m3, c3 = self._cav_lumped(self.L3)
        m5, c5 = self._cav_lumped(self.L5)
        pars = self._lumped_hr_parameters()

        Z_hr1 = pars["r_a2"] + 1j * omega * pars["m_a2"] + 1.0 / (1j * omega * pars["c_a2"])
        Z_hr2 = pars["r_a4"] + 1j * omega * pars["m_a4"] + 1.0 / (1j * omega * pars["c_a4"])

        Z = 1.0 / (1j * omega * c5)
        Z = Z + 1j * omega * m5
        Z = 1.0 / (1.0 / Z + 1j * omega * c3 + 1.0 / Z_hr2)
        Z = Z + 1j * omega * m3
        Z = 1.0 / (1.0 / Z + 1j * omega * c1 + 1.0 / Z_hr1)
        return Z + 1j * omega * m1

    def branch_resonance_frequencies(self, omega: np.ndarray | None = None) -> dict[str, float]:
        if self.model == "compliance":
            return {}
        if self.model == "lumped":
            pars = self._lumped_hr_parameters()
            return {
                "hr1": 1.0 / (2.0 * np.pi * np.sqrt(pars["m_a2"] * pars["c_a2"])),
                "hr2": 1.0 / (2.0 * np.pi * np.sqrt(pars["m_a4"] * pars["c_a4"])),
            }
        if omega is None:
            raise ValueError("omega must be provided to estimate TMM branch resonances")
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        k0 = omega / self.c0
        f = omega / (2.0 * np.pi)
        Z_hr1 = self._hr1_impedance(omega, k0)
        Z_hr2 = self._hr2_impedance(omega, k0)
        return {
            "hr1": float(f[np.argmin(np.abs(Z_hr1))]),
            "hr2": float(f[np.argmin(np.abs(Z_hr2))]),
        }

    def Z(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")
        if self.model == "compliance":
            z_comp = 1.0 / (1j * omega * self.compliance)
            return self.R_loss + z_comp
        if self.model == "lumped":
            return self._lumped_impedance(omega)

        k0 = omega / self.c0
        Z1 = self.rho0 * self.c0 / self._main_area
        Z_hr1 = self._hr1_impedance(omega, k0)
        Z_hr2 = self._hr2_impedance(omega, k0)

        T_es = self._matmul(
            self._tm_tube(k0, Z1, self.L1),
            self._matmul(
                self._tm_shunt(Z_hr1),
                self._matmul(
                    self._tm_tube(k0, Z1, self.L3),
                    self._matmul(
                        self._tm_shunt(Z_hr2),
                        self._tm_tube(k0, Z1, self.L5),
                    ),
                ),
            ),
        )

        Rs = (T_es[0, 0] - T_es[1, 0] * Z1) / (T_es[0, 0] + T_es[1, 0] * Z1)
        return Z1 * (1.0 + Rs) / (1.0 - Rs)
