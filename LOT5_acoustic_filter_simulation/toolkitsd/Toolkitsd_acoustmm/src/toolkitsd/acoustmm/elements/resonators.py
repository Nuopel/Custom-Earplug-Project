"""Reusable resonator constructions built from basic acoustic ingredients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AcousticElement
from .end_corrections import total_neck_end_correction
from .loss_model import KirchhoffStinsonEquivalentFluidModel, KirchhoffStinsonEquivalentFluidModelRectangular, LosslessCircularModel


@dataclass(frozen=True)
class HRImpedanceResult:
    """Minimal Helmholtz resonator impedance evaluation outputs."""

    Z_HR: np.ndarray
    Delta_l: float
    k_c: np.ndarray
    Z_c: np.ndarray
    k_n: np.ndarray
    Z_n: np.ndarray


@dataclass(frozen=True)
class RectangularHRImpedanceResult:
    """Minimal rectangular Helmholtz resonator impedance evaluation outputs."""

    Z_HR: np.ndarray
    Delta_l: float
    k_c: np.ndarray
    Z_c: np.ndarray
    k_n: np.ndarray
    Z_n: np.ndarray


class HelmholtzResonator:
    """Basic neck-cavity Helmholtz resonator model."""

    def __init__(
        self,
        *,
        radius_neck: float,
        length_neck: float,
        radius_cavity: float,
        length_cavity: float,
        radius_waveguide: float | None = None,
        outside_flanged: bool = False,
        c0: float,
        rho0: float,
        P0: float = 101325.0,
        eta0: float = 1.839e-5,
        Pr: float = 0.71,
        use_losses: bool = True,
        include_end_correction: bool = True,
    ) -> None:
        if radius_neck <= 0.0 or length_neck <= 0.0:
            raise ValueError("radius_neck and length_neck must be positive")
        if radius_cavity <= 0.0 or length_cavity <= 0.0:
            raise ValueError("radius_cavity and length_cavity must be positive")
        if radius_waveguide is not None and radius_waveguide <= 0.0:
            raise ValueError("radius_waveguide must be positive when provided")
        if c0 <= 0.0 or rho0 <= 0.0 or P0 <= 0.0 or eta0 <= 0.0 or Pr <= 0.0:
            raise ValueError("c0, rho0, P0, eta0, and Pr must be positive")

        self.radius_neck = float(radius_neck)
        self.length_neck = float(length_neck)
        self.radius_cavity = float(radius_cavity)
        self.length_cavity = float(length_cavity)
        self.radius_waveguide = None if radius_waveguide is None else float(radius_waveguide)
        self.outside_flanged = bool(outside_flanged)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.P0 = float(P0)
        self.eta0 = float(eta0)
        self.Pr = float(Pr)
        self.use_losses = bool(use_losses)
        self.include_end_correction = bool(include_end_correction)

        self.area_neck = np.pi * self.radius_neck**2
        self.area_cavity = np.pi * self.radius_cavity**2
        self.area_waveguide = None if self.radius_waveguide is None else np.pi * self.radius_waveguide**2

    def _model_cls(self):
        return KirchhoffStinsonEquivalentFluidModel if self.use_losses else LosslessCircularModel

    def _section_properties(self, *, radius: float, area: float, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        model_cls = self._model_cls()
        if self.use_losses:
            model = model_cls(
                radius=radius,
                area=area,
                c0=self.c0,
                rho0=self.rho0,
                P0=self.P0,
                eta0=self.eta0,
                Pr=self.Pr,
            )
        else:
            model = model_cls(
                radius=radius,
                area=area,
                c0=self.c0,
                rho0=self.rho0,
                P0=self.P0,
            )
        return model.equivalent_fluid_properties(omega)

    def end_correction(self) -> float:
        if not self.include_end_correction:
            return 0.0
        return total_neck_end_correction(
            self.radius_neck,
            self.radius_cavity,
            self.radius_waveguide,
            outside_flanged=self.outside_flanged,
        )

    def helmholtz_frequency_lumped_approximate(self, *, include_end_correction: bool | None = None) -> float:
        """Return the classical lossless lumped Helmholtz resonance estimate in Hz."""
        use_end_correction = self.include_end_correction if include_end_correction is None else bool(include_end_correction)
        Delta_l = self.end_correction() if use_end_correction else 0.0
        cavity_volume = self.area_cavity * self.length_cavity
        effective_neck_length = self.length_neck + Delta_l
        return (self.c0 / (2.0 * np.pi)) * np.sqrt(self.area_neck / (effective_neck_length * cavity_volume))

    def approximate_helmholtz_frequency(self, *, include_end_correction: bool | None = None) -> float:
        """Backward-compatible alias for the lumped Helmholtz estimate."""
        return self.helmholtz_frequency_lumped_approximate(include_end_correction=include_end_correction)

    def helmholtz_frequency_numerical(
        self,
        freq: np.ndarray | None = None,
        *,
        search_factor: tuple[float, float] = (0.25, 4.0),
        n_points: int = 4000,
    ) -> float:
        """Return the resonance frequency extracted from the minimum of |Z_HR|."""
        if freq is None:
            f_ref = self.helmholtz_frequency_lumped_approximate()
            f_min = max(1.0e-9, float(search_factor[0]) * f_ref)
            f_max = float(search_factor[1]) * f_ref
            if f_max <= f_min:
                raise ValueError("search_factor must define an increasing positive frequency range")
            freq = np.linspace(f_min, f_max, int(n_points), dtype=np.float64)
        else:
            freq = np.asarray(freq, dtype=np.float64).ravel()
            if freq.size == 0 or np.any(freq <= 0.0):
                raise ValueError("freq must contain strictly positive values")

        omega = 2.0 * np.pi * freq
        return float(freq[np.argmin(np.abs(self.impedance(omega)))])

    def impedance_result(self, omega: np.ndarray) -> HRImpedanceResult:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        _, _, k_c, Z_c = self._section_properties(radius=self.radius_cavity, area=self.area_cavity, omega=omega)
        _, _, k_n, Z_n = self._section_properties(radius=self.radius_neck, area=self.area_neck, omega=omega)

        Delta_l = self.end_correction() if self.include_end_correction else 0.0

        c_n = np.cos(k_n * self.length_neck)
        s_n = np.sin(k_n * self.length_neck)
        c_c = np.cos(k_c * self.length_cavity)
        s_c = np.sin(k_c * self.length_cavity)

        Z_HR = -1j * (
            c_n * c_c
            - (Z_n * k_n * c_n * s_c * Delta_l) / Z_c
            - (Z_n * s_n * s_c) / Z_c
        ) / (
            (s_n * c_c) / Z_n
            - (k_n * Delta_l * s_n * s_c) / Z_c
            + (c_n * s_c) / Z_c
        )

        return HRImpedanceResult(
            Z_HR=Z_HR,
            Delta_l=Delta_l,
            k_c=k_c,
            Z_c=Z_c,
            k_n=k_n,
            Z_n=Z_n,
        )

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        return self.impedance_result(omega).Z_HR


class HelmholtzResonatorRectangular:
    """Rectangular neck-cavity Helmholtz resonator model."""

    def __init__(
        self,
        *,
        width_neck: float,
        height_neck: float,
        length_neck: float,
        width_cavity: float,
        height_cavity: float,
        length_cavity: float,
        width_waveguide: float | None = None,
        height_waveguide: float | None = None,
        c0: float,
        rho0: float,
        P0: float = 101325.0,
        eta0: float = 1.839e-5,
        Pr: float = 0.71,
        use_losses: bool = True,
        include_end_correction: bool = True,
        n_modes: int = 24,
    ) -> None:
        if min(width_neck, height_neck, length_neck, width_cavity, height_cavity, length_cavity) <= 0.0:
            raise ValueError('All neck/cavity dimensions must be positive')
        if width_waveguide is not None and width_waveguide <= 0.0:
            raise ValueError('width_waveguide must be positive when provided')
        if height_waveguide is not None and height_waveguide <= 0.0:
            raise ValueError('height_waveguide must be positive when provided')
        if c0 <= 0.0 or rho0 <= 0.0 or P0 <= 0.0 or eta0 <= 0.0 or Pr <= 0.0:
            raise ValueError('c0, rho0, P0, eta0, and Pr must be positive')
        if n_modes < 1:
            raise ValueError('n_modes must be >= 1')

        self.width_neck = float(width_neck)
        self.height_neck = float(height_neck)
        self.length_neck = float(length_neck)
        self.width_cavity = float(width_cavity)
        self.height_cavity = float(height_cavity)
        self.length_cavity = float(length_cavity)
        self.width_waveguide = None if width_waveguide is None else float(width_waveguide)
        self.height_waveguide = None if height_waveguide is None else float(height_waveguide)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.P0 = float(P0)
        self.eta0 = float(eta0)
        self.Pr = float(Pr)
        self.use_losses = bool(use_losses)
        self.include_end_correction = bool(include_end_correction)
        self.n_modes = int(n_modes)

        self.area_neck = self.width_neck * self.height_neck
        self.area_cavity = self.width_cavity * self.height_cavity
        self.area_waveguide = None if self.width_waveguide is None or self.height_waveguide is None else self.width_waveguide * self.height_waveguide

    def _section_properties(self, *, width: float, height: float, area: float, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.use_losses:
            model = KirchhoffStinsonEquivalentFluidModelRectangular(
                width=width,
                height=height,
                area=area,
                c0=self.c0,
                rho0=self.rho0,
                P0=self.P0,
                eta0=self.eta0,
                Pr=self.Pr,
                n_modes=self.n_modes,
            )
            return model.equivalent_fluid_properties(omega)
        rho_eff = np.full(omega.shape, self.rho0 + 0j, dtype=np.complex128)
        K_eff = np.full(omega.shape, self.rho0 * self.c0**2 + 0j, dtype=np.complex128)
        k_eff = omega / self.c0
        zc_eff = np.full(omega.shape, self.rho0 * self.c0 / area + 0j, dtype=np.complex128)
        return rho_eff, K_eff, k_eff, zc_eff

    def end_correction(self) -> float:
        if not self.include_end_correction:
            return 0.0
        if self.width_waveguide is None:
            return 0.0
        delta_l_1 = 0.41 * (1.0 - 1.35 * (self.width_neck / self.width_cavity) + 0.31 * (self.width_neck / self.width_cavity) ** 3) * self.width_neck
        ratio = self.width_neck / self.width_waveguide
        delta_l_2 = 0.41 * (1.0 - 0.235 * ratio - 1.32 * ratio**2 + 1.54 * ratio**3 - 0.86 * ratio**4) * self.width_neck
        return delta_l_1 + delta_l_2

    def helmholtz_frequency_lumped_approximate(self, *, include_end_correction: bool | None = None) -> float:
        use_end_correction = self.include_end_correction if include_end_correction is None else bool(include_end_correction)
        Delta_l = self.end_correction() if use_end_correction else 0.0
        cavity_volume = self.area_cavity * self.length_cavity
        effective_neck_length = self.length_neck + Delta_l
        return (self.c0 / (2.0 * np.pi)) * np.sqrt(self.area_neck / (effective_neck_length * cavity_volume))

    def approximate_helmholtz_frequency(self, *, include_end_correction: bool | None = None) -> float:
        return self.helmholtz_frequency_lumped_approximate(include_end_correction=include_end_correction)

    def helmholtz_frequency_numerical(self, freq: np.ndarray | None = None, *, search_factor: tuple[float, float] = (0.25, 4.0), n_points: int = 4000) -> float:
        if freq is None:
            f_ref = self.helmholtz_frequency_lumped_approximate()
            f_min = max(1.0e-9, float(search_factor[0]) * f_ref)
            f_max = float(search_factor[1]) * f_ref
            if f_max <= f_min:
                raise ValueError('search_factor must define an increasing positive frequency range')
            freq = np.linspace(f_min, f_max, int(n_points), dtype=np.float64)
        else:
            freq = np.asarray(freq, dtype=np.float64).ravel()
            if freq.size == 0 or np.any(freq <= 0.0):
                raise ValueError('freq must contain strictly positive values')
        omega = 2.0 * np.pi * freq
        return float(freq[np.argmin(np.abs(self.impedance(omega)))])

    def impedance_result(self, omega: np.ndarray) -> RectangularHRImpedanceResult:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError('omega must have strictly positive real part')

        _, _, k_c, Z_c = self._section_properties(width=self.width_cavity, height=self.height_cavity, area=self.area_cavity, omega=omega)
        _, _, k_n, Z_n = self._section_properties(width=self.width_neck, height=self.height_neck, area=self.area_neck, omega=omega)
        Delta_l = self.end_correction() if self.include_end_correction else 0.0

        c_n = np.cos(k_n * self.length_neck)
        s_n = np.sin(k_n * self.length_neck)
        c_c = np.cos(k_c * self.length_cavity)
        s_c = np.sin(k_c * self.length_cavity)

        Z_HR = -1j * (
            c_n * c_c
            - (Z_n * k_n * c_n * s_c * Delta_l) / Z_c
            - (Z_n * s_n * s_c) / Z_c
        ) / (
            (s_n * c_c) / Z_n
            - (k_n * Delta_l * s_n * s_c) / Z_c
            + (c_n * s_c) / Z_c
        )

        return RectangularHRImpedanceResult(Z_HR=Z_HR, Delta_l=Delta_l, k_c=k_c, Z_c=Z_c, k_n=k_n, Z_n=Z_n)

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        return self.impedance_result(omega).Z_HR


class HelmholtzResonatorShunt(AcousticElement):
    """Shunt transfer-matrix element built from a Helmholtz resonator impedance."""

    def __init__(self, resonator: HelmholtzResonator | HelmholtzResonatorRectangular) -> None:
        self.resonator = resonator
        self.Z_HR = None

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        self.Z_HR = self.resonator.impedance(omega)
        T = np.zeros((self.Z_HR.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 1, 0] = 1.0 / self.Z_HR
        T[:, 1, 1] = 1.0
        return T
