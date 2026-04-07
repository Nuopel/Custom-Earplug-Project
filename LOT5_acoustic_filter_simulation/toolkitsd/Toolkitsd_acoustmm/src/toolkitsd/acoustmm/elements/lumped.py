"""Lumped/acoustic discontinuity and plate-derived waveguide elements."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import iv, jv

from .base import AcousticElement, SeriesImpedanceElement
from .infinite_layers import _plate_specific_impedance


class ImpedanceJunction(AcousticElement):
    """Area discontinuity for a [p, U] state vector.

    For an ideal 1D discontinuity:
        p1 = p2
        U1 = U2
    so the transfer matrix is the identity.

    Optionally, a lumped reactive correction can be added as a series impedance:
        p1 = p2 + Z_mass * U2
        U1 = U2
    """

    def __init__(
        self,
        S1: float,
        S2: float,
        *,
        end_correction: bool = False,
        rho0: float = 1.2,
    ) -> None:
        if S1 <= 0.0 or S2 <= 0.0:
            raise ValueError("S1 and S2 must be positive")
        if rho0 <= 0.0:
            raise ValueError("rho0 must be positive")

        self.S1 = float(S1)
        self.S2 = float(S2)
        self.end_correction = bool(end_correction)
        self.rho0 = float(rho0)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        T = np.zeros((omega.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 1, 1] = 1.0

        if not self.end_correction:
            return T

        S_small = min(self.S1, self.S2)
        r_small = np.sqrt(S_small / np.pi)
        delta_l = 0.6133 * r_small
        Z_mass = 1j * omega * self.rho0 * delta_l / S_small

        T[:, 0, 1] = Z_mass
        return T


class PlateSeriesImpedance(SeriesImpedanceElement):
    """Area-normalized series plate element for an acoustic duct chain.

    The underlying plate relation is shared with ``InfinitePlate``.
    For use as a transverse series obstruction in a 1D acoustic duct, this
    impedance is normalized by the duct cross-section area to obtain an
    acoustic series impedance in Pa.s/m^3.
    """

    def __init__(
        self,
        *,
        area: float,
        rho_plate: float,
        h: float,
        E: float,
        nu: float,
        theta: float = 0.0,
        c0: float = 340.0,
    ) -> None:
        if area <= 0.0:
            raise ValueError("area must be positive")
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
        self.area = float(area)
        self.rho_plate = float(rho_plate)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.theta = float(theta)
        self.c0 = float(c0)

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        return _plate_specific_impedance(
            omega,
            rho_plate=self.rho_plate,
            h=self.h,
            E=self.E,
            nu=self.nu,
            theta=self.theta,
            c0=self.c0,
        ) / self.area

class GenericFilmSeriesImpedance(SeriesImpedanceElement):
    """Generic film-like series impedance ``R + j*omega*M + K/(j*omega)``."""

    def __init__(
        self,
        *,
        resistance: np.ndarray | complex | float = 0.0,
        mass: np.ndarray | complex | float = 0.0,
        stiffness: np.ndarray | complex | float = 0.0,
    ) -> None:
        self.resistance = resistance
        self.mass = mass
        self.stiffness = stiffness

    @staticmethod
    def _as_frequency_parameter(value: np.ndarray | complex | float, omega: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.complex128)
        if arr.ndim == 0:
            return np.full(omega.shape, arr, dtype=np.complex128)
        arr = arr.ravel()
        if arr.size != omega.size:
            raise ValueError(f"{name} and omega must have same length, got {arr.size} and {omega.size}")
        return arr

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        resistance = self._as_frequency_parameter(self.resistance, omega, "resistance")
        mass = self._as_frequency_parameter(self.mass, omega, "mass")
        stiffness = self._as_frequency_parameter(self.stiffness, omega, "stiffness")
        return resistance + 1j * omega * mass + stiffness / (1j * omega)


class MembraneSeriesImpedance(SeriesImpedanceElement):
    """Lumped membrane-like series impedance from surface density and tension."""

    def __init__(
        self,
        *,
        radius: float,
        surface_density: float,
        tension: float,
        resistance: np.ndarray | complex | float = 0.0,
        geometry_constant: float = 1.0,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if surface_density <= 0.0:
            raise ValueError("surface_density must be positive")
        if tension < 0.0:
            raise ValueError("tension must be non-negative")
        if geometry_constant <= 0.0:
            raise ValueError("geometry_constant must be positive")
        self.radius = float(radius)
        self.surface_density = float(surface_density)
        self.tension = float(tension)
        self.resistance = resistance
        self.geometry_constant = float(geometry_constant)
        self.area = np.pi * self.radius**2

    @property
    def equivalent_mass(self) -> float:
        return self.surface_density / self.area

    @property
    def equivalent_stiffness(self) -> float:
        return self.geometry_constant * self.tension / (self.radius**2 * self.area)

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        return GenericFilmSeriesImpedance(
            resistance=self.resistance,
            mass=self.equivalent_mass,
            stiffness=self.equivalent_stiffness,
        ).acoustic_series_impedance(omega)


class ExactFlexuralPlateSeriesImpedance(SeriesImpedanceElement):
    """Exact clamped circular flexural-plate series impedance from the D1 equation."""

    def __init__(
        self,
        *,
        radius: float,
        rho_plate: float,
        h: float,
        E: float,
        nu: float,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if rho_plate <= 0.0:
            raise ValueError("rho_plate must be positive")
        if h <= 0.0:
            raise ValueError("h must be positive")
        if E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")
        self.radius = float(radius)
        self.rho_plate = float(rho_plate)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.area = np.pi * self.radius**2

    @property
    def bending_stiffness(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def plate_mass(self) -> float:
        return self.rho_plate * self.h * self.area

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        flexural_wavenumber = (self.rho_plate * self.h * omega**2 / self.bending_stiffness) ** 0.25
        x = flexural_wavenumber * self.radius
        num = iv(1, x) * jv(0, x) + jv(1, x) * iv(0, x)
        den = iv(1, x) * jv(2, x) - jv(1, x) * iv(2, x)
        return -1j * omega * self.plate_mass / (self.area**2) * (num / den)


class LowFrequencyFlexuralPlateSeriesImpedance(SeriesImpedanceElement):
    """Low-frequency flexural-plate series impedance from the D2 approximation."""

    def __init__(
        self,
        *,
        radius: float,
        rho_plate: float,
        h: float,
        E: float,
        nu: float,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if rho_plate <= 0.0:
            raise ValueError("rho_plate must be positive")
        if h <= 0.0:
            raise ValueError("h must be positive")
        if E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")
        self.radius = float(radius)
        self.rho_plate = float(rho_plate)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.area = np.pi * self.radius**2

    @property
    def bending_stiffness(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def surface_mass(self) -> float:
        return self.rho_plate * self.h

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        flexural_wavenumber = (self.rho_plate * self.h * omega**2 / self.bending_stiffness) ** 0.25
        x = flexural_wavenumber * self.radius
        return -1j * omega * self.surface_mass * (192.0 / self.area) * (1.0 / x**4 - 3.0 / 320.0)


class FlexuralPlateSeriesImpedance(AcousticElement):
    """Series flexural circular-plate impedance matching the historical A3 model."""

    def __init__(
        self,
        *,
        radius: float,
        rho_plate: float,
        h: float,
        E: float,
        nu: float,
        rho0: float | None = None,
        c0: float | None = None,
        cell_length: float | None = None,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive")
        if rho_plate <= 0.0:
            raise ValueError("rho_plate must be positive")
        if h <= 0.0:
            raise ValueError("h must be positive")
        if E <= 0.0:
            raise ValueError("E must be positive")
        if not (-1.0 < nu < 0.5):
            raise ValueError("nu must be in (-1, 0.5)")
        self.radius = float(radius)
        self.rho_plate = float(rho_plate)
        self.h = float(h)
        self.E = float(E)
        self.nu = float(nu)
        self.rho0 = None if rho0 is None else float(rho0)
        self.c0 = None if c0 is None else float(c0)
        self.cell_length = None if cell_length is None else float(cell_length)
        if self.rho0 is not None and self.rho0 <= 0.0:
            raise ValueError("rho0 must be positive when provided")
        if self.c0 is not None and self.c0 <= 0.0:
            raise ValueError("c0 must be positive when provided")
        if self.cell_length is not None and self.cell_length <= 0.0:
            raise ValueError("cell_length must be positive when provided")
        self.area = np.pi * self.radius**2

    @property
    def bending_stiffness(self) -> float:
        return self.E * self.h**3 / (12.0 * (1.0 - self.nu**2))

    @property
    def plate_mass(self) -> float:
        return self.rho_plate * self.h * self.area

    def approximate_plate_cuton_frequency(self) -> float:
        return (1.0 / (2.0 * np.pi)) * np.sqrt(384.0 * self.bending_stiffness / (5.0 * self.radius**4 * self.rho_plate * self.h))

    def trace_minus1(
        self,
        freq: np.ndarray | float,
        *,
        rho0: float | None = None,
        c0: float | None = None,
        cell_length: float | None = None,
    ) -> np.ndarray | float:
        rho0_use = self.rho0 if rho0 is None else float(rho0)
        c0_use = self.c0 if c0 is None else float(c0)
        length_use = self.cell_length if cell_length is None else float(cell_length)
        if rho0_use is None or c0_use is None or length_use is None:
            raise ValueError("rho0, c0, and cell_length must be provided either at construction or call time")
        freq_arr = np.asarray(freq, dtype=np.float64)
        wi = 2.0 * np.pi * freq_arr
        z_ref = rho0_use * c0_use / self.area
        ki = wi / c0_use
        zp = self.acoustic_series_impedance(np.atleast_1d(wi))
        trace_val = np.real(np.cos(ki * length_use) + 1j * zp * np.sin(ki * length_use) / (2.0 * z_ref)) - 1.0
        trace_val = np.asarray(trace_val)
        if trace_val.ndim == 0 or trace_val.size == 1:
            return float(trace_val.reshape(-1)[0])
        return trace_val

    def plate_cuton_frequency_numerical(
        self,
        *,
        rho0: float | None = None,
        c0: float | None = None,
        cell_length: float | None = None,
        bracket_factor: tuple[float, float] = (0.5, 2.0),
    ) -> float:
        f_ref = self.approximate_plate_cuton_frequency()
        f_low = float(bracket_factor[0]) * f_ref
        f_high = float(bracket_factor[1]) * f_ref
        if f_low <= 0.0 or f_high <= f_low:
            raise ValueError("bracket_factor must define an increasing positive bracket")
        root_fn = lambda ff: float(self.trace_minus1(float(ff), rho0=rho0, c0=c0, cell_length=cell_length))
        return float(brentq(root_fn, f_low, f_high))

    def acoustic_series_impedance(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega, dtype=np.float64).ravel()
        if np.any(omega <= 0.0):
            raise ValueError("omega must be strictly positive")

        mass = self.plate_mass
        k_p = np.sqrt(omega * np.sqrt(self.rho_plate * self.h / self.bending_stiffness))
        x = k_p * self.radius
        num = iv(1, x) * jv(0, x) + jv(1, x) * iv(0, x)
        den = iv(1, x) * jv(2, x) - jv(1, x) * iv(2, x)
        return -1j * omega * mass / (self.area**2) * (num / den)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        z_w = self.acoustic_series_impedance(omega)
        T = np.zeros((z_w.size, 2, 2), dtype=np.complex128)
        T[:, 0, 0] = 1.0
        T[:, 0, 1] = -np.real(z_w)+ 1j* np.imag(z_w)
        T[:, 1, 1] = 1.0
        return T
