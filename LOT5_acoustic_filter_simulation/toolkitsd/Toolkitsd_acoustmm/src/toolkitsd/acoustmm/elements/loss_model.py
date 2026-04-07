"""Viscothermal loss models for waveguide elements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import jve


class CircularLossModel(Protocol):
    """Constitutive model returning propagation constant and characteristic impedance."""

    @property
    def gamma(self) -> float: ...

    @property
    def zc_lossless(self) -> float: ...

    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class RectangularLossModel(Protocol):
    """Constitutive model for rectangular ducts returning propagation constant and characteristic impedance."""

    @property
    def gamma(self) -> float: ...

    @property
    def zc_lossless(self) -> float: ...

    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class CircularLossModelBase(ABC):
    """Shared thermodynamic state for circular-duct loss models."""

    radius: float
    area: float
    c0: float
    rho0: float
    P0: float = 101325.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.area <= 0.0:
            raise ValueError("area must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0 or self.P0 <= 0.0:
            raise ValueError("c0, rho0, and P0 must be positive")
        if self.gamma <= 1.0:
            raise ValueError("derived gamma must be > 1")

    @property
    def gamma(self) -> float:
        return self.rho0 * self.c0**2 / self.P0

    @property
    def K0(self) -> float:
        return self.rho0 * self.c0**2

    @property
    def zc_lossless(self) -> float:
        return self.rho0 * self.c0 / self.area

    @abstractmethod
    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True)
class RectangularLossModelBase(ABC):
    """Shared thermodynamic state for rectangular-duct loss models."""

    width: float
    height: float
    area: float
    c0: float
    rho0: float
    P0: float = 101325.0

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("width and height must be positive")
        if self.area <= 0.0:
            raise ValueError("area must be positive")
        if self.c0 <= 0.0 or self.rho0 <= 0.0 or self.P0 <= 0.0:
            raise ValueError("c0, rho0, and P0 must be positive")
        if self.gamma <= 1.0:
            raise ValueError("derived gamma must be > 1")

    @property
    def gamma(self) -> float:
        return self.rho0 * self.c0**2 / self.P0

    @property
    def K0(self) -> float:
        return self.rho0 * self.c0**2

    @property
    def zc_lossless(self) -> float:
        return self.rho0 * self.c0 / self.area

    @abstractmethod
    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass(frozen=True)
class KirchhoffStinsonEquivalentFluidModelRectangular(RectangularLossModelBase):
    """Kirchhoff/Stinson equivalent-fluid model for rectangular ducts using double modal sums."""

    eta0: float = 1.839e-5
    Pr: float = 0.71
    n_modes: int = 24

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eta0 <= 0.0 or self.Pr <= 0.0:
            raise ValueError("eta0 and Pr must be positive")
        if self.n_modes < 1:
            raise ValueError("n_modes must be >= 1")

    def equivalent_fluid_properties(
        self, omega: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        k_idx = np.arange(self.n_modes, dtype=np.float64)
        m_idx = np.arange(self.n_modes, dtype=np.float64)
        alpha = (k_idx + 0.5) * np.pi / self.width
        beta = (m_idx + 0.5) * np.pi / self.height
        alpha2 = alpha[:, None] ** 2
        beta2 = beta[None, :] ** 2
        modal = alpha2 + beta2

        g_rho = np.lib.scimath.sqrt(-1j * omega * self.rho0 / self.eta0)
        g_kappa = np.lib.scimath.sqrt(-1j * omega * self.Pr * self.rho0 / self.eta0)

        denom_rho = alpha2[None, :, :] * beta2[None, :, :] * (modal[None, :, :] - g_rho[:, None, None] ** 2)
        sum_rho = np.sum(1.0 / denom_rho, axis=(1, 2))
        rho_eff = -(self.rho0 * self.width**2 * self.height**2) / (4.0 * g_rho**2 * sum_rho)

        denom_kappa = alpha2[None, :, :] * beta2[None, :, :] * (modal[None, :, :] - g_kappa[:, None, None] ** 2)
        sum_kappa = np.sum(1.0 / denom_kappa, axis=(1, 2))
        K_eff = self.K0 / (
            self.gamma + (4.0 * (self.gamma - 1.0) * g_kappa**2 / (self.width**2 * self.height**2)) * sum_kappa
        )

        k_eff = omega * np.lib.scimath.sqrt(rho_eff / K_eff)
        zc_eff = np.lib.scimath.sqrt(rho_eff * K_eff) / self.area
        return rho_eff, K_eff, k_eff, zc_eff

    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, k_eff, zc_eff = self.equivalent_fluid_properties(omega)
        gamma_eff = 1j * k_eff
        gamma_eff = np.where(np.real(gamma_eff) < 0.0, -gamma_eff, gamma_eff)
        zc_eff = np.where(np.real(zc_eff) < 0.0, -zc_eff, zc_eff)
        return gamma_eff, zc_eff


@dataclass(frozen=True)
class LosslessCircularModel(CircularLossModelBase):
    """Lossless circular duct constitutive model."""

    def equivalent_fluid_properties(
        self, omega: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        rho_eff = np.full_like(omega, self.rho0, dtype=np.complex128)
        K_eff = np.full_like(omega, self.K0, dtype=np.complex128)
        k_eff = omega / self.c0
        zc_eff = np.full(omega.shape, self.zc_lossless + 0j, dtype=np.complex128)
        return rho_eff, K_eff, k_eff, zc_eff

    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, k_eff, zc_eff = self.equivalent_fluid_properties(omega)
        gamma = 1j * k_eff
        return gamma, zc_eff


@dataclass(frozen=True)
class KirchhoffStinsonEquivalentFluidModel(CircularLossModelBase):
    """Kirchhoff/Stinson equivalent-fluid circular loss model."""

    eta0: float = 1.839e-5
    Pr: float = 0.71

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.eta0 <= 0.0 or self.Pr <= 0.0:
            raise ValueError("eta0 and Pr must be positive")

    def equivalent_fluid_properties(
        self, omega: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        if np.any(np.real(omega) <= 0.0):
            raise ValueError("omega must have strictly positive real part")

        G_r = np.lib.scimath.sqrt(-1j * omega * self.rho0 / self.eta0)
        G_k = np.lib.scimath.sqrt(-1j * omega * self.Pr * self.rho0 / self.eta0)

        x_r = self.radius * G_r
        x_k = self.radius * G_k

        rho_ratio = 2.0 * jve(1, x_r) / (x_r * jve(0, x_r))
        bulk_ratio = 2.0 * jve(1, x_k) / (x_k * jve(0, x_k))

        rho_eff = self.rho0 * (1.0 - rho_ratio) ** (-1)
        K_eff = self.K0 * (1.0 + (self.gamma - 1.0) * bulk_ratio) ** (-1)
        k_eff = omega * np.lib.scimath.sqrt(rho_eff / K_eff)
        zc_eff = np.lib.scimath.sqrt(rho_eff * K_eff) / self.area
        return rho_eff, K_eff, k_eff, zc_eff

    def gamma_zc(self, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _, _, k_eff, zc_eff = self.equivalent_fluid_properties(omega)
        gamma_eff = 1j * k_eff
        gamma_eff = np.where(np.real(gamma_eff) < 0.0, -gamma_eff, gamma_eff)
        zc_eff = np.where(np.real(zc_eff) < 0.0, -zc_eff, zc_eff)
        return gamma_eff, zc_eff


if __name__ == "__main__":
    radius = 2.5e-3
    area = np.pi * radius**2
    c0 = 343.0
    rho0 = 1.213
    freqs = np.linspace(100.0, 4000.0, 200)
    omega = 2.0 * np.pi * freqs

    models: dict[str, CircularLossModel] = {
        "lossless": LosslessCircularModel(radius=radius, area=area, c0=c0, rho0=rho0),
        "ks_equiv": KirchhoffStinsonEquivalentFluidModel(radius=radius, area=area, c0=c0, rho0=rho0),
    }

    gamma_data: dict[str, np.ndarray] = {}
    zc_data: dict[str, np.ndarray] = {}
    for name, model in models.items():
        gamma_data[name], zc_data[name] = model.gamma_zc(omega)

    print("Max abs differences against ks_equiv:")
    gamma_err = np.max(np.abs(gamma_data["lossless"] - gamma_data["ks_equiv"]))
    zc_err = np.max(np.abs(zc_data["lossless"] - zc_data["ks_equiv"]))
    print(f"  lossless: max|dgamma|={gamma_err:.6e}, max|dZc|={zc_err:.6e}")

    model = models["ks_equiv"]
    _, _, k_eff, zc_eff = model.equivalent_fluid_properties(omega)
    gamma_from_k = 1j * k_eff
    gamma_from_k = np.where(np.real(gamma_from_k) < 0.0, -gamma_from_k, gamma_from_k)
    zc_eff = np.where(np.real(zc_eff) < 0.0, -zc_eff, zc_eff)
    print(
        f"  ks_equiv: max|gamma - i*k|={np.max(np.abs(gamma_data['ks_equiv'] - gamma_from_k)):.6e}, "
        f"max|Zc - Zc_eff|={np.max(np.abs(zc_data['ks_equiv'] - zc_eff)):.6e}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        for name in models:
            axes[0, 0].plot(freqs, np.real(gamma_data[name]), label=name)
            axes[0, 1].plot(freqs, np.imag(gamma_data[name]), label=name)
            axes[1, 0].plot(freqs, np.real(zc_data[name]), label=name)
            axes[1, 1].plot(freqs, np.imag(zc_data[name]), label=name)

        axes[0, 0].set_ylabel("Re(gamma)")
        axes[0, 1].set_ylabel("Im(gamma)")
        axes[1, 0].set_ylabel("Re(Zc)")
        axes[1, 1].set_ylabel("Im(Zc)")
        axes[1, 0].set_xlabel("Frequency [Hz]")
        axes[1, 1].set_xlabel("Frequency [Hz]")
        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend()
        fig.tight_layout()
        plt.show()
