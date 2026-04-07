"""
Equivalent-fluid porous models (Miki, JCA).

Classes here provide a common interface to compute characteristic impedance
and wavenumber, keeping the physical inputs and the surface responses
separated for clarity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .materials import AirProperties, BasePorousMaterial, DEFAULT_AIR
from .utils import as_frequency_array


@dataclass
class EquivalentFluidProperties:
    """Equivalent-fluid properties as a function of frequency.

    Attributes:
        impedance: Characteristic impedance of the porous medium (Pa·s/m).
        wavenumber: Complex wavenumber (rad/m).
        omega: Angular frequency (rad/s).
        rho_eff: Effective density (kg/m³).
        bulk_modulus: Effective bulk modulus (Pa).
        model: Identifier of the model used.
        metadata: Optional dictionary carrying provenance (e.g., material params).
    """

    impedance: NDArray[np.complex128]
    wavenumber: NDArray[np.complex128]
    omega: NDArray[np.float64]
    rho_eff: NDArray[np.complex128]
    bulk_modulus: NDArray[np.complex128]
    model: str
    metadata: dict | None = None


class EquivalentFluidModel(ABC):
    """Interface for porous equivalent-fluid models."""

    name: str

    @abstractmethod
    def properties(
        self,
        material: BasePorousMaterial,
        frequencies: ArrayLike,
        air: AirProperties = DEFAULT_AIR,
    ) -> EquivalentFluidProperties:
        """Return characteristic impedance and wavenumber."""


class MikiModel(EquivalentFluidModel):
    """Empirical Miki model."""

    name = "Miki"

    def properties(
        self,
        material: BasePorousMaterial,
        frequencies: ArrayLike,
        air: AirProperties = DEFAULT_AIR,
    ) -> EquivalentFluidProperties:
        material.validate_for_miki()
        f = as_frequency_array(frequencies)
        omega = 2 * np.pi * f
        ratio = material.rho0 * f / material.sigma
        impedance = material.rho0 * material.c0 * (
            1 + 0.0785 * ratio ** (-0.632) - 1j * 0.120 * ratio ** (-0.632)
        )
        wavenumber = omega / material.c0 * (
            1 + 0.122 * ratio ** (-0.618) - 1j * 0.180 * ratio ** (-0.618)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            rho_eff = impedance * (wavenumber / omega)
        bulk_modulus = impedance * (omega / wavenumber)

        metadata = {
            "model": self.name,
            "material": {
                **material.material_dict(),
            },
        }

        return EquivalentFluidProperties(
            impedance=impedance,
            wavenumber=wavenumber,
            omega=omega,
            rho_eff=rho_eff,
            bulk_modulus=bulk_modulus,
            model=self.name,
            metadata=metadata,
        )


class JCAModel(EquivalentFluidModel):
    """Champoux-Allard-Johnson equivalent-fluid model."""

    name = "JCA"

    def properties(
        self,
        material: BasePorousMaterial,
        frequencies: ArrayLike,
        air: AirProperties = DEFAULT_AIR,
    ) -> EquivalentFluidProperties:
        material.validate_for_jca()
        f = as_frequency_array(frequencies)
        omega = 2 * np.pi * f

        visc_term = np.sqrt(
            1
            + 4j
            * material.tortu**2
            * air.eta
            * material.rho0
            * omega
            / (material.sigma**2 * material.lambda1**2 * material.phi**2)
        )
        rho_eff = material.rho0 * material.tortu * (
            1
            + (material.sigma * material.phi)
            / (1j * material.tortu * material.rho0 * omega)
            * visc_term
        )

        therm_term = np.sqrt(
            1 + (1j * material.rho0 * material.lambdap**2 * air.prandtl * omega) / (16 * air.eta)
        )
        bulk_modulus = air.gamma * air.p0 / (
            air.gamma
            - (air.gamma - 1)
            / (1 + (8 * air.eta * therm_term) / (1j * material.lambdap**2 * material.rho0 * air.prandtl * omega))
        )

        wavenumber = omega * np.sqrt(rho_eff / bulk_modulus)
        impedance = np.sqrt(rho_eff * bulk_modulus)

        metadata = {
            "model": self.name,
            "material": {
                **material.material_dict(),
            },
        }

        return EquivalentFluidProperties(
            impedance=impedance,
            wavenumber=wavenumber,
            omega=omega,
            rho_eff=rho_eff,
            bulk_modulus=bulk_modulus,
            model=self.name,
            metadata=metadata,
        )


def compute_miki_properties(
    material: BasePorousMaterial, frequencies: ArrayLike, air: AirProperties = DEFAULT_AIR
) -> EquivalentFluidProperties:
    """Convenience wrapper around :class:`MikiModel`."""
    return MikiModel().properties(material, frequencies, air=air)


def compute_jca_properties(
    material: BasePorousMaterial, frequencies: ArrayLike, air: AirProperties = DEFAULT_AIR
) -> EquivalentFluidProperties:
    """Convenience wrapper around :class:`JCAModel`."""
    return JCAModel().properties(material, frequencies, air=air)
