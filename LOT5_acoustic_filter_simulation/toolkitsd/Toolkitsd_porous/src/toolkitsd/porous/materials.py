"""
Data structures describing porous materials and ambient air.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from .material_bank import MaterialPreset  # pragma: no cover


@dataclass
class AirProperties:
    """Thermo-viscous properties of ambient air.

    Attributes:
        gamma: Specific heat ratio (-).
        eta: Dynamic viscosity (kg / (m·s)).
        prandtl: Prandtl number (-).
        p0: Ambient static pressure (Pa).
    """

    gamma: float = 1.4
    eta: float = 0.184e-4
    prandtl: float = 0.71
    p0: float = 0.10132e6


DEFAULT_AIR = AirProperties()


@dataclass
class BasePorousMaterial:
    """Shared porous-material fields across empirical and equivalent-fluid models."""

    sigma: float
    thickness: float
    rho0: float = 1.213
    c0: float = 342.2
    name: Optional[str] = None
    phi: float | None = None
    lambda1: float | None = None
    lambdap: float | None = None
    tortu: float | None = None

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.thickness <= 0:
            raise ValueError("thickness must be positive")
        if self.rho0 <= 0 or self.c0 <= 0:
            raise ValueError("rho0 and c0 must be positive")

    @property
    def z0(self) -> float:
        """Characteristic impedance of air."""
        return self.rho0 * self.c0

    def validate_for_miki(self) -> None:
        """Validate the subset of properties required by Miki."""

    def validate_for_jca(self) -> None:
        """Validate the full JCA parameter set."""
        if self.phi is None or not 0 < self.phi <= 1:
            raise ValueError("phi must be in (0, 1]")
        if self.lambda1 is None or self.lambda1 <= 0:
            raise ValueError("lambda1 must be positive")
        if self.lambdap is None or self.lambdap <= 0:
            raise ValueError("lambdap must be positive")
        if self.tortu is None or self.tortu <= 0:
            raise ValueError("tortu must be positive")

    def material_dict(self) -> dict[str, float | None]:
        """Serializable material payload for metadata/reporting."""
        return {
            "sigma": self.sigma,
            "thickness": self.thickness,
            "phi": self.phi,
            "lambda1": self.lambda1,
            "lambdap": self.lambdap,
            "tortu": self.tortu,
            "rho0": self.rho0,
            "c0": self.c0,
            "z0": self.z0,
        }

    def __str__(self) -> str:
        label = self.name if self.name is not None else "unnamed"
        parts = [
            f"name={label}",
            f"sigma={self.sigma:.6g}",
            f"thickness={self.thickness:.6g}",
        ]
        if self.phi is not None:
            parts.append(f"phi={self.phi:.6g}")
        if self.lambda1 is not None:
            parts.append(f"lambda1={self.lambda1:.6g}")
        if self.lambdap is not None:
            parts.append(f"lambdap={self.lambdap:.6g}")
        if self.tortu is not None:
            parts.append(f"tortu={self.tortu:.6g}")
        parts.extend([f"rho0={self.rho0:.6g}", f"c0={self.c0:.6g}", f"z0={self.z0:.6g}"])
        return f"{self.__class__.__name__}(" + ", ".join(parts) + ")"


@dataclass
class MikiMaterial(BasePorousMaterial):
    """Material definition for the Miki empirical model."""

    @classmethod
    def from_sigma(
        cls,
        *,
        sigma: float,
        thickness: float,
        rho0: float = 1.213,
        c0: float = 342.2,
        name: str | None = None,
    ) -> "MikiMaterial":
        return cls(sigma=sigma, thickness=thickness, rho0=rho0, c0=c0, name=name)


@dataclass
class JCAMaterial(BasePorousMaterial):
    """Material definition for the JCA equivalent-fluid model."""

    phi: float
    lambda1: float
    lambdap: float
    tortu: float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.validate_for_jca()

    @classmethod
    def from_parameters(
        cls,
        *,
        sigma: float,
        thickness: float,
        phi: float,
        lambda1: float,
        lambdap: float,
        tortu: float,
        rho0: float = 1.213,
        c0: float = 342.2,
        name: str | None = None,
    ) -> "JCAMaterial":
        return cls(
            sigma=sigma,
            thickness=thickness,
            phi=phi,
            lambda1=lambda1,
            lambdap=lambdap,
            tortu=tortu,
            rho0=rho0,
            c0=c0,
            name=name,
        )


class _PorousMaterialMeta(type):
    """Compatibility constructor for the legacy PorousMaterial API."""

    def __call__(cls, *args, **kwargs):  # type: ignore[override]
        warnings.warn(
            "PorousMaterial(...) is deprecated; use MikiMaterial or JCAMaterial instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if args:
            raise TypeError("PorousMaterial(...) only supports keyword arguments")
        has_jca_values = {key: kwargs.get(key) is not None for key in ("phi", "lambda1", "lambdap", "tortu")}
        if not any(has_jca_values.values()):
            return MikiMaterial(**kwargs)
        if all(has_jca_values.values()):
            return JCAMaterial(**kwargs)
        raise ValueError("Incomplete JCA material definition; provide phi, lambda1, lambdap, and tortu together")

    def __instancecheck__(cls, instance):  # type: ignore[override]
        return isinstance(instance, BasePorousMaterial)


class PorousMaterial(metaclass=_PorousMaterialMeta):
    """Deprecated compatibility shim for legacy porous-material construction."""

    @classmethod
    def get_material_preset(cls, name: str, *, rho0: float = 1.213, c0: float = 342.0) -> BasePorousMaterial:
        """Convenience wrapper around the central porous material preset bank."""
        from .material_bank import get_material_preset

        return get_material_preset(name, rho0=rho0, c0=c0)

