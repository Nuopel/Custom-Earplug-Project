"""
Surface responses for porous layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .models import EquivalentFluidProperties
from .medium import PorousMediumProps
from .utils import as_angle_array, column_vector

# Compat wrapper across NumPy versions without eager fallback evaluation.
if hasattr(np, "trapezoid"):
    _TRAPZ = np.trapezoid
elif hasattr(np, "trapz"):
    _TRAPZ = np.trapz
else:
    raise AttributeError("NumPy has neither 'trapezoid' nor 'trapz'.")

@dataclass
class SurfaceResponse:
    """Surface impedance and absorption results."""

    surface_impedance: np.ndarray
    reflection_coefficient: np.ndarray
    absorption: np.ndarray


@dataclass
class DiffuseFieldResult:
    """Diffuse-field absorption bundled with the angle-dependent data used."""

    diffuse_absorption: np.ndarray
    absorption_by_angle: np.ndarray
    angles_deg: np.ndarray  # plane-referenced angles (0° grazing, 90° normal)


def surface_response_on_rigid_backing(
    porous_props: PorousMediumProps | None = None,
    incidence_angle_deg: ArrayLike = 90.0,
    time_convention: str | None = None,
) -> SurfaceResponse:
    """Compute surface impedance, reflection, and absorption on a rigid wall.

    Args:
        porous_props: Convenience bundle carrying both material and properties.
        incidence_angle_deg: Scalar or array of incidence angles measured from the
            reflecting plane (0° = grazing, 90° = normal). Values outside [0, 90]
            are clipped to that range.
        time_convention: Optional time convention flag. If set to ``"neg_jwt"``,
            the result is conjugated to match a +jωt convention (legacy code
            path). Any other value leaves the default -jωt convention.

    Returns:
        SurfaceResponse with shape (n_freqs, n_angles).
    """
    if porous_props is None:
        raise ValueError("porous_props (material + properties) must be provided")
    material = porous_props.material
    properties = porous_props.props
    phi_plane = as_angle_array(incidence_angle_deg)  # 0° = grazing, 90° = normal
    phi = np.deg2rad(phi_plane).reshape(1, -1)

    omega = column_vector(properties.omega)
    k_porous = column_vector(properties.wavenumber)
    z_porous = column_vector(properties.impedance)

    # Plane-referenced angles: grazing->normal (0->90). In terms of the normal:
    # theta_normal = 90 - phi_plane, so:
    # sin(theta_normal) = cos(phi_plane), cos(theta_normal) = sin(phi_plane)
    sin_theta_normal = np.cos(phi)
    cos_theta_normal = np.sin(phi)

    # Snell in terms of plane angle: tangential component continuity still uses sin(normal),
    # so we use sin_theta_normal computed from phi.
    costheta_porous = np.sqrt(1 - (omega / (material.c0 * k_porous)) ** 2 * (sin_theta_normal**2) + 0j)

    porous_phi = 1.0 if material.phi is None else material.phi
    surface_impedance = -1j * z_porous / (porous_phi * costheta_porous) / np.tan(
        k_porous * material.thickness * costheta_porous
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        reflection = (surface_impedance * cos_theta_normal - material.z0) / (
            surface_impedance * cos_theta_normal + material.z0
        )
    absorption = 1 - np.abs(reflection) ** 2

    # Legacy (+jωt) convention: conjugate the result to flip the phase sign.
    if time_convention == "neg_jwt":
        surface_impedance = np.conj(surface_impedance)
        reflection = np.conj(reflection)

    return SurfaceResponse(
        surface_impedance=surface_impedance,
        reflection_coefficient=reflection,
        absorption=absorption,
    )


def diffuse_field_absorption_discrete(
    porous_props: PorousMediumProps,
    incidence_angles_deg: ArrayLike,
    time_convention: str | None = None,
) -> DiffuseFieldResult:
    """Approximate diffuse-field absorption using a discrete set of incidences.

    This implements the trapezoidal integration used in the legacy example,
    weighting each angle by ``sin(2*phi)`` where ``phi`` is measured from the
    normal (0° = normal incidence, 90° = grazing). Input angles are elevations
    from the plane (0° = grazing, 90° = normal).
    """
    angles_plane = as_angle_array(incidence_angles_deg)  # 0° grazing, 90° normal
    if angles_plane.ndim != 1:
        angles_plane = angles_plane.ravel()
    angles_rad = np.deg2rad(angles_plane)

    response = surface_response_on_rigid_backing(
        porous_props=porous_props, incidence_angle_deg=angles_plane, time_convention=time_convention
    )
    # sin(2*theta_normal) = sin(2*(90-phi_plane)) = sin(2*phi_plane)
    weights = np.sin(2 * angles_rad)
    denominator = _TRAPZ(weights, x=angles_rad)
    if np.isclose(denominator, 0):
        raise ValueError("Angle range is too small to integrate diffuse absorption.")

    diffuse_absorption = _TRAPZ(response.absorption * weights, x=angles_rad, axis=1) / denominator
    return DiffuseFieldResult(
        diffuse_absorption=diffuse_absorption,
        absorption_by_angle=response.absorption,
        angles_deg=angles_plane,
    )


def diffuse_field_absorption(
    porous_props: PorousMediumProps,
    angle_upper_limit: float = 90.0,
    angle_lower_limit: float = 0.0,
    n_integration: int = 720,
    time_convention: str | None = None,
    method: str = "trapz",
) -> DiffuseFieldResult:
    """Diffuse-field absorption via dense sampling or continuous integration.

    Args:
        porous_props: Porous material properties.
        angle_upper_limit: Maximum elevation **from the plane** (0° = grazing, 90° = normal).
        angle_lower_limit: Lower bound for the elevation sweep (from the plane).
        n_integration: Number of angular samples (>= 2).
        time_convention: Forward/backward time convention (passed through).
        method: "trapz" (default, vectorized trapezoid) or "quad" (SciPy ``quad_vec``).
    """
    if angle_upper_limit <= angle_lower_limit:
        raise ValueError(f"angle_upper_limit {angle_upper_limit} must be greater than angle_lower_limit {angle_lower_limit}.")


    if method == "quad":
        from scipy import integrate

        lower_rad = np.deg2rad(angle_lower_limit)
        upper_rad = np.deg2rad(angle_upper_limit)

        def _integrand(phi: float) -> np.ndarray:
            # phi is in radians, plane-referenced
            resp = surface_response_on_rigid_backing(
                porous_props=porous_props,
                incidence_angle_deg=np.rad2deg(phi),
                time_convention=time_convention,
            )
            return resp.absorption.squeeze() * np.sin(2 * phi)

        numerator = integrate.quad_vec(_integrand, lower_rad, upper_rad)[0]
        denominator = float(integrate.quad(lambda phi: np.sin(2 * phi), lower_rad, upper_rad)[0])
        if np.isclose(denominator, 0):
            raise ValueError("Angle range is too small to integrate diffuse absorption.")

        diffuse_absorption = numerator / denominator
        angles_deg_dense = np.linspace(angle_lower_limit, angle_upper_limit, n_integration)
        dense_response = surface_response_on_rigid_backing(
            porous_props=porous_props,
            incidence_angle_deg=angles_deg_dense,
            time_convention=time_convention,
        )
        return DiffuseFieldResult(
            diffuse_absorption=diffuse_absorption,
            absorption_by_angle=dense_response.absorption,
            angles_deg=angles_deg_dense,
        )

    if n_integration < 2:
        raise ValueError("n_integration must be at least 2 to integrate over angles.")
    angles_deg = np.linspace(angle_lower_limit, angle_upper_limit, n_integration)
    return diffuse_field_absorption_discrete(
        porous_props, angles_deg, time_convention=time_convention
    )

def diffuse_from_angles(absorption: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Discrete diffuse coefficient from angle-dependent absorption."""
    phi_rad = np.deg2rad(angles_deg.reshape(-1))
    weights = np.sin(2 * phi_rad)
    denom = _TRAPZ(weights, x=phi_rad)
    return _TRAPZ(absorption * weights, x=phi_rad, axis=1) / denom
