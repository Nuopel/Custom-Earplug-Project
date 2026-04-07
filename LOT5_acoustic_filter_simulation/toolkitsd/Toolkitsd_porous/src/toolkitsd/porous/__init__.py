"""Porous material acoustic models (JCA, Miki)."""

from .materials import AirProperties, BasePorousMaterial, DEFAULT_AIR, JCAMaterial, MikiMaterial, PorousMaterial
from .material_bank import get_material_preset, list_material_presets
from .models import (
    EquivalentFluidModel,
    EquivalentFluidProperties,
    JCAModel,
    MikiModel,
    compute_jca_properties,
    compute_miki_properties,
)
from .medium import PorousMediumProps, build_porous_medium_props
from .responses import (
    DiffuseFieldResult,
    SurfaceResponse,
    diffuse_field_absorption,
    diffuse_field_absorption_discrete,
    surface_response_on_rigid_backing,
)
from .measurement import (
    BaseMeasurement,
    TwoMicPlaneWave,
    OneMicPlaneWave,
    TwoMicSpherical,
    two_mic_plane_wave_pv,
    two_mic_plane_wave_transfer,
    one_mic_plane_wave,
    two_mic_spherical_wave_transfer,
)


__all__ = [
    "AirProperties",
    "BasePorousMaterial",
    "DEFAULT_AIR",
    "EquivalentFluidModel",
    "EquivalentFluidProperties",
    "JCAModel",
    "JCAMaterial",
    "MikiModel",
    "MikiMaterial",
    "PorousMaterial",
    "list_material_presets",
    "get_material_preset",
    "DiffuseFieldResult",
    "SurfaceResponse",
    "PorousMediumProps",
    "build_porous_medium_props",
    "compute_jca_properties",
    "compute_miki_properties",
    "diffuse_field_absorption",
    "diffuse_field_absorption_discrete",
    "surface_response_on_rigid_backing",
    "BaseMeasurement",
    "TwoMicPlaneWave",
    "OneMicPlaneWave",
    "TwoMicSpherical",
    "two_mic_plane_wave_pv",
    "two_mic_plane_wave_transfer",
    "one_mic_plane_wave",
    "two_mic_spherical_wave_transfer",
]
