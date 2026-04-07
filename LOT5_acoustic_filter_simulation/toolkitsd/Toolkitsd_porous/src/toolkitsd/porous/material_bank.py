"""Named porous-material presets for quick reuse in examples and studies."""

from __future__ import annotations

from dataclasses import dataclass

from .materials import BasePorousMaterial, JCAMaterial


@dataclass(frozen=True)
class MaterialPreset:
    """Serializable preset definition for PorousMaterial creation."""

    sigma: float
    phi: float
    lambda1: float
    lambdap: float
    tortu: float
    thickness: float
    name: str


_PRESETS: dict[str, MaterialPreset] = {
    "melamine_cttm": MaterialPreset(
        sigma=10000, phi=0.93, lambda1=60e-6, lambdap=100e-6, tortu=1.1, thickness=0.04, name="Melamine CTTM"
    ),
    "melamine_thesis": MaterialPreset(
        sigma=1500, phi=0.97, lambda1=300e-6, lambdap=170e-6, tortu=1.01, thickness=0.04, name="Melamine Thesis"
    ),
    "melamine_thesis_cttm": MaterialPreset(
        sigma=11533, phi=0.998, lambda1=124e-6, lambdap=183e-6, tortu=1.005, thickness=0.04, name="Melamine Thesis CTTM"
    ),
    "mousse_materiau1_cttm": MaterialPreset(
        sigma=17949, phi=0.94, lambda1=58e-6, lambdap=185e-6, tortu=1.06, thickness=0.08, name="Mousse Materiau 1 CTTM"
    ),
    "mousse_materiau3_cttm": MaterialPreset(
        sigma=3320, phi=0.97, lambda1=163e-6, lambdap=255e-6, tortu=1.06, thickness=0.05, name="Mousse Materiau 3 CTTM"
    ),
    "mousse_materiau4_cttm": MaterialPreset(
        sigma=10605, phi=0.956, lambda1=71e-6, lambdap=317e-6, tortu=1.67, thickness=0.0505, name="Mousse Materiau 4 CTTM"
    ),
    "laine_de_verre_cttm": MaterialPreset(
        sigma=13112, phi=0.954, lambda1=55.6e-6, lambdap=116e-6, tortu=1.0, thickness=0.052, name="Laine de Verre CTTM"
    ),
    "melamine_mathis_ets": MaterialPreset(
        sigma=8644, phi=0.971, lambda1=123.6e-6, lambdap=168e-6, tortu=1.02, thickness=0.0508, name="Melamine Mathis ETS"
    ),
    "lainederoche_mathis_ets": MaterialPreset(
        sigma=9105, phi=0.963, lambda1=124e-6, lambdap=175e-6, tortu=1.01, thickness=0.0761, name="Laine de Roche Mathis ETS"
    ),
}


def list_material_presets() -> list[str]:
    """Return available material preset keys."""
    return sorted(_PRESETS.keys())


def get_material_preset(name: str, *, rho0: float = 1.213, c0: float = 342.0) -> BasePorousMaterial:
    """Instantiate a PorousMaterial from a named preset."""
    key = name.strip().lower()
    if key not in _PRESETS:
        available = ", ".join(list_material_presets())
        raise KeyError(f"Unknown material preset '{name}'. Available: {available}")
    p = _PRESETS[key]
    return JCAMaterial(
        sigma=p.sigma,
        thickness=p.thickness,
        phi=p.phi,
        lambda1=p.lambda1,
        lambdap=p.lambdap,
        tortu=p.tortu,
        rho0=rho0,
        c0=c0,
        name=p.name,
    )
