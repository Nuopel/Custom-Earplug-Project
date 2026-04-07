from __future__ import annotations

from dataclasses import dataclass

from .materials import BasePorousMaterial
from .models import EquivalentFluidProperties, EquivalentFluidModel, JCAModel
import numpy as np

@dataclass(frozen=True)
class PorousMediumProps:
    material: BasePorousMaterial
    props: EquivalentFluidProperties
    model: EquivalentFluidModel  # actual model instance

    @staticmethod
    def from_material(material: BasePorousMaterial, frequencies,
                      model: EquivalentFluidModel | None = None) -> "PorousMediumProps":
        model = model or JCAModel()
        props = model.properties(material, frequencies)
        return PorousMediumProps(material=material, props=props, model=model)

    def update_frequencies(self, frequencies) -> "PorousMediumProps":
        props = self.model.properties(self.material, frequencies)
        return PorousMediumProps(material=self.material, props=props, model=self.model)

    # Convenience wrappers to mirror the functional API
    def diffuse_field_absorption(self, *args, **kwargs):
        """Compute diffuse-field absorption (plane-referenced angles, 0° grazing, 90° normal)."""
        from .responses import diffuse_field_absorption  # local import to avoid circular dependency

        return diffuse_field_absorption(self, *args, **kwargs)

    def diffuse_field_absorption_discrete(self, *args, **kwargs):
        """Discrete-angle diffuse-field absorption helper."""
        from .responses import diffuse_field_absorption_discrete  # local import to avoid circular dependency

        return diffuse_field_absorption_discrete(self, *args, **kwargs)

    def surface_response_on_rigid_backing(self, *args, **kwargs):
        """Convenience wrapper for surface response (plane angles: 0° grazing, 90° normal)."""
        from .responses import surface_response_on_rigid_backing  # local import to avoid circular dependency

        return surface_response_on_rigid_backing(porous_props=self, *args, **kwargs)

    @staticmethod
    def diffuse_from_angles(absorption: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
        """Discrete diffuse coefficient from angle-dependent absorption helper."""
        from .responses import diffuse_from_angles# local import to avoid circular dependency
        return diffuse_from_angles(absorption, angles_deg)


def build_porous_medium_props(
    material: BasePorousMaterial, frequencies, model: EquivalentFluidModel | None = None
) -> PorousMediumProps:
    """Convenience wrapper to build PorousMediumProps from a material and model."""
    return PorousMediumProps.from_material(material, frequencies, model=model)

