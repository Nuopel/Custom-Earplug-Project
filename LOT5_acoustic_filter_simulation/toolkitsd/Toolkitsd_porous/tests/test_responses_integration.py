import numpy as np
import pytest

from toolkitsd.porous import BasePorousMaterial, JCAMaterial, MikiModel, PorousMediumProps
from toolkitsd.porous.responses import (
    diffuse_field_absorption,
    diffuse_field_absorption_discrete,
    diffuse_from_angles,
)


def _material() -> BasePorousMaterial:
    return JCAMaterial(
        sigma=20000,
        thickness=0.04,
        phi=0.9,
        lambda1=120e-6,
        lambdap=180e-6,
        tortu=1.05,
        rho0=1.21,
        c0=343.0,
    )


def test_diffuse_from_angles_matches_discrete_helper() -> None:
    freqs = np.array([250.0, 1000.0, 2000.0])
    props = PorousMediumProps.from_material(_material(), freqs, model=MikiModel())
    angles = np.linspace(0, 90, 13)

    discrete = diffuse_field_absorption_discrete(props, angles)
    via_helper = diffuse_from_angles(discrete.absorption_by_angle, angles)

    np.testing.assert_allclose(via_helper, discrete.diffuse_absorption)


def test_diffuse_field_absorption_validates_inputs() -> None:
    props = PorousMediumProps.from_material(_material(), np.array([500.0]), model=MikiModel())

    with pytest.raises(ValueError):
        diffuse_field_absorption(props, angle_upper_limit=10, angle_lower_limit=20)
    with pytest.raises(ValueError):
        diffuse_field_absorption(props, n_integration=1)
    with pytest.raises(ValueError):
        diffuse_field_absorption_discrete(props, [0.0], time_convention="jwt")


def test_quad_and_trapz_diffuse_methods_are_consistent() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy  # silence unused warning in case importorskip returns module

    freqs = np.array([500.0, 1000.0])
    props = PorousMediumProps.from_material(_material(), freqs, model=MikiModel())

    trapz_result = diffuse_field_absorption(
        props,
        angle_lower_limit=10,
        angle_upper_limit=80,
        n_integration=361,
        method="trapz",
    )
    quad_result = diffuse_field_absorption(
        props,
        angle_lower_limit=10,
        angle_upper_limit=80,
        n_integration=50,
        method="quad",
    )

    np.testing.assert_allclose(
        quad_result.diffuse_absorption,
        trapz_result.diffuse_absorption,
        rtol=1e-3,
        atol=1e-4,
    )
