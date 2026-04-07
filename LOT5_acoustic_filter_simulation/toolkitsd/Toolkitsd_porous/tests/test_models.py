import numpy as np
import pytest

from toolkitsd.porous import (
    BasePorousMaterial,
    JCAMaterial,
    PorousMediumProps,
    JCAModel,
    MikiModel,
    compute_jca_properties,
    compute_miki_properties,
    surface_response_on_rigid_backing,
)


def _default_material() -> BasePorousMaterial:
    return JCAMaterial(
        rho0=1.213,
        c0=342.2,
        thickness=0.04,
        sigma=50000,
        phi=0.93,
        lambda1=60e-6,
        lambdap=100e-6,
        tortu=1.1,
    )


def test_frequency_validation() -> None:
    with pytest.raises(ValueError):
        compute_miki_properties(_default_material(), [-10, 0, 100])


def test_miki_and_jca_match_reference_values() -> None:
    freqs = np.array([1000.0])
    material = _default_material()

    miki_props = PorousMediumProps(
        material=material,
        props=compute_miki_properties(material, freqs),
        model=MikiModel(),
    )
    jca_props = PorousMediumProps(
        material=material,
        props=compute_jca_properties(material, freqs),
        model=JCAModel(),
    )

    miki_surface = surface_response_on_rigid_backing(miki_props)
    jca_surface = surface_response_on_rigid_backing(jca_props)

    # Reference values generated from the legacy script at 1 kHz
    assert np.isclose(miki_surface.surface_impedance[0, 0].real, 713.0790234726943, rtol=1e-6)
    assert np.isclose(miki_surface.surface_impedance[0, 0].imag, -475.4751865845184, rtol=1e-6)
    assert np.isclose(jca_surface.surface_impedance[0, 0].real, 692.6026058252513, rtol=1e-6)
    assert np.isclose(jca_surface.surface_impedance[0, 0].imag, -496.6946945010381, rtol=1e-6)

    assert np.isclose(miki_surface.absorption[0, 0], 0.7899207458488177, rtol=1e-6)
    assert np.isclose(jca_surface.absorption[0, 0], 0.7803332807382899, rtol=1e-6)


def test_angle_vector_support() -> None:
    freqs = np.array([500.0, 1000.0])
    angles = [0.0, 30.0]
    material = _default_material()

    props = PorousMediumProps(
        material=material,
        props=compute_miki_properties(material, freqs),
        model=MikiModel(),
    )
    response = surface_response_on_rigid_backing(props, angles)

    assert response.surface_impedance.shape == (2, 2)
    assert response.absorption.shape == (2, 2)


def test_surface_response_requires_props() -> None:
    with pytest.raises(ValueError):
        surface_response_on_rigid_backing(None)


def test_surface_response_time_convention_neg_jwt_conjugates() -> None:
    freqs = np.array([1000.0])
    material = _default_material()
    props = PorousMediumProps(
        material=material,
        props=compute_miki_properties(material, freqs),
        model=MikiModel(),
    )

    default_resp = surface_response_on_rigid_backing(props)
    neg_jwt_resp = surface_response_on_rigid_backing(props, time_convention="neg_jwt")

    assert np.allclose(neg_jwt_resp.surface_impedance, np.conj(default_resp.surface_impedance))
    assert np.allclose(neg_jwt_resp.reflection_coefficient, np.conj(default_resp.reflection_coefficient))
