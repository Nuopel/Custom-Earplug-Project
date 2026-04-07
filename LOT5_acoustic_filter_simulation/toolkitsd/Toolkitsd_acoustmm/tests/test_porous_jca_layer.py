import numpy as np

from toolkitsd.acoustmm import JCALayer, MikiLayer, RigidWall
from toolkitsd.porous import (
    JCAMaterial,
    JCAModel,
    MikiMaterial,
    MikiModel,
    PorousMediumProps,
    surface_response_on_rigid_backing,
)


def test_jca_layer_matrix_shape_and_finite_values():
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 120)
    omega = 2.0 * np.pi * freqs
    area = np.pi * (10e-3) ** 2
    layer = JCALayer(
        phi=0.93,
        sigma=10000,
        alpha_inf=1.1,
        lambda_v=60e-6,
        lambda_t=100e-6,
        length=0.04,
        area=area,
        rho0=1.213,
        c0=342.2,
    )
    T = layer.matrix(omega)
    assert T.shape == (freqs.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_jca_layer_rigid_backing_surface_matches_toolkitsd_porous():
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 220)
    omega = 2.0 * np.pi * freqs
    radius = 10e-3
    area = np.pi * radius**2

    layer = JCALayer(
        phi=0.93,
        sigma=10000,
        alpha_inf=1.1,
        lambda_v=60e-6,
        lambda_t=100e-6,
        length=0.04,
        area=area,
        rho0=1.213,
        c0=342.2,
    )
    z_tmm_vol = layer.Z_in(RigidWall().Z(omega), omega)
    z_tmm = z_tmm_vol * area  # convert volume-velocity impedance -> surface impedance

    material = JCAMaterial(
        rho0=1.213,
        c0=342.2,
        thickness=0.04,
        sigma=10000,
        phi=0.93,
        lambda1=60e-6,
        lambdap=100e-6,
        tortu=1.1,
        name="melamine_cttm",
    )
    pprops = PorousMediumProps.from_material(material, freqs, model=JCAModel())
    z_ref = surface_response_on_rigid_backing(pprops, incidence_angle_deg=90.0).surface_impedance[:, 0]

    rel = np.abs(z_tmm - z_ref) / np.maximum(np.abs(z_ref), 1e-12)
    assert float(np.nanmean(rel)) < 1e-2
    assert float(np.nanmax(rel)) < 5e-2


def test_miki_layer_matrix_shape_and_finite_values():
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 120)
    omega = 2.0 * np.pi * freqs
    area = np.pi * (10e-3) ** 2
    layer = MikiLayer(
        sigma=10000,
        length=0.04,
        area=area,
        rho0=1.213,
        c0=342.2,
    )
    T = layer.matrix(omega)
    assert T.shape == (freqs.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_miki_layer_uses_miki_material_without_jca_fields():
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 32)
    omega = 2.0 * np.pi * freqs
    area = np.pi * (10e-3) ** 2
    layer = MikiLayer(
        sigma=10000,
        length=0.04,
        area=area,
        rho0=1.213,
        c0=342.2,
    )

    assert layer.material.phi is None
    assert layer.material.lambda1 is None
    assert layer.material.lambdap is None
    assert layer.material.tortu is None
    T = layer.matrix(omega)
    assert T.shape == (freqs.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_miki_layer_rigid_backing_surface_matches_toolkitsd_porous():
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 220)
    omega = 2.0 * np.pi * freqs
    radius = 10e-3
    area = np.pi * radius**2

    layer = MikiLayer(
        sigma=10000,
        length=0.04,
        area=area,
        rho0=1.213,
        c0=342.2,
    )
    z_tmm_vol = layer.Z_in(RigidWall().Z(omega), omega)
    z_tmm = z_tmm_vol * area

    material = MikiMaterial(
        rho0=1.213,
        c0=342.2,
        thickness=0.04,
        sigma=10000,
        name="miki-equivalent",
    )
    pprops = PorousMediumProps.from_material(material, freqs, model=MikiModel())
    z_ref = surface_response_on_rigid_backing(pprops, incidence_angle_deg=90.0).surface_impedance[:, 0]

    rel = np.abs(z_tmm - z_ref) / np.maximum(np.abs(z_ref), 1e-12)
    assert float(np.nanmean(rel)) < 1e-2
    assert float(np.nanmax(rel)) < 5e-2
