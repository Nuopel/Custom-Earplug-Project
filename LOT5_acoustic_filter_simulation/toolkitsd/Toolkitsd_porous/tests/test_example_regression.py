import numpy as np

import pytest

try:
    from toolkitsd.acoustic.pressure_gen import AcousticParameters, PorousGroundPlaneWave
    from toolkitsd.porous import (
        JCAMaterial,
        JCAModel,
        OneMicPlaneWave,
        PorousMediumProps,
        TwoMicPlaneWave,
        surface_response_on_rigid_backing,
    )
    from toolkitsd.array.array import Array
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    pytest.skip(f"Optional dependency missing: {exc}", allow_module_level=True)


def _material() -> JCAMaterial:
    return JCAMaterial(
        sigma=11533,
        thickness=0.05,
        phi=0.998,
        lambda1=124e-6,
        lambdap=183e-6,
        tortu=1.005,
        rho0=1.2,
        c0=340.0,
    )


def _doublet(offset_z: float, spacing: float) -> Array:
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [offset_z, offset_z + spacing]])
    return Array.from_coordinates(coords)


def _single_mic(offset_z: float) -> Array:
    coords = np.array([[0.0], [0.0], [offset_z]])
    return Array.from_coordinates(coords)


def test_doublet_and_one_mic_match_surface_response() -> None:
    freqs = np.array([100.0, 500.0, 1000.0, 2000.0])
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    material = _material()
    jca = JCAModel()

    elevation_deg = 85.0
    azimuth_deg = 0.0
    mic_offset = 0.01
    spacing = 0.01

    doublet = _doublet(offset_z=mic_offset, spacing=spacing)
    single = _single_mic(offset_z=mic_offset)

    props = jca.properties(material, freqs)
    porous_props = PorousMediumProps(material=material, props=props, model=jca)
    response = surface_response_on_rigid_backing(
        porous_props, incidence_angle_deg=elevation_deg, time_convention=params.time_convention
    )

    p_total, reflection = PorousGroundPlaneWave.compute_plane_wave_over_porous(
        receivers=doublet.coordcart,
        params=params,
        material=material,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        model=jca,
        reflection_override=response.reflection_coefficient,
    )

    rigid_pressure, _ = PorousGroundPlaneWave.compute_plane_wave_over_porous(
        receivers=single.coordcart,
        params=params,
        material=material,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        reflection_override=np.ones_like(reflection),
    )

    pressures_single = p_total[:, :, 0]

    alpha_pv, zs_pv, _ = TwoMicPlaneWave.pv(
        pressures_single,
        doublet.coordcart,
        params.wavenumbers * params.c0,
        elevation_deg,
        params.z0,
        params.c0,
        mic_offset,
        params.time_convention,
    )
    alpha_tf, zs_tf, _ = TwoMicPlaneWave.transfer(
        pressures_single,
        doublet.coordcart,
        params.wavenumbers * params.c0,
        elevation_deg,
        params.z0,
        params.c0,
        mic_offset,
        params.time_convention,
    )

    alpha_mono, zs_mono = OneMicPlaneWave.estimate(
        pressures_single[:, [0]],
        rigid_pressure[:, :, 0],
        single.coordcart,
        params.wavenumbers,
        elevation_deg,
        params.z0,
        z_ground=0,
        time_convention=params.time_convention,
    )

    expected_alpha = response.absorption.squeeze()
    expected_zs = response.surface_impedance.squeeze()

    np.testing.assert_allclose(alpha_pv, expected_alpha, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(alpha_tf, expected_alpha, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(alpha_mono.squeeze(), expected_alpha, rtol=1e-2, atol=1e-3)

    np.testing.assert_allclose(zs_pv, expected_zs, rtol=2e-2, atol=5e-3)
    np.testing.assert_allclose(zs_tf, expected_zs, rtol=2e-2, atol=5e-3)
    np.testing.assert_allclose(zs_mono.squeeze(), expected_zs, rtol=2e-2, atol=5e-3)
