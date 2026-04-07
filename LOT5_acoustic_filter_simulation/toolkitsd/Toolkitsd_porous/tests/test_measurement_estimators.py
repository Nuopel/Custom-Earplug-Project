import numpy as np
import pytest

from toolkitsd.porous.measurement import BaseMeasurement, TwoMicPlaneWave


def test_time_convention_and_normalization_helpers() -> None:
    data = np.array([1 + 2j, 3 + 4j])
    np.testing.assert_allclose(BaseMeasurement._apply_time_convention(data, "jwt"), np.conj(data))
    np.testing.assert_allclose(BaseMeasurement._apply_time_convention(data, "neg_jwt"), data)

    reshaped = BaseMeasurement._normalize_pressures(np.array([1 + 1j, 2 + 2j]), expected_mics=1)
    assert reshaped.shape == (2, 1)

    flattened = BaseMeasurement._normalize_pressures(np.ones((3, 1, 1)), expected_mics=1)
    assert flattened.shape == (3, 1)

    with pytest.raises(ValueError):
        BaseMeasurement._normalize_pressures(np.ones((2, 2, 2)), expected_mics=2)
    with pytest.raises(ValueError):
        BaseMeasurement._normalize_pressures(np.ones((2, 3)), expected_mics=2)

    valid_mics = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.2]])
    np.testing.assert_allclose(BaseMeasurement._normalize_mics(valid_mics, expected_mics=2), valid_mics)
    with pytest.raises(ValueError):
        BaseMeasurement._normalize_mics(np.ones((2, 2)))
    with pytest.raises(ValueError):
        BaseMeasurement._normalize_mics(np.ones((3, 3)), expected_mics=2)
    with pytest.raises(ValueError):
        BaseMeasurement._dz(np.array([[0.0], [0.0], [0.1]]))


def test_two_mic_estimators_recover_reflection_coefficient() -> None:
    freqs = np.array([500.0, 1000.0])
    omega = 2 * np.pi * freqs
    c0 = 340.0
    rho0 = 1.2
    z0 = rho0 * c0
    incidence = 60.0  # elevation from plane
    sin_inc = np.sin(np.deg2rad(incidence))

    mic_height = 0.02
    dz = 0.015
    mic_positions = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [mic_height, mic_height + dz],
        ]
    )

    r_true = 0.35 + 0.1j
    k_z = omega / c0 * sin_inc

    def _field(z: float) -> np.ndarray:
        return np.exp(-1j * k_z * z) + r_true * np.exp(1j * k_z * z)

    pressures = np.stack([_field(mic_height), _field(mic_height + dz)], axis=1)

    alpha_tf, zs_tf, r_tf = TwoMicPlaneWave.transfer(
        pressures,
        mic_positions,
        omega,
        incidence_deg=incidence,
        z0=z0,
        c0=c0,
        mic_height=mic_height,
        time_convention="neg_jwt",
    )

    alpha_pv, zs_pv, r_pv = TwoMicPlaneWave.pv(
        pressures,
        mic_positions,
        omega,
        incidence_deg=incidence,
        z0=z0,
        c0=c0,
        mic_height=mic_height,
        time_convention="neg_jwt",
    )

    expected_alpha = 1 - np.abs(r_true) ** 2
    expected_zs = z0 / np.sin(np.deg2rad(incidence)) * (1 + r_true) / (1 - r_true)

    np.testing.assert_allclose(r_tf, r_true, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r_pv, r_true, rtol=1e-2, atol=3e-3)
    np.testing.assert_allclose(alpha_tf, expected_alpha, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(alpha_pv, expected_alpha, rtol=1e-3, atol=5e-4)
    np.testing.assert_allclose(zs_tf, expected_zs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(zs_pv, expected_zs, rtol=1e-2, atol=1e-2)
