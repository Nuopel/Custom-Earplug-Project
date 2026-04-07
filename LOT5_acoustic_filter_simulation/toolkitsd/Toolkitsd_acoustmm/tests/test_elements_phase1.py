import numpy as np

from toolkitsd.acoustmm import (
    AcousticElement,
    ConicalDuct,
    CylindricalDuct,
    FrozenMatrixElement,
    ImpedanceJunction,
    MatchedLoad,
    ParallelElement,
    RadiationImpedance,
    RigidWall,
)


class IdentityElement(AcousticElement):
    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega).ravel()
        return np.broadcast_to(np.eye(2, dtype=np.complex128), (omega.size, 2, 2)).copy()


class SingularElement(AcousticElement):
    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = np.asarray(omega).ravel()
        base = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        return np.broadcast_to(base, (omega.size, 2, 2)).copy()


def test_sum_chain_returns_valid_matrix_shape():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0])
    chain = sum([IdentityElement(), IdentityElement(), IdentityElement()])
    T = chain.matrix(omega)
    assert T.shape == (2, 2, 2)
    np.testing.assert_allclose(T, np.broadcast_to(np.eye(2), (2, 2, 2)))


def test_subtraction_uncascades_the_rightmost_element():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0])
    air = CylindricalDuct(radius=4e-3, length=20e-3, c0=340.0, rho0=1.2)
    slab = IdentityElement()
    recovered = (slab + air - air).matrix(omega)
    np.testing.assert_allclose(recovered, slab.matrix(omega), rtol=1e-9, atol=1e-9)


def test_subtraction_recovers_left_element_from_physical_chain():
    omega = 2.0 * np.pi * np.array([200.0, 1500.0])
    air1 = CylindricalDuct(radius=4e-3, length=15e-3, c0=340.0, rho0=1.2)
    air2 = CylindricalDuct(radius=4e-3, length=8e-3, c0=340.0, rho0=1.2)
    recovered = ((air1 + air2) - air2).matrix(omega)
    np.testing.assert_allclose(recovered, air1.matrix(omega), rtol=1e-12, atol=1e-12)


def test_decascade_right_matches_subtraction():
    omega = 2.0 * np.pi * np.array([250.0, 900.0])
    air1 = CylindricalDuct(radius=4e-3, length=12e-3, c0=340.0, rho0=1.2)
    air2 = CylindricalDuct(radius=4e-3, length=9e-3, c0=340.0, rho0=1.2)
    recovered_named = (air1 + air2).decascade_right(air2).matrix(omega)
    recovered_minus = ((air1 + air2) - air2).matrix(omega)
    np.testing.assert_allclose(recovered_named, recovered_minus, rtol=1e-12, atol=1e-12)


def test_decascade_left_removes_the_first_element():
    omega = 2.0 * np.pi * np.array([250.0, 900.0])
    air1 = CylindricalDuct(radius=4e-3, length=12e-3, c0=340.0, rho0=1.2)
    air2 = CylindricalDuct(radius=4e-3, length=9e-3, c0=340.0, rho0=1.2)
    recovered = (air1 + air2).decascade_left(air1).matrix(omega)
    np.testing.assert_allclose(recovered, air2.matrix(omega), rtol=1e-12, atol=1e-12)


def test_parallel_of_series_impedances_matches_scalar_parallel_formula():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0])
    z1 = np.array([2.0 + 3.0j, 4.0 + 1.0j], dtype=np.complex128)
    z2 = np.array([5.0 + 2.0j, 3.0 + 6.0j], dtype=np.complex128)
    branch_1 = FrozenMatrixElement.from_pu(
        np.stack(
            [
                np.array([[1.0, z_val], [0.0, 1.0]], dtype=np.complex128)
                for z_val in z1
            ],
            axis=0,
        )
    )
    branch_2 = FrozenMatrixElement.from_pu(
        np.stack(
            [
                np.array([[1.0, z_val], [0.0, 1.0]], dtype=np.complex128)
                for z_val in z2
            ],
            axis=0,
        )
    )

    equivalent = branch_1.in_parallel_with(branch_2).matrix(omega)
    z_eq = (z1 * z2) / (z1 + z2)
    expected = np.stack(
        [
            np.array([[1.0, z_val], [0.0, 1.0]], dtype=np.complex128)
            for z_val in z_eq
        ],
        axis=0,
    )
    np.testing.assert_allclose(equivalent, expected, rtol=1e-12, atol=1e-12)


def test_parallel_operator_matches_named_method_and_supports_chaining():
    omega = 2.0 * np.pi * np.array([200.0])
    branch_1 = FrozenMatrixElement.from_pu(np.array([[[1.0, 2.0], [0.0, 1.0]]], dtype=np.complex128))
    branch_2 = FrozenMatrixElement.from_pu(np.array([[[1.0, 3.0], [0.0, 1.0]]], dtype=np.complex128))
    branch_3 = FrozenMatrixElement.from_pu(np.array([[[1.0, 6.0], [0.0, 1.0]]], dtype=np.complex128))

    via_method = branch_1.in_parallel_with(branch_2, branch_3)
    via_operator = branch_1 // branch_2 // branch_3

    assert isinstance(via_method, ParallelElement)
    assert isinstance(via_operator, ParallelElement)
    np.testing.assert_allclose(via_operator.matrix(omega), via_method.matrix(omega), rtol=1e-12, atol=1e-12)


def test_parallel_rejects_non_pu_state_basis():
    pu_element = FrozenMatrixElement.from_pu(np.broadcast_to(np.eye(2), (1, 2, 2)))
    pv_element = FrozenMatrixElement.from_pv(np.broadcast_to(np.eye(2), (1, 2, 2)))

    with np.testing.assert_raises(ValueError):
        pu_element.in_parallel_with(pv_element)


def test_frozen_matrix_can_convert_pv_to_pu():
    area = 2.5
    matrix_pv = np.array([[[1.0, 10.0], [20.0, 1.0]]], dtype=np.complex128)
    converted = FrozenMatrixElement.from_pv_converted_to_pu(matrix_pv, area).matrices
    expected = np.array([[[1.0, 4.0], [50.0, 1.0]]], dtype=np.complex128)
    np.testing.assert_allclose(converted, expected)


def test_cascade_rejects_mixed_state_bases():
    omega = 2.0 * np.pi * np.array([1000.0])
    pu_element = FrozenMatrixElement.from_pu(np.broadcast_to(np.eye(2), (1, 2, 2)))
    pv_element = FrozenMatrixElement.from_pv(np.broadcast_to(np.eye(2), (1, 2, 2)))
    with np.testing.assert_raises(ValueError):
        (pu_element + pv_element).matrix(omega)


def test_decascade_direct_raises_on_singular_removed_matrix():
    omega = 2.0 * np.pi * np.array([250.0, 900.0])
    with np.testing.assert_raises(np.linalg.LinAlgError):
        IdentityElement().decascade_right(SingularElement()).matrix(omega)


def test_decascade_tikhonov_returns_finite_matrix_for_singular_removed_matrix():
    omega = 2.0 * np.pi * np.array([250.0, 900.0])
    recovered = IdentityElement().decascade_right(
        SingularElement(),
        method="tikhonov",
        regularization=1.0e-6,
    ).matrix(omega)
    assert np.all(np.isfinite(recovered.real))
    assert np.all(np.isfinite(recovered.imag))


def test_decascade_lcurve_returns_finite_matrix_for_singular_removed_matrix():
    omega = 2.0 * np.pi * np.array([250.0, 900.0])
    recovered = IdentityElement().decascade_right(
        SingularElement(),
        method="lcurve",
        lambda_grid=np.logspace(-12.0, -2.0, 21),
    ).matrix(omega)
    assert np.all(np.isfinite(recovered.real))
    assert np.all(np.isfinite(recovered.imag))


def test_cylindrical_matched_load_gives_characteristic_impedance():
    f = np.linspace(100.0, 2000.0, 32)
    omega = 2.0 * np.pi * f
    duct = CylindricalDuct(radius=4e-3, length=20e-3, c0=340.0, rho0=1.2)
    Z_load = MatchedLoad(area=duct.area, c0=duct.c0, rho0=duct.rho0).Z(omega)
    Zin = duct.Z_in(Z_load, omega)
    np.testing.assert_allclose(Zin, duct.Zc, rtol=1e-12, atol=1e-12)


def test_cylindrical_has_first_mode_metadata_on_creation():
    duct = CylindricalDuct(radius=4e-3, length=20e-3, c0=340.0, rho0=1.2)
    assert duct.first_mode_bc == "rigid"
    assert duct.first_mode_id == (1, 1)
    assert duct.first_mode_cutoff_hz > 0.0


def test_quarter_wave_with_rigid_wall_has_impedance_null():
    L = 25e-3
    c0 = 340.0
    f_qw = c0 / (4.0 * L)
    omega = np.array([2.0 * np.pi * f_qw])
    duct = CylindricalDuct(radius=4e-3, length=L, c0=c0, rho0=1.2)
    Z_load = RigidWall().Z(omega)
    Zin = duct.Z_in(Z_load, omega)
    assert np.abs(Zin[0]) < 1e-6 * duct.Zc


def test_rigid_wall_reflection_magnitude_is_one():
    freqs = np.linspace(150.0, 3500.0, 64)
    omega = 2.0 * np.pi * freqs
    duct = CylindricalDuct(radius=4e-3, length=18e-3, c0=340.0, rho0=1.2)
    Zin = duct.Z_in(RigidWall().Z(omega), omega)
    r = (Zin - duct.Zc) / (Zin + duct.Zc)
    np.testing.assert_allclose(np.abs(r), 1.0, rtol=1e-10, atol=1e-10)


def test_conical_converges_to_cylindrical_when_radii_equal():
    f = np.linspace(200.0, 4000.0, 40)
    omega = 2.0 * np.pi * f
    cyl = CylindricalDuct(radius=3e-3, length=30e-3)
    cone = ConicalDuct(r1=3e-3, r2=3e-3, length=30e-3)
    np.testing.assert_allclose(cone.matrix(omega), cyl.matrix(omega), rtol=1e-10, atol=1e-10)


def test_conical_reciprocity_determinant_is_unity_for_p_u_state():
    omega = 2.0 * np.pi * np.linspace(200.0, 3000.0, 20)
    r1 = 2.5e-3
    r2 = 4.0e-3
    cone = ConicalDuct(r1=r1, r2=r2, length=25e-3)
    T = cone.matrix(omega)
    det = T[:, 0, 0] * T[:, 1, 1] - T[:, 0, 1] * T[:, 1, 0]
    np.testing.assert_allclose(det, 1.0, rtol=1e-9, atol=1e-9)


def test_impedance_junction_ratio_and_end_correction():
    omega = 2.0 * np.pi * np.array([1000.0])
    S1 = np.pi * (4e-3) ** 2
    S2 = np.pi * (2e-3) ** 2
    no_ec = ImpedanceJunction(S1=S1, S2=S2, end_correction=False).matrix(omega)
    with_ec = ImpedanceJunction(S1=S1, S2=S2, end_correction=True).matrix(omega)
    np.testing.assert_allclose(no_ec, np.broadcast_to(np.eye(2), no_ec.shape))
    assert np.abs(with_ec[0, 0, 1]) > 0.0


def test_radiation_impedance_is_finite_and_positive_real_unflanged():
    omega = 2.0 * np.pi * np.array([200.0, 1000.0, 3000.0])
    z_rad = RadiationImpedance(radius=2e-3, mode="unflanged").Z(omega)
    assert np.all(np.isfinite(z_rad.real))
    assert np.all(z_rad.real >= 0.0)


def test_radiation_impedance_unflanged_v2_is_finite_and_differs_from_low_order():
    omega = 2.0 * np.pi * np.array([200.0, 1000.0, 3000.0, 8000.0])
    z_v1 = RadiationImpedance(radius=2e-3, mode="unflanged").Z(omega)
    z_v2 = RadiationImpedance(radius=2e-3, mode="unflanged_v2").Z(omega)
    assert np.all(np.isfinite(z_v2.real))
    assert np.all(np.isfinite(z_v2.imag))
    # v2 includes higher-order terms and should not collapse to v1 on this band.
    assert np.any(np.abs(z_v2 - z_v1) > 0.0)
