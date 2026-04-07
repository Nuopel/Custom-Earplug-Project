import numpy as np
import pytest

from scipy.special import jv

from toolkitsd.acoustmm import (
    BLIDuct,
    CylindricalDuct,
    KirchhoffStinsonEquivalentFluidModel,
    LosslessCircularModel,
    ViscothermalDuct,
)


def test_viscothermal_matrix_shape_and_finite_values():
    freqs = np.linspace(100.0, 4000.0, 50)
    omega = 2.0 * np.pi * freqs
    duct = ViscothermalDuct(radius=2.5e-3, length=20e-3, c0=340.0, rho0=1.2)
    T = duct.matrix(omega)

    assert T.shape == (freqs.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_viscothermal_converges_to_lossless_for_large_radius():
    freqs = np.linspace(100.0, 3000.0, 60)
    omega = 2.0 * np.pi * freqs

    r = 5e-2
    L = 0.15
    vt = ViscothermalDuct(radius=r, length=L, c0=340.0, rho0=1.2)
    ls = CylindricalDuct(radius=r, length=L, c0=340.0, rho0=1.2)

    T_vt = vt.matrix(omega)
    T_ls = ls.matrix(omega)
    abs_err = np.abs(T_vt - T_ls)
    mag_err_01 = np.abs(np.abs(T_vt[:, 0, 1]) - np.abs(T_ls[:, 0, 1]))
    mag_err_10 = np.abs(np.abs(T_vt[:, 1, 0]) - np.abs(T_ls[:, 1, 0]))

    assert float(np.max(abs_err[:, 0, 0])) < 3e-2
    assert float(np.max(abs_err[:, 1, 1])) < 3e-2
    # Off-diagonal phase/sign is convention-sensitive; compare magnitude convergence.
    assert float(np.max(mag_err_01)) < 3.5e2
    assert float(np.max(mag_err_10)) < 1.5e-7


def test_viscothermal_attenuation_component_increases_with_frequency():
    freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0])
    omega = 2.0 * np.pi * freqs
    vt = ViscothermalDuct(radius=2.0e-3, length=20e-3, c0=340.0, rho0=1.2)

    gamma_vt, _ = vt._gamma_zc(omega)
    alpha = np.real(gamma_vt)

    assert np.all(alpha >= 0.0)
    assert alpha[-1] > alpha[0]


def test_bli_matrix_shape_and_finite_values():
    freqs = np.linspace(100.0, 4000.0, 50)
    omega = 2.0 * np.pi * freqs
    duct = BLIDuct(radius=2.5e-3, length=20e-3, c0=340.0, rho0=1.2)
    T = duct.matrix(omega)
    assert T.shape == (freqs.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_bli_converges_to_lossless_for_large_radius():
    freqs = np.linspace(100.0, 3000.0, 60)
    omega = 2.0 * np.pi * freqs
    r = 5e-2
    L = 0.15

    bli = BLIDuct(radius=r, length=L, c0=340.0, rho0=1.2)
    ls = CylindricalDuct(radius=r, length=L, c0=340.0, rho0=1.2)

    T_bli = bli.matrix(omega)
    T_ls = ls.matrix(omega)
    abs_err = np.abs(T_bli - T_ls)

    assert float(np.max(abs_err[:, 0, 0])) < 5e-2
    assert float(np.max(abs_err[:, 1, 1])) < 5e-2


def test_bli_and_viscothermal_have_same_attenuation_order():
    omega = 2.0 * np.pi * np.array([500.0, 1000.0, 2000.0, 4000.0])
    vt = ViscothermalDuct(radius=0.5e-3, length=20e-3, c0=340.0, rho0=1.2)
    bli = BLIDuct(radius=0.5e-3, length=20e-3, c0=340.0, rho0=1.2)

    gamma_vt, _ = vt._gamma_zc(omega)
    gamma_bli, _ = bli._gamma_zc(omega)
    ratio = np.real(gamma_bli) / np.maximum(np.real(gamma_vt), 1e-12)

    # BLI is approximate; keep a wide envelope but same order of magnitude.
    assert np.all(ratio > 0.3)
    assert np.all(ratio < 3.0)


def test_bli_optional_zc_correction_changes_characteristic_impedance():
    omega = 2.0 * np.pi * np.array([500.0, 1000.0, 2000.0])
    bli_no_zc = BLIDuct(radius=0.5e-3, length=20e-3, c0=340.0, rho0=1.2, correct_zc=False)
    bli_yes_zc = BLIDuct(radius=0.5e-3, length=20e-3, c0=340.0, rho0=1.2, correct_zc=True)

    _, zc_no = bli_no_zc._gamma_zc(omega)
    _, zc_yes = bli_yes_zc._gamma_zc(omega)

    assert np.all(np.isfinite(zc_yes.real))
    assert np.all(np.isfinite(zc_yes.imag))
    assert np.any(np.abs(zc_yes - zc_no) > 0.0)


def test_bli_warns_for_marginal_radius_validity():
    with pytest.warns(RuntimeWarning, match="BLIDuct validity may be marginal"):
        BLIDuct(radius=0.1e-3, length=20e-3, c0=340.0, rho0=1.2)


def test_equivalent_fluid_circular_model_matches_reference_helper():
    rho_0 = 1.213
    gamma = 1.4
    P_0 = 101325.0
    eta_0 = 1.839e-5
    Pr = 0.71
    K_0 = gamma * P_0
    c_0 = np.sqrt(K_0 / rho_0)

    radius = 2.5e-3
    area = np.pi * radius**2
    omega = 2.0 * np.pi * np.array([100.0, 500.0, 1000.0, 2500.0])

    model = KirchhoffStinsonEquivalentFluidModel(
        radius=radius,
        area=area,
        c0=c_0,
        rho0=rho_0,
        P0=P_0,
        eta0=eta_0,
        Pr=Pr,
    )

    rho_eff, K_eff, k_eff, z_line = model.equivalent_fluid_properties(omega)

    G_r = np.sqrt(-1j * omega * rho_0 / eta_0)
    G_k = np.sqrt(-1j * omega * Pr * rho_0 / eta_0)
    x_r = radius * G_r
    x_k = radius * G_k

    rho_ref = rho_0 * (1.0 - (2.0 * jv(1, x_r)) / (x_r * jv(0, x_r))) ** (-1)
    K_ref = K_0 * (1.0 + (gamma - 1.0) * (2.0 * jv(1, x_k)) / (x_k * jv(0, x_k))) ** (-1)
    k_ref = omega * np.sqrt(rho_ref / K_ref)
    z_ref = np.sqrt(rho_ref * K_ref) / area

    assert np.allclose(rho_eff, rho_ref)
    assert np.allclose(K_eff, K_ref)
    assert np.allclose(k_eff, k_ref)
    assert np.allclose(z_line, z_ref)


def test_viscothermal_duct_accepts_equivalent_fluid_loss_model():
    P_0 = 101325.0
    gamma = 1.4
    rho_0 = 1.213
    eta_0 = 1.839e-5
    Pr = 0.71
    K_0 = gamma * P_0
    c_0 = np.sqrt(K_0 / rho_0)

    radius = 2.5e-3
    length = 20e-3
    area = np.pi * radius**2
    omega = 2.0 * np.pi * np.array([100.0, 500.0, 1000.0])

    model = KirchhoffStinsonEquivalentFluidModel(
        radius=radius,
        area=area,
        c0=c_0,
        rho0=rho_0,
        P0=P_0,
        eta0=eta_0,
        Pr=Pr,
    )
    duct = ViscothermalDuct(
        radius=radius,
        length=length,
        c0=c_0,
        rho0=rho_0,
        loss_model=model,
    )

    T = duct.matrix(omega)

    assert T.shape == (omega.size, 2, 2)
    assert np.all(np.isfinite(T.real))
    assert np.all(np.isfinite(T.imag))


def test_derived_gamma_matches_ideal_gas_relation():
    model = KirchhoffStinsonEquivalentFluidModel(
        radius=2.5e-3,
        area=np.pi * (2.5e-3)**2,
        c0=343.0,
        rho0=1.213,
        P0=101325.0,
    )
    assert np.isclose(model.gamma, model.rho0 * model.c0**2 / model.P0)


def test_lossless_circular_model_matches_duct_limit():
    rho_0 = 1.213
    c_0 = 343.0
    radius = 2.5e-3
    area = np.pi * radius**2
    omega = 2.0 * np.pi * np.array([100.0, 500.0, 1000.0])

    model = LosslessCircularModel(
        radius=radius,
        area=area,
        c0=c_0,
        rho0=rho_0,
    )

    gamma, zc = model.gamma_zc(omega)

    assert np.allclose(gamma, 1j * omega / c_0)
    assert np.allclose(zc, np.full(omega.shape, rho_0 * c_0 / area + 0j))


def test_equivalent_fluid_properties_are_consistent():
    model = KirchhoffStinsonEquivalentFluidModel(
        radius=2.5e-3,
        area=np.pi * (2.5e-3)**2,
        c0=343.0,
        rho0=1.213,
    )
    omega = 2.0 * np.pi * np.array([100.0, 500.0, 1000.0])

    rho_eff, K_eff, k_eff, zc_eff = model.equivalent_fluid_properties(omega)
    gamma, zc = model.gamma_zc(omega)

    assert np.all(np.isfinite(rho_eff.real))
    assert np.all(np.isfinite(K_eff.real))
    gamma_from_k = 1j * k_eff
    gamma_from_k = np.where(np.real(gamma_from_k) < 0.0, -gamma_from_k, gamma_from_k)
    assert np.allclose(gamma, gamma_from_k)
    assert np.allclose(zc, zc_eff)
