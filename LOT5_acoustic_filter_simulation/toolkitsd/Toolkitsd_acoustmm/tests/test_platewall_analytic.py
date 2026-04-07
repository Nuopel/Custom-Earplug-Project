import numpy as np

from toolkitsd.acoustmm import InfinitePlate, calculate_zp_parois_simple, integrate_3d_diffuse, tl_paroi_analytic


def test_infinite_plate_specific_impedance_matches_analytic_formula():
    freqs = np.linspace(80.0, 8000.0, 50)
    omega = 2.0 * np.pi * freqs
    c0 = 343.0

    plate = InfinitePlate(rho_plate=2500.0, h=5.0e-3, E=70.0e9, nu=0.2, theta=0.0, c0=c0)
    z_elem = plate.specific_impedance(omega)

    z_ref = calculate_zp_parois_simple(
        omega=omega,
        k0=omega / c0,
        theta=0.0,
        E=70.0e9,
        h=5.0e-3,
        nu=0.2,
        mu=2500.0 * 5.0e-3,
    )
    np.testing.assert_allclose(z_elem, z_ref, rtol=1e-12, atol=1e-12)


def test_infinite_plate_tl_matches_tl_paroi_analytic_at_normal_incidence():
    freqs = np.linspace(100.0, 5000.0, 80)
    omega = 2.0 * np.pi * freqs
    rho0 = 1.2
    c0 = 343.0
    z0 = rho0 * c0

    rho_plate = 750.0
    h = 12.8e-3
    mu = rho_plate * h
    E = 3.0e9
    nu = 0.245

    plate = InfinitePlate(rho_plate=rho_plate, h=h, E=E, nu=nu, theta=0.0, c0=c0)
    tl_elem = plate.TL(Z_c=z0, omega=omega)
    tl_ref = tl_paroi_analytic(
        omega=omega,
        theta=0.0,
        rho0=rho0,
        c0=c0,
        mu=mu,
        E=E,
        h=h,
        nu=nu,
    )

    np.testing.assert_allclose(tl_elem, tl_ref, rtol=1e-11, atol=1e-11)


def test_infinite_plate_tl_matches_tl_paroi_analytic_at_oblique_incidence():
    freqs = np.linspace(100.0, 5000.0, 80)
    omega = 2.0 * np.pi * freqs
    rho0 = 1.2
    c0 = 343.0
    z0 = rho0 * c0

    theta = np.deg2rad(40.0)
    rho_plate = 2500.0
    h = 5.0e-3
    mu = rho_plate * h
    E = 70.0e9
    nu = 0.2

    plate = InfinitePlate(rho_plate=rho_plate, h=h, E=E, nu=nu, theta=theta, c0=c0)
    tl_elem = plate.TL(Z_c=z0, omega=omega)
    tl_ref = tl_paroi_analytic(
        omega=omega,
        theta=theta,
        rho0=rho0,
        c0=c0,
        mu=mu,
        E=E,
        h=h,
        nu=nu,
    )

    np.testing.assert_allclose(tl_elem, tl_ref, rtol=1e-11, atol=1e-11)


def test_infinite_plate_tl_diffuse_matches_legacy_integration():
    freqs = np.linspace(100.0, 5000.0, 60)
    omega = 2.0 * np.pi * freqs
    rho0 = 1.2
    c0 = 343.0
    z0 = rho0 * c0

    rho_plate = 2500.0
    h = 5.0e-3
    mu = rho_plate * h
    E = 70.0e9
    nu = 0.2
    theta_lim = np.pi / 2.0 - 1e-9
    n_eval = 48

    plate = InfinitePlate(rho_plate=rho_plate, h=h, E=E, nu=nu, theta=0.0, c0=c0)
    tl_diff_plate = plate.TL_diffuse(Z_c=z0, omega=omega, theta_lim=theta_lim, n_eval=n_eval)

    tl_diff_legacy = integrate_3d_diffuse(
        lambda f, theta: tl_paroi_analytic(2.0 * np.pi * f, theta, rho0, c0, mu, E, h, nu),
        frequencies=freqs,
        theta_lim=theta_lim,
        n_eval=n_eval,
    )

    np.testing.assert_allclose(tl_diff_plate, tl_diff_legacy, rtol=5e-12, atol=5e-12)
