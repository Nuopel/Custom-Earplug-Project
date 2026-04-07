import numpy as np

from toolkitsd.acoustmm import EardrumImpedance, IEC711Coupler


def test_eardrum_impedance_is_finite_and_has_positive_real_part():
    freqs = np.array([100.0, 500.0, 1000.0, 4000.0])
    omega = 2.0 * np.pi * freqs
    z = EardrumImpedance().Z(omega)
    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))
    assert np.all(z.real > 0.0)


def test_iec711_coupler_tmm_returns_finite_impedance_with_resonant_variation():
    freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    omega = 2.0 * np.pi * freqs
    z = IEC711Coupler(c0=340.0, rho0=1.2).Z(omega)

    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))
    assert np.all(np.abs(z) > 0.0)
    assert np.ptp(np.abs(z)) > 1.0e6
    assert np.any(z.real > 0.0)
    assert np.any(z.imag > 0.0)
    assert np.any(z.imag < 0.0)


def test_iec711_coupler_compliance_mode_matches_legacy_behavior():
    freqs = np.array([100.0, 500.0, 1000.0, 4000.0])
    omega = 2.0 * np.pi * freqs
    z = IEC711Coupler(model="compliance", volume_m3=2.0e-6, c0=340.0, rho0=1.2).Z(omega)

    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))
    assert np.all(np.imag(z) < 0.0)
    assert np.abs(z[-1]) < np.abs(z[0])


def test_iec711_coupler_lumped_mode_is_finite_and_resonant():
    freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
    omega = 2.0 * np.pi * freqs
    coupler = IEC711Coupler(model="lumped", c0=340.0, rho0=1.2)
    z = coupler.Z(omega)
    resonances = coupler.branch_resonance_frequencies()

    assert np.all(np.isfinite(z.real))
    assert np.all(np.isfinite(z.imag))
    assert np.ptp(np.abs(z)) > 1.0e5
    assert resonances["hr2"] > resonances["hr1"] > 0.0
