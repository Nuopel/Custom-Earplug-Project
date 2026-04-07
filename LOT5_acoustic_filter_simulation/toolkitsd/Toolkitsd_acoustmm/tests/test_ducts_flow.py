import numpy as np
import pytest

from toolkitsd.acoustmm import CylindricalDuct, FlowDuct, MatchedLoad


def test_flowduct_mach_zero_matches_cylindrical_duct():
    freqs = np.linspace(100.0, 5000.0, 80)
    omega = 2.0 * np.pi * freqs

    cyl = CylindricalDuct(radius=3e-3, length=40e-3, c0=340.0, rho0=1.2)
    flow = FlowDuct(radius=3e-3, length=40e-3, mach=0.0, c0=340.0, rho0=1.2)

    np.testing.assert_allclose(flow.matrix(omega), cyl.matrix(omega), rtol=1e-12, atol=1e-12)


def test_flowduct_nonzero_mach_changes_transfer_matrix_and_stays_finite():
    freqs = np.linspace(200.0, 4000.0, 60)
    omega = 2.0 * np.pi * freqs
    flow0 = FlowDuct(radius=3e-3, length=40e-3, mach=0.0, c0=340.0, rho0=1.2).matrix(omega)
    flow1 = FlowDuct(radius=3e-3, length=40e-3, mach=0.1, c0=340.0, rho0=1.2).matrix(omega)

    assert np.all(np.isfinite(flow1.real))
    assert np.all(np.isfinite(flow1.imag))
    assert np.any(np.abs(flow1 - flow0) > 0.0)


def test_flowduct_zin_changes_with_mach_for_same_load():
    freqs = np.linspace(100.0, 5000.0, 80)
    omega = 2.0 * np.pi * freqs
    area = np.pi * (3e-3) ** 2
    z_load = MatchedLoad(area=area, c0=340.0, rho0=1.2).Z(omega)

    zin0 = FlowDuct(radius=3e-3, length=40e-3, mach=0.0, c0=340.0, rho0=1.2).Z_in(z_load, omega)
    zin1 = FlowDuct(radius=3e-3, length=40e-3, mach=0.1, c0=340.0, rho0=1.2).Z_in(z_load, omega)
    assert np.any(np.abs(zin1 - zin0) > 0.0)


def test_flowduct_invalid_mach_raises():
    with pytest.raises(ValueError, match="mach must satisfy"):
        FlowDuct(radius=3e-3, length=40e-3, mach=1.0, c0=340.0, rho0=1.2)
