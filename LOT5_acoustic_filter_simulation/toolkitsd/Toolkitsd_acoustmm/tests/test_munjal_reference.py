import numpy as np

from toolkitsd.acoustmm import CylindricalDuct, tl_simple_expansion_analytic


def _tl_simple_expansion_tmm_from_elements(
    frequencies: np.ndarray,
    chamber_length: float,
    area_ratio: float,
    *,
    c0: float = 340.0,
    rho0: float = 1.2,
    inlet_radius: float = 4e-3,
) -> np.ndarray:
    freqs = np.asarray(frequencies, dtype=np.float64).ravel()
    S1 = np.pi * inlet_radius**2
    S2 = float(area_ratio) * S1
    r2 = np.sqrt(S2 / np.pi)
    omega = 2.0 * np.pi * freqs
    Zc1 = rho0 * c0 / S1
    chamber = CylindricalDuct(radius=r2, length=chamber_length, c0=c0, rho0=rho0)
    return chamber.TL(Z_c=Zc1, omega=omega)


def test_munjal_simple_expansion_zero_when_area_ratio_is_one():
    freqs = np.linspace(100.0, 2000.0, 20)
    tl = tl_simple_expansion_analytic(freqs, chamber_length=0.25, area_ratio=1.0)
    np.testing.assert_allclose(tl, 0.0, atol=1e-12, rtol=0.0)


def test_munjal_simple_expansion_tmm_matches_analytic_reference():
    freqs = np.linspace(100.0, 2000.0, 200)
    tl_ref = tl_simple_expansion_analytic(freqs, chamber_length=0.4, area_ratio=4.0, c0=340.0)
    tl_tmm = _tl_simple_expansion_tmm_from_elements(
        freqs,
        chamber_length=0.4,
        area_ratio=4.0,
        c0=340.0,
        rho0=1.2,
        inlet_radius=4e-3,
    )
    np.testing.assert_allclose(tl_tmm, tl_ref, atol=1e-10, rtol=1e-10)
