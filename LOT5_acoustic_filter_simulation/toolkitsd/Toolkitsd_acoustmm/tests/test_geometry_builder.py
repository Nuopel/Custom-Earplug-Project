import numpy as np
import pytest
import warnings

from toolkitsd.acoustmm import EarCanalBuilder


def test_ear_canal_builder_placeholder_warns_and_builds():
    builder = EarCanalBuilder(n_segments=20, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * np.array([200.0, 1000.0, 4000.0])
    z_load = np.full(omega.shape, 1.0e9 + 0j)

    with pytest.warns(RuntimeWarning, match="temporary placeholder ear-canal profile"):
        canal, segments = builder.build(return_segments=True)
    assert len(segments) == 20
    zin = canal.Z_in(z_load, omega)
    assert zin.shape == omega.shape
    assert np.all(np.isfinite(zin.real))
    assert np.all(np.isfinite(zin.imag))


def test_ear_canal_builder_custom_profile_no_placeholder_warning():
    builder = EarCanalBuilder(n_segments=16, radius_scale=1.0, c0=340.0, rho0=1.2)
    x = np.array([0.0, 6e-3, 12e-3, 18e-3, 24e-3])
    r = 1e-3 * np.array([4.0, 3.8, 3.5, 3.2, 3.0])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        canal, segments = builder.build(x=x, radius=r, return_segments=True)
    assert len(rec) == 0
    assert len(segments) == 16
    T = canal.matrix(2.0 * np.pi * np.array([500.0, 1500.0]))
    assert T.shape == (2, 2, 2)


def test_ear_canal_builder_rejects_invalid_profile():
    builder = EarCanalBuilder(n_segments=8, c0=340.0, rho0=1.2)
    with pytest.raises(ValueError, match="strictly increasing"):
        builder.build(x=np.array([0.0, 1.0, 1.0]), radius=np.array([1e-3, 1e-3, 1e-3]))


def test_ear_canal_builder_radius_scale_changes_segment_radii():
    x = np.array([0.0, 8e-3, 16e-3, 24e-3])
    r = np.array([4.0e-3, 3.7e-3, 3.3e-3, 3.0e-3])

    b0 = EarCanalBuilder(n_segments=12, radius_scale=1.0, c0=340.0, rho0=1.2)
    b1 = EarCanalBuilder(n_segments=12, radius_scale=1.1, c0=340.0, rho0=1.2)

    _, s0 = b0.build(x=x, radius=r, return_segments=True)
    _, s1 = b1.build(x=x, radius=r, return_segments=True)

    radii0 = np.array([seg.radius for seg in s0])
    radii1 = np.array([seg.radius for seg in s1])
    np.testing.assert_allclose(radii1 / radii0, 1.1, rtol=1e-12, atol=1e-12)
