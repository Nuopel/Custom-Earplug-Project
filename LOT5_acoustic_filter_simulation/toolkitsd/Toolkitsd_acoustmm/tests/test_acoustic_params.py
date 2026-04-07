import numpy as np

from toolkitsd.acoustmm import AcousticParameters


def test_acoustic_parameters_defaults_are_accessible_for_acoustmm():
    freqs = np.array([100.0, 250.0, 1000.0])
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)

    np.testing.assert_allclose(params.frequencies, freqs)
    assert params.c0 == 340.0
    assert params.rho0 == 1.2
    assert params.z0 == 408.0
    assert params.wavenumbers.shape == freqs.shape
