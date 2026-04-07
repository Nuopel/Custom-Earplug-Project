import numpy as np

from toolkitsd.acoustmm import ElasticSlab, ElasticSlabSeries, ElasticSlabThin


def _material_params():
    return {
        "radius": 7.5e-3,
        "length": 6.0e-3,
        "rho": 1500.0,
        "young": 2.9e6,
        "poisson": 0.49,
        "loss_factor": 0.20,
    }


def test_elastic_slab_exact_matches_a5_full_matrix():
    params = _material_params()
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    slab = ElasticSlab(**params)
    T = slab.matrix(omega)

    e_complex = params["young"] * (1.0 + 1j * params["loss_factor"])
    m_long = e_complex * (1.0 - params["poisson"]) / ((1.0 + params["poisson"]) * (1.0 - 2.0 * params["poisson"]))
    c_long = np.sqrt(m_long / params["rho"])
    z_long_acoustic = params["rho"] * c_long / slab.area
    phi = omega * params["length"] / c_long

    np.testing.assert_allclose(T[:, 0, 0], np.cos(phi), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 0, 1], 1j * z_long_acoustic * np.sin(phi), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 0], 1j * np.sin(phi) / z_long_acoustic, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 1], np.cos(phi), rtol=1e-12, atol=1e-12)


def test_elastic_slab_thin_matches_a5_thin_matrix_in_p_u_state():
    params = _material_params()
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    slab = ElasticSlabThin(**params)
    T = slab.matrix(omega)

    e_complex = params["young"] * (1.0 + 1j * params["loss_factor"])
    m_long = e_complex * (1.0 - params["poisson"]) / ((1.0 + params["poisson"]) * (1.0 - 2.0 * params["poisson"]))
    c_long = np.sqrt(m_long / params["rho"])

    t12_ref = 1j * omega * params["rho"] * params["length"] * (1.0 + 1j * params["loss_factor"]) / slab.area
    t21_ref = (1j * omega * params["length"] / (params["rho"] * c_long**2)) * slab.area

    np.testing.assert_allclose(T[:, 0, 0], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 0, 1], t12_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 0], t21_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 1], 1.0, rtol=1e-12, atol=1e-12)


def test_elastic_slab_series_matches_mass_only_a5_model_in_p_u_state():
    params = _material_params()
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    slab = ElasticSlabSeries(**params)
    T = slab.matrix(omega)

    z_ref = 1j * omega * params["rho"] * params["length"] * (1.0 + 1j * params["loss_factor"]) / slab.area

    np.testing.assert_allclose(T[:, 0, 0], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 0, 1], z_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 0], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 1], 1.0, rtol=1e-12, atol=1e-12)


def test_elastic_slab_thin_converges_to_exact_at_low_frequency():
    params = _material_params()
    params["loss_factor"] = 0.0
    omega = 2.0 * np.pi * np.array([20.0, 50.0, 100.0])
    exact = ElasticSlab(**params).matrix(omega)
    thin = ElasticSlabThin(**params).matrix(omega)
    np.testing.assert_allclose(thin, exact, rtol=2e-2, atol=2e-2)
