import numpy as np
from scipy.special import iv, jv

from toolkitsd.acoustmm import ElasticSlabSeries, ExactFlexuralPlateSeriesImpedance, GenericFilmSeriesImpedance, LowFrequencyFlexuralPlateSeriesImpedance, MembraneSeriesImpedance, PlateSeriesImpedance


def test_generic_film_series_impedance_matches_r_m_k_formula():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    element = GenericFilmSeriesImpedance(
        resistance=12.0,
        mass=3.0e-5,
        stiffness=8.0e3,
    )

    z_ref = 12.0 + 1j * omega * 3.0e-5 + 8.0e3 / (1j * omega)
    T = element.matrix(omega)

    np.testing.assert_allclose(T[:, 0, 0], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 0, 1], z_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 0], 0.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T[:, 1, 1], 1.0, rtol=1e-12, atol=1e-12)


def test_generic_film_series_impedance_accepts_frequency_dependent_parameters():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    resistance = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0e-5, 2.0e-5, 3.0e-5])
    stiffness = np.array([100.0, 200.0, 300.0])
    element = GenericFilmSeriesImpedance(
        resistance=resistance,
        mass=mass,
        stiffness=stiffness,
    )

    z_ref = resistance + 1j * omega * mass + stiffness / (1j * omega)
    np.testing.assert_allclose(element.acoustic_series_impedance(omega), z_ref, rtol=1e-12, atol=1e-12)


def test_plate_series_matrix_is_built_from_acoustic_series_impedance():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    element = PlateSeriesImpedance(
        area=np.pi * (7.5e-3) ** 2,
        rho_plate=1200.0,
        h=80e-6,
        E=2.0e9,
        nu=0.3,
    )

    z_series = element.acoustic_series_impedance(omega)
    T = element.matrix(omega)
    np.testing.assert_allclose(T[:, 0, 1], z_series, rtol=1e-12, atol=1e-12)


def test_membrane_series_impedance_maps_surface_density_and_tension_to_rkm_terms():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    radius = 5.0e-3
    surface_density = 0.12
    tension = 15.0
    resistance = 42.0
    geometry_constant = 2.5
    area = np.pi * radius**2
    element = MembraneSeriesImpedance(
        radius=radius,
        surface_density=surface_density,
        tension=tension,
        resistance=resistance,
        geometry_constant=geometry_constant,
    )

    mass_ref = surface_density / area
    stiffness_ref = geometry_constant * tension / (radius**2 * area)
    z_ref = resistance + 1j * omega * mass_ref + stiffness_ref / (1j * omega)

    np.testing.assert_allclose(element.equivalent_mass, mass_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(element.equivalent_stiffness, stiffness_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(element.acoustic_series_impedance(omega), z_ref, rtol=1e-12, atol=1e-12)


def test_exact_flexural_plate_series_impedance_matches_d1_formula():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    radius = 5.0e-3
    rho_plate = 1200.0
    h = 80e-6
    young = 2.0e9
    poisson = 0.3
    area = np.pi * radius**2
    element = ExactFlexuralPlateSeriesImpedance(
        radius=radius,
        rho_plate=rho_plate,
        h=h,
        E=young,
        nu=poisson,
    )

    bending_stiffness = young * h**3 / (12.0 * (1.0 - poisson**2))
    plate_mass = rho_plate * h * area
    flexural_wavenumber = (rho_plate * h * omega**2 / bending_stiffness) ** 0.25
    x = flexural_wavenumber * radius
    num = iv(1, x) * jv(0, x) + jv(1, x) * iv(0, x)
    den = iv(1, x) * jv(2, x) - jv(1, x) * iv(2, x)
    z_ref = -1j * omega * plate_mass / (area**2) * (num / den)

    np.testing.assert_allclose(element.acoustic_series_impedance(omega), z_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(element.matrix(omega)[:, 0, 1], z_ref, rtol=1e-12, atol=1e-12)


def test_low_frequency_flexural_plate_series_impedance_matches_d2_formula():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    radius = 5.0e-3
    rho_plate = 1200.0
    h = 80e-6
    young = 2.0e9
    poisson = 0.3
    area = np.pi * radius**2
    element = LowFrequencyFlexuralPlateSeriesImpedance(
        radius=radius,
        rho_plate=rho_plate,
        h=h,
        E=young,
        nu=poisson,
    )

    bending_stiffness = young * h**3 / (12.0 * (1.0 - poisson**2))
    surface_mass = rho_plate * h
    flexural_wavenumber = (rho_plate * h * omega**2 / bending_stiffness) ** 0.25
    x = flexural_wavenumber * radius
    z_ref = -1j * omega * surface_mass * (192.0 / area) * (1.0 / x**4 - 3.0 / 320.0)

    np.testing.assert_allclose(element.acoustic_series_impedance(omega), z_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(element.matrix(omega)[:, 0, 1], z_ref, rtol=1e-12, atol=1e-12)


def test_elastic_slab_series_matrix_is_built_from_acoustic_series_impedance():
    omega = 2.0 * np.pi * np.array([100.0, 1000.0, 4000.0])
    element = ElasticSlabSeries(
        radius=7.5e-3,
        length=6.0e-3,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )

    z_series = element.acoustic_series_impedance(omega)
    T = element.matrix(omega)
    np.testing.assert_allclose(T[:, 0, 1], z_series, rtol=1e-12, atol=1e-12)
