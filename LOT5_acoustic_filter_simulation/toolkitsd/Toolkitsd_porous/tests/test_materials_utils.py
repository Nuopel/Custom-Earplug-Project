import numpy as np
import pytest

from toolkitsd.porous.materials import JCAMaterial, MikiMaterial, PorousMaterial
from toolkitsd.porous.utils import as_angle_array, as_frequency_array, column_vector


def _valid_material_kwargs() -> dict:
    return dict(
        sigma=10000.0,
        thickness=0.05,
        phi=0.8,
        lambda1=1e-4,
        lambdap=2e-4,
        tortu=1.1,
        rho0=1.2,
        c0=340.0,
        name="fixture",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sigma", 0.0),
        ("thickness", -0.01),
        ("phi", 1.5),
        ("phi", 0.0),
        ("lambda1", 0.0),
        ("lambdap", 0.0),
        ("tortu", 0.0),
        ("rho0", -1.0),
        ("c0", 0.0),
    ],
)
def test_porous_material_validation(field: str, value: float) -> None:
    kwargs = _valid_material_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        PorousMaterial(**kwargs)


def test_porous_material_z0_property() -> None:
    mat = JCAMaterial(**_valid_material_kwargs())
    assert mat.z0 == pytest.approx(mat.rho0 * mat.c0)


def test_legacy_porous_material_dispatches_to_jca_with_warning() -> None:
    with pytest.deprecated_call():
        mat = PorousMaterial(**_valid_material_kwargs())
    assert isinstance(mat, JCAMaterial)


def test_legacy_porous_material_dispatches_to_miki_with_warning() -> None:
    with pytest.deprecated_call():
        mat = PorousMaterial(sigma=10000.0, thickness=0.05, rho0=1.2, c0=340.0, name="fixture")
    assert isinstance(mat, MikiMaterial)


def test_legacy_porous_material_rejects_partial_jca_definition() -> None:
    with pytest.deprecated_call():
        with pytest.raises(ValueError, match="Incomplete JCA material definition"):
            PorousMaterial(sigma=10000.0, thickness=0.05, phi=0.8, rho0=1.2, c0=340.0)


def test_frequency_and_angle_validators() -> None:
    freqs = as_frequency_array([[100, 200, 300]])
    assert freqs.shape == (3,)
    assert freqs.dtype == float

    with pytest.raises(ValueError):
        as_frequency_array([0, 100])
    with pytest.raises(ValueError):
        as_frequency_array([])

    angles = as_angle_array([[0, 30, 90]])
    assert angles.tolist() == [0.0, 30.0, 90.0]
    with pytest.raises(ValueError):
        as_angle_array([-10, 45])
    with pytest.raises(ValueError):
        as_angle_array([])


def test_column_vector_shapes_arrays() -> None:
    vec = column_vector(np.array([1, 2, 3]))
    assert vec.shape == (3, 1)
