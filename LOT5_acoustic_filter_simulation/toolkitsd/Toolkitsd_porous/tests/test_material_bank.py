import pytest

from toolkitsd.porous import JCAMaterial, PorousMaterial, get_material_preset, list_material_presets


def test_material_bank_lists_known_presets():
    names = list_material_presets()
    assert "melamine_cttm" in names
    assert "mousse_materiau3_cttm" in names


def test_material_bank_builds_material_with_overridden_air_props():
    mat = get_material_preset("melamine_cttm", rho0=1.2, c0=340.0)
    assert mat.name is not None
    assert mat.sigma > 0.0
    assert mat.thickness > 0.0
    assert mat.rho0 == 1.2
    assert mat.c0 == 340.0


def test_material_bank_class_wrapper_is_available_on_porous_material():
    mat = PorousMaterial.get_material_preset("melamine_cttm", rho0=1.213, c0=342.2)
    assert isinstance(mat, PorousMaterial)
    assert isinstance(mat, JCAMaterial)
    assert mat.name is not None


def test_material_bank_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown material preset"):
        get_material_preset("does_not_exist")
