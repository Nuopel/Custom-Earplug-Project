import numpy as np
import pytest

from toolkitsd.acoustmm import (
    first_mode_rect_duct,
    first_mode_round_duct,
    mode_frequencies_rect_duct,
    mode_frequencies_round_duct,
)


def test_rect_first_mode_for_wider_dimension():
    mode, f = first_mode_rect_duct(l=0.2, h=0.1, c0=340.0)
    assert mode == (1, 0)
    np.testing.assert_allclose(f, 340.0 / (2.0 * 0.2))


def test_rect_modes_sorted_excluding_00():
    f_grid, modes = mode_frequencies_rect_duct(l=0.1, h=0.05, c0=340.0, N=2, M=2, include_00=False)
    assert f_grid.shape == (3, 3)
    assert modes[0][0] in {(1, 0), (0, 1)}
    assert modes[0][1] > 0.0
    assert all(modes[i][1] <= modes[i + 1][1] for i in range(len(modes) - 1))


def test_round_first_mode_rigid_and_soft():
    mode_rigid, f_rigid = first_mode_round_duct(a=5e-3, c0=340.0, bc="rigid")
    mode_soft, f_soft = first_mode_round_duct(a=5e-3, c0=340.0, bc="soft")
    assert mode_rigid == (1, 1)
    assert mode_soft == (0, 1)
    assert f_rigid < f_soft


def test_round_mode_list_sorted_and_contains_plane_wave():
    cutoffs, modes = mode_frequencies_round_duct(
        a=5e-3, c0=340.0, m_max=2, n_max=2, bc="rigid", include_plane_wave=True
    )
    assert ((0, 1) in cutoffs) and (((0, 0), 0.0) in modes)
    assert all(modes[i][1] <= modes[i + 1][1] for i in range(len(modes) - 1))


def test_round_mode_invalid_bc_raises():
    with pytest.raises(ValueError):
        mode_frequencies_round_duct(a=5e-3, c0=340.0, bc="invalid")

