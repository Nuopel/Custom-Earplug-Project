"""Higher-order mode cutoff helpers for rectangular and circular ducts."""

from __future__ import annotations

import numpy as np
from scipy.special import jn_zeros, jnp_zeros


def mode_frequencies_rect_duct(l, h, c0, N=5, M=5, include_00=False):
    """
    Cutoff frequencies f_nm (Hz) for a rectangular duct with dimensions l x h.

    f_nm = (c0/2) * sqrt((n/l)^2 + (m/h)^2)
    """
    if l <= 0 or h <= 0 or c0 <= 0:
        raise ValueError("l, h, c0 must be positive")
    if N < 0 or M < 0:
        raise ValueError("N and M must be >= 0")

    n = np.arange(N + 1)[:, None]
    m = np.arange(M + 1)[None, :]
    f = (c0 / 2.0) * np.sqrt((n / l) ** 2 + (m / h) ** 2)

    if include_00:
        nn, mm = np.indices(f.shape)
        modes = [((int(i), int(j)), float(freq)) for i, j, freq in zip(nn.ravel(), mm.ravel(), f.ravel())]
    else:
        mask = ~((n == 0) & (m == 0))
        n_idx, m_idx = np.where(mask)
        modes = [((int(i), int(j)), float(f[i, j])) for i, j in zip(n_idx, m_idx)]

    modes.sort(key=lambda x: x[1])
    return f, modes


def first_mode_rect_duct(l, h, c0):
    """
    First non-zero cutoff mode for a rectangular duct.
    """
    if l <= 0 or h <= 0 or c0 <= 0:
        raise ValueError("l, h, c0 must be positive")
    f10 = c0 / (2.0 * l)
    f01 = c0 / (2.0 * h)
    return ((1, 0), float(f10)) if f10 <= f01 else ((0, 1), float(f01))


def mode_frequencies_round_duct(a, c0, m_max=6, n_max=6, bc="rigid", include_plane_wave=False):
    """
    Cutoff frequencies for a circular duct of radius a (m).
    """
    if a <= 0 or c0 <= 0:
        raise ValueError("a and c0 must be positive")
    if m_max < 0 or n_max < 1:
        raise ValueError("m_max must be >= 0 and n_max must be >= 1")
    if bc not in {"rigid", "soft"}:
        raise ValueError("bc must be 'rigid' or 'soft'")

    zeros_fn = jnp_zeros if bc == "rigid" else jn_zeros

    cutoffs = {}
    modes = []
    if include_plane_wave:
        modes.append(((0, 0), 0.0))

    for m in range(m_max + 1):
        x = zeros_fn(m, n_max)
        f = (c0 / (2.0 * np.pi)) * (x / a)
        for n in range(1, n_max + 1):
            cutoffs[(m, n)] = float(f[n - 1])
            modes.append(((m, n), float(f[n - 1])))

    modes.sort(key=lambda t: t[1])
    return cutoffs, modes


def first_mode_round_duct(a, c0, bc="rigid"):
    """
    First higher-order cutoff mode for a circular duct (excluding plane wave).
    """
    if a <= 0 or c0 <= 0:
        raise ValueError("a and c0 must be positive")
    if bc == "rigid":
        x11 = float(jnp_zeros(1, 1)[0])
        return ((1, 1), (c0 / (2.0 * np.pi)) * (x11 / a))
    if bc == "soft":
        x01 = float(jn_zeros(0, 1)[0])
        return ((0, 1), (c0 / (2.0 * np.pi)) * (x01 / a))
    raise ValueError("bc must be 'rigid' or 'soft'")

