from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from function import (
    build_fem_element_from_sparameters,
    compute_end_pressures,
    compute_rta_from_sparameters,
    plot_end_pressures,
    plot_matrix_comparison,
    plot_rta_comparison,
)
from toolkitsd.acoustmm import FrozenMatrixElement,ViscothermalDuct

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import CylindricalDuct, ConicalDuct


def successive_cone_approx(
    r1: float,
    r2: float,
    length: float,
    n_sub: int,
    *,
    c0: float,
    rho0: float,
):
    radii = np.linspace(r1, r2, n_sub + 1)
    r_mid = 0.5 * (radii[:-1] + radii[1:])
    sub_length = length / n_sub
    segments = [ViscothermalDuct(radius=r, length=sub_length, c0=c0, rho0=rho0) for r in r_mid]
    return sum(segments)



if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    LENGTH = 2.0e-2
    LENGTH_TAMPON = 5.0e-3
    r1 = 1e-3
    r2 = 2e-2
    P_INCIDENT = 1.0
    FEM_FILE = HERE / "fem_rslt" / "rslt_fem_cone_loss.txt"

    freqs = np.logspace(np.log10(50.0), np.log10(1000.0), 100)
    omega = 2.0 * np.pi * freqs
    area_in = np.pi * r1 ** 2
    area_out = np.pi * r2 ** 2
    total_length = 2.0 * LENGTH_TAMPON + LENGTH
    z0_in = np.full(omega.shape, RHO0 * C0 / area_in + 0j, dtype=np.complex128)
    z0_out = np.full(omega.shape, RHO0 * C0 / area_out + 0j, dtype=np.complex128)
    k0 = omega / C0

    duct_in = CylindricalDuct(radius=r1, length=LENGTH_TAMPON, c0=C0, rho0=RHO0)
    duct_out = CylindricalDuct(radius=r2, length=LENGTH_TAMPON, c0=C0, rho0=RHO0)


    cone_discret = successive_cone_approx(r1, r2, LENGTH, n_sub=100, c0=C0, rho0=RHO0)
    system_discret = duct_in + cone_discret + duct_out

    selected_system =system_discret
    matrix_tmm = selected_system.matrix(omega)
    p_end_rigid, p_end_open = compute_end_pressures(selected_system, omega, z0_in, p_incident=P_INCIDENT)

    freqs_fem, omega_fem, z01_fem, z02_fem, k01_fem, k02_fem, s11, s21, matrix_fem, fem_element = build_fem_element_from_sparameters(
        FEM_FILE,
        area_in,
        area_out,
        RHO0,
        C0,
    )

    plot_matrix_comparison(
        freqs,
        matrix_tmm,
        freqs_fem,
        matrix_fem,
        mode="abs_phase",
        title_prefix="Circular duct with losses: transfer matrix comparison",
    )
    plt.show()

    p_end_rigid_fem, p_end_open_fem = compute_end_pressures(
        fem_element,
        omega_fem,
        z01_fem,
        p_incident=P_INCIDENT,
    )
    #
    r_tmm, t_tmm, a_tmm = selected_system.reflection_transmission_absorption_unequal_refs(Z_in=z0_in,Z_out=z0_out, omega=omega, k_ref=k0, length=total_length)
    r_fem_tm, t_fem_tm, a_fem_tm = fem_element.reflection_transmission_absorption_unequal_refs(
        Z_in=z01_fem,
        Z_out=z02_fem,
        omega=omega_fem,
        k_ref=k02_fem,
        length=total_length,
    )
    r_fem_s, t_fem_s, a_fem_s = compute_rta_from_sparameters(s11, s21, z01_fem, z02_fem)


    plot_end_pressures(
        freqs,
        p_end_rigid,
        p_end_open,
        freqs_fem,
        p_end_rigid_fem,
        p_end_open_fem,
        title_prefix="Circular duct with losses: end pressure comparison",
    )
    plt.show()
    #
    plot_rta_comparison(
        freqs,
        r_tmm,
        t_tmm,
        a_tmm,
        freqs_fem,
        r_fem_s,
        t_fem_s,
        a_fem_s,
        r_fem_tm,
        t_fem_tm,
        a_fem_tm,
        title_prefix="Circular duct with losses: R/T/A comparison",
    )
    plt.show()
