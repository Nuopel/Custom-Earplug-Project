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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import CylindricalDuct, ViscothermalDuct


C0 = 343.0
RHO0 = 1.2
ETA0 = 1.839e-5
P_ATM = 101325.0
PR = 0.71
RADIUS = 1.0e-3 / np.pi**0.5
LENGTH = 2.0e-2
LENGTH_TAMPON = 1.0e-4
P_INCIDENT = 1.0
FEM_FILE = HERE / "fem_rslt" / "rslt_duct_circular_with_loss.txt"





if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(3000.0), 100)
    omega = 2.0 * np.pi * freqs
    area = np.pi * RADIUS**2
    total_length = 2.0 * LENGTH_TAMPON + LENGTH
    z_w = np.full(omega.shape, RHO0 * C0 / area + 0j, dtype=np.complex128)
    k_w = omega / C0

    duct_in = CylindricalDuct(radius=RADIUS, length=LENGTH_TAMPON, c0=C0, rho0=RHO0)
    duct_mid = ViscothermalDuct(radius=RADIUS, length=LENGTH, c0=C0, rho0=RHO0, eta0=ETA0, P0=P_ATM, Pr=PR)
    duct_out = CylindricalDuct(radius=RADIUS, length=LENGTH_TAMPON, c0=C0, rho0=RHO0)

    system = duct_in + duct_mid + duct_out
    matrix_tmm = system.matrix(omega)
    p_end_rigid, p_end_open = compute_end_pressures(system, omega, z_w, p_incident=P_INCIDENT)

    freqs_fem, omega_fem, z_w_fem, k_w_fem, s11, s21, matrix_fem, fem_element = build_fem_element_from_sparameters(
        FEM_FILE,
        area,
        RHO0,
        C0,
    )
    p_end_rigid_fem, p_end_open_fem = compute_end_pressures(
        fem_element,
        omega_fem,
        z_w_fem,
        p_incident=P_INCIDENT,
    )

    r_tmm, t_tmm, a_tmm = system.reflection_transmission_absorption(Z_c=z_w, omega=omega, k_ref=k_w, length=total_length)
    r_fem_tm, t_fem_tm, a_fem_tm = fem_element.reflection_transmission_absorption(
        Z_c=z_w_fem,
        omega=omega_fem,
        k_ref=k_w_fem,
        length=total_length,
    )
    r_fem_s, t_fem_s, a_fem_s = compute_rta_from_sparameters(s11, s21)

    plot_matrix_comparison(
        freqs,
        matrix_tmm,
        freqs_fem,
        matrix_fem,
        mode="abs_phase",
        title_prefix="Circular duct with losses: transfer matrix comparison",
    )
    plt.show()

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

    axs = plot_rta_comparison(
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
    axs[1].set_ylim([0,0.1])

    plt.show()
