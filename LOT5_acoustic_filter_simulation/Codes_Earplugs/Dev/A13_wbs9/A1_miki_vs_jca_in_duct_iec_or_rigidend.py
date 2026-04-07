from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np

from function import plot_matrix_comparison
from toolkitsd.acoustmm import CylindricalDuct, IEC711Coupler, JCALayer, MikiLayer, ViscothermalDuct
from toolkitsd.porous import PorousMaterial


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    r_inlet = 5e-3
    r_outlet = r_inlet
    length_outlet = length_inlet

    r_duct = r_inlet
    length_duct = 2.0e-2
    porous_thickness = 10e-3

    p_incident = 1.0

    s_in = np.pi * r_inlet**2
    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 200)
    omega = 2.0 * np.pi * freqs
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)

    mat = PorousMaterial.get_material_preset("melamine_cttm", rho0=RHO0, c0=C0)
    print(mat)

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=length_duct / 2.0, c0=C0, rho0=RHO0)

    jca_layer = JCALayer(
        phi=mat.phi,
        sigma=mat.sigma,
        alpha_inf=mat.tortu,
        lambda_v=mat.lambda1,
        lambda_t=mat.lambdap,
        length=porous_thickness,
        area=s_in,
        rho0=RHO0,
        c0=C0,
        name="JCA layer",
    )
    miki_layer = MikiLayer(
        sigma=mat.sigma,
        length=porous_thickness,
        area=s_in,
        rho0=RHO0,
        c0=C0,
        name="Miki layer",
    )

    jca_system = inlet + halfduct + jca_layer + halfduct + outlet
    miki_system = inlet + halfduct + miki_layer + halfduct + outlet
    air_system = inlet + halfduct + halfduct + outlet

    matrix_jca = jca_system.matrix(omega)
    matrix_miki = miki_system.matrix(omega)
    plot_matrix_comparison(
        freqs,
        matrix_jca,
        freqs,
        matrix_miki,
        mode="abs_phase",
        title_prefix="JCA vs Miki porous slab in duct",
    )
    plt.show()

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    p_end_air = air_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_jca = jca_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_miki = miki_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    il_jca_db = 20.0 * np.log10(np.maximum(np.abs(p_end_air / p_end_jca), np.finfo(float).tiny))
    il_miki_db = 20.0 * np.log10(np.maximum(np.abs(p_end_air / p_end_miki), np.finfo(float).tiny))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax.semilogx(freqs, il_jca_db, linewidth=2.2, label="JCA layer")
    ax.semilogx(freqs, il_miki_db, "--", linewidth=2.0, label="Miki layer")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("IEC711 insertion loss: JCA vs Miki porous slab")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()
