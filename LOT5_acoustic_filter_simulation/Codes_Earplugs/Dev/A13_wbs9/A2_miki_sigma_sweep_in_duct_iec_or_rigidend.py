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

from toolkitsd.acoustmm import CylindricalDuct, IEC711Coupler, MikiLayer, ViscothermalDuct


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    length_outlet = 5e-3
    r_duct = 5e-3
    cavity_each_side = 10e-3
    p_incident = 1.0

    thickness_values = [
        5e-3,
        10e-3,
        20e-3,
    ]
    sigma_values = [
        5.0e3,
        1.0e4,
        2.0e4,
        5.0e4,
        1.0e5,
    ]

    area = np.pi * r_duct**2
    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 200)
    omega = 2.0 * np.pi * freqs
    z0_in = np.full(omega.shape, RHO0 * C0 / area + 0j, dtype=np.complex128)

    inlet = CylindricalDuct(radius=r_duct, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_duct, length=length_outlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=cavity_each_side, c0=C0, rho0=RHO0)
    air_system = inlet + halfduct + halfduct + outlet

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    z_rigid = np.full(omega.shape, np.inf + 0.0j, dtype=np.complex128)
    p_end_air_iec711 = air_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_air_rigid = air_system.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]

    n_thickness = len(thickness_values)
    fig, axes = plt.subplots(2, n_thickness, figsize=(5.5 * n_thickness, 8), sharex=True, constrained_layout=True)
    if n_thickness == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, foam_thickness in enumerate(thickness_values):
        il_iec711_curves = {}
        il_rigid_curves = {}

        for sigma in sigma_values:
            foam = MikiLayer(
                sigma=sigma,
                length=foam_thickness,
                area=area,
                rho0=RHO0,
                c0=C0,
                name=f"Miki sigma={sigma:.3e}",
            )
            system = inlet + halfduct + foam + halfduct + outlet

            p_end_filter_iec711 = system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
            p_end_filter_rigid = system.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]

            il_iec711_curves[sigma] = 20.0 * np.log10(
                np.maximum(np.abs(p_end_air_iec711 / p_end_filter_iec711), np.finfo(float).tiny)
            )
            il_rigid_curves[sigma] = 20.0 * np.log10(
                np.maximum(np.abs(p_end_air_rigid / p_end_filter_rigid), np.finfo(float).tiny)
            )

        for sigma, il_db in il_iec711_curves.items():
            axes[0, col].semilogx(freqs, il_db, linewidth=2.0, label=fr"$\sigma={sigma:.1e}$")
        axes[0, col].set_ylabel("IL [dB]")
        axes[0, col].set_title(f"IEC711, thickness = {foam_thickness*1e3:.1f} mm")
        axes[0, col].grid(True, which="both", alpha=0.3)
        axes[0, col].legend(loc="best")

        for sigma, il_db in il_rigid_curves.items():
            axes[1, col].semilogx(freqs, il_db, linewidth=2.0, label=fr"$\sigma={sigma:.1e}$")
        axes[1, col].set_xlabel("Frequency [Hz]")
        axes[1, col].set_ylabel("IL [dB]")
        axes[1, col].set_title(f"Rigid load, thickness = {foam_thickness*1e3:.1f} mm")
        axes[1, col].grid(True, which="both", alpha=0.3)
        axes[1, col].legend(loc="best")

    plt.show()
