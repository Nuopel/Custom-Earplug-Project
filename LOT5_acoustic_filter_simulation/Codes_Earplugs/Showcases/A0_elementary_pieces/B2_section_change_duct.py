from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
candidate_paths.extend(sorted((ROOT / "toolkitsd").glob("Toolkitsd_*/src")))
candidate_paths.extend(sorted((ROOT.parent / "toolkitsd").glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import (
    CylindricalDuct,
    IEC711Coupler,
    ViscothermalDuct,
    ImpedanceJunction,
)


def plot_pressure_and_il_results(
    freqs_hz: np.ndarray,
    p_end_iec711_straight: np.ndarray,
    p_end_iec711_section_change: np.ndarray,
    il_iec711_db: np.ndarray,
    *,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(
        freqs_hz,
        20.0 * np.log10(np.maximum(np.abs(p_end_iec711_straight), np.finfo(float).tiny)),
        lw=2.0,
        label="Straight duct, IEC711 load",
    )
    axes[0].semilogx(
        freqs_hz,
        20.0 * np.log10(np.maximum(np.abs(p_end_iec711_section_change), np.finfo(float).tiny)),
        lw=2.0,
        label="Section-change duct, IEC711 load",
    )
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title(f"End pressure, {title_suffix}")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_iec711_db, lw=2.0, label="IL")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title(f"Insertion loss, {title_suffix}")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 2e-3
    length_outlet = 6.4e-3
    length_duct = 6.6e-3

    r_inlet = 3.5e-3
    r_outlet = r_inlet
    r_duct = 0.8e-3
    total_length = length_inlet + length_outlet + length_duct
    p_incident = 1.0

    freqs = np.logspace(np.log10(50.0), np.log10(10000.0), 100)
    omega = 2.0 * np.pi * freqs
    k0 = omega / C0

    s_in = np.pi * r_inlet**2
    s_duct = np.pi * r_duct**2
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)
    z0_out = z0_in

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    duct = ViscothermalDuct(radius=r_duct, length=length_duct, c0=C0, rho0=RHO0)
    straight_duct = CylindricalDuct(radius=r_inlet, length=length_duct, c0=C0, rho0=RHO0)

    section_junction_expansion = ImpedanceJunction(s_in, s_duct, rho0=RHO0, end_correction=True)
    section_junction_reduction = ImpedanceJunction(s_duct, s_in, rho0=RHO0, end_correction=True)

    system_air_only = inlet + straight_duct + outlet
    system_filter = inlet + section_junction_expansion + duct + section_junction_reduction + outlet

    r_tmm, t_tmm, a_tmm = system_filter.reflection_transmission_absorption_unequal_refs(
        Z_in=z0_in,
        Z_out=z0_out,
        omega=omega,
        k_ref=k0,
        length=total_length,
    )

    system_filter.plot.rta(freq=freqs, R=r_tmm, T=t_tmm, A=a_tmm)
    plt.show()

    # Compute IEC-loaded pressure and IL.

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_iec711_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    il_iec711_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_system_filter), np.finfo(float).tiny)
    )

    plot_pressure_and_il_results(
        freqs,
        p_end_iec711_system_air_only,
        p_end_iec711_system_filter,
        il_iec711_db,
        title_suffix="section-change duct",
    )
