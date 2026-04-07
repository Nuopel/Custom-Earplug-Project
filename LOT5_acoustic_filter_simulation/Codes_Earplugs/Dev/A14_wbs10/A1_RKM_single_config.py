from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import (
    CylindricalDuct,
    GenericFilmSeriesImpedance,
    IEC711Coupler,
    ImpedanceJunction,
    ViscothermalDuct,
)


def plot_single_config_results(
    freqs_hz: np.ndarray,
    p_end_air_rigid: np.ndarray,
    p_end_filter_rigid: np.ndarray,
    il_rigid_db: np.ndarray,
    p_end_air_iec711: np.ndarray,
    p_end_filter_iec711: np.ndarray,
    il_iec711_db: np.ndarray,
    *,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air_rigid)), lw=2.0, label="Air cavity only, rigid")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_filter_rigid)), lw=2.0, label="Filter + air cavity, rigid")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air_iec711)), "--", lw=2.0, label="Air cavity only, IEC711")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_filter_iec711)), "--", lw=2.0, label="Filter + air cavity, IEC711")
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title(f"End pressure, {title_suffix}")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_rigid_db, lw=2.2, label="IL, rigid")
    axes[1].semilogx(freqs_hz, il_iec711_db, "--", lw=2.2, label="IL, IEC711")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Insertion loss from end-pressure ratio")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_ylim([-10,50])
    plt.show()


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 6.4e-3
    r_inlet = 3.5e-3
    r_duct = 1e-3
    length_duct = 6.6e-3

    p_incident = 1.0

    # Single R/K/M set to tweak manually.
    film_resistance = 0
    film_mass = 0
    film_stiffness = 4.e11

    freqs = np.logspace(np.log10(50.0), np.log10(6000.0), 100)
    omega = 2.0 * np.pi * freqs

    s_in = np.pi * r_inlet**2
    s_duct = np.pi * r_duct**2
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=length_duct / 2.0, c0=C0, rho0=RHO0)
    halfductair = ViscothermalDuct(radius=r_inlet, length=length_duct / 2.0, c0=C0, rho0=RHO0)
    junction_reduction = ImpedanceJunction(s_in, s_duct, rho0=RHO0, end_correction=True)
    junction_expansion = ImpedanceJunction(s_duct, s_in, rho0=RHO0, end_correction=True)
    film = GenericFilmSeriesImpedance(
        resistance=film_resistance,
        mass=film_mass,
        stiffness=film_stiffness,
    )

    system_filter = inlet + junction_reduction + halfduct + film + halfduct + junction_expansion + outlet
    system_air_only = inlet + halfductair + halfductair + outlet

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    z_rigid = np.full(np.asarray(omega).shape, np.inf + 0.0j, dtype=np.complex128)

    p_end_rigid_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]
    p_end_rigid_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]

    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_iec711_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    il_rigid_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_rigid_system_air_only / p_end_rigid_system_filter), np.finfo(float).tiny)
    )
    il_iec711_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_system_filter), np.finfo(float).tiny)
    )

    title_suffix = f"R={film_resistance:.2e}, M={film_mass:.2e}, K={film_stiffness:.2e}"
    plot_single_config_results(
        freqs,
        p_end_rigid_system_air_only,
        p_end_rigid_system_filter,
        il_rigid_db,
        p_end_iec711_system_air_only,
        p_end_iec711_system_filter,
        il_iec711_db,
        title_suffix=title_suffix,
    )
