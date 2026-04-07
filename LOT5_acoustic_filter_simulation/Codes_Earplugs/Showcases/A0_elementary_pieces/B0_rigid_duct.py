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
)


def plot_loaded_pressure(
    freqs_hz: np.ndarray,
    p_end_iec711: np.ndarray,
    *,
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5), constrained_layout=True)

    ax.semilogx(
        freqs_hz,
        20.0 * np.log10(np.maximum(np.abs(p_end_iec711), np.finfo(float).tiny)),
        lw=2.0,
        label="End pressure, IEC711 load",
    )
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    ax.set_title(f"End pressure, {title_suffix}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 6.4e-3
    length_outlet = 6.4e-3
    length_duct = 6.6e-3

    r_inlet = 3.5e-3
    r_outlet = r_inlet
    r_duct = r_inlet
    total_length = length_inlet+length_outlet+length_duct
    p_incident = 1.0

    freqs = np.logspace(np.log10(50.0), np.log10(6000.0), 100)
    omega = 2.0 * np.pi * freqs
    k0 = omega / C0

    s_in = np.pi * r_inlet**2
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    duct = CylindricalDuct(radius=r_duct, length=length_duct, c0=C0, rho0=RHO0)

    system_air_only = inlet + duct + outlet

    r_tmm, t_tmm, a_tmm = system_air_only.reflection_transmission_absorption_unequal_refs(
        Z_in=z0_in,
        Z_out=z0_in,
        omega=omega,
        k_ref=k0,
        length=total_length,
    )

    system_air_only.plot.rta(freq=freqs, R=r_tmm, T=t_tmm, A=a_tmm)
    plt.show()

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    plot_loaded_pressure(
        freqs,
        p_end_iec711_system_air_only,
        title_suffix="rigid duct with IEC711 load",
    )
