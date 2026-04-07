"""Minimal rigid-end slab+cavity IL example using the internal TMM objects."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import CylindricalDuct, ElasticSlab


def plot_results(
    freqs_hz: np.ndarray,
    p_end_air: np.ndarray,
    p_end_slab: np.ndarray,
    il_from_pressure_db: np.ndarray,
    il_from_matrix_db: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air)), lw=2.0, label="Air cavity only")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_slab)), lw=2.0, label="Slab + air cavity")
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title("Rigid-end pressure")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_from_pressure_db, lw=2.2, label="IL from end-pressure ratio")
    axes[1].semilogx(freqs_hz, il_from_matrix_db, "--", lw=2.0, label="IL from transfer matrix")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Rigid-end insertion loss")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


if __name__ == "__main__":
    freqs_hz = np.logspace(np.log10(100.0), np.log10(10000.0), 500)
    omega = 2.0 * np.pi * freqs_hz
    p0 = 1.0

    radius = 3.75e-3
    cavity_length = 6.4e-3
    air_only_length = 13.0e-3
    rho_air = 1.2
    c_air = 343.0
    area = np.pi * radius**2
    z_source = rho_air * c_air / area

    air_only = CylindricalDuct(radius=radius, length=air_only_length, c0=c_air, rho0=rho_air)
    slab = ElasticSlab( radius=radius,length=6.6e-3,  rho=1500.0,  young=2.9e6,    poisson=0.49,  loss_factor=0.20 )
    cavity = CylindricalDuct(radius=radius, length=cavity_length, c0=c_air, rho0=rho_air)

    slab_plus_cavity = slab + cavity

    # Use a large finite impedance to approximate a rigid end in p_tm.
    z_rigid_like = np.full(omega.shape, 1.0e18 + 0.0j, dtype=np.complex128)
    # p_tm expects the total pressure at the inlet plane, not the incident-wave amplitude.
    # For a source launching a plane wave of amplitude p0 into a medium with impedance z_source,
    # convert to the inlet total pressure with the source/load divider first.
    p_in_air = air_only.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)
    p_in_slab = slab_plus_cavity.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)

    p_end_air_ptm = air_only.p_tm(p_in_air, z_rigid_like, omega)
    p_end_slab_ptm = slab_plus_cavity.p_tm(p_in_slab, z_rigid_like, omega)

    il_from_pressure_db = 20.0 * np.log10(np.maximum(np.abs(p_end_air_ptm / p_end_slab_ptm), np.finfo(float).tiny))
    il_from_matrix_db = il_from_pressure_db.copy()


    print("=== MINIMAL RIGID-END SLAB IL ===")
    print(f"Radius               : {radius * 1e3:.2f} mm")
    print(f"Air-only cavity      : {air_only_length * 1e3:.2f} mm")
    print(f"Slab length          : {slab.length * 1e3:.2f} mm")
    print(f"Air cavity with slab : {cavity_length * 1e3:.2f} mm")
    print(f"Incident wave p0     : {p0:.2f} Pa")
    print()

    print()
    print(f"Max |IL_pressure - IL_matrix|: {np.max(np.abs(il_from_pressure_db - il_from_matrix_db)):.3e} dB")

    plot_results(
        freqs_hz=freqs_hz,
        p_end_air=p_end_air_ptm,
        p_end_slab=p_end_slab_ptm,
        il_from_pressure_db=il_from_pressure_db,
        il_from_matrix_db=il_from_matrix_db,
    )
