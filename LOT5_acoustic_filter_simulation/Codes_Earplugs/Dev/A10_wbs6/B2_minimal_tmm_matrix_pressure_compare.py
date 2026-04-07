"""Minimal WBS6 TMM-only comparison for lossy duct models."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import BLIDuct, CylindricalDuct, ViscothermalDuct


if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(16000.0), 500)
    omega = 2.0 * np.pi * freqs

    radius = 3.75e-3
    length = 6.6e-3
    c0 = 343.0
    rho0 = 1.2
    area = np.pi * radius**2

    p0 = 1.0
    z_rigid_like = np.full(omega.shape, 1.0e18 + 0.0j, dtype=np.complex128)
    z_source = rho0 * c0 / area

    ducts = {
        "ViscothermalDuct": ViscothermalDuct(radius=radius, length=length, c0=c0, rho0=rho0),
        "BLIDuct": BLIDuct(radius=radius, length=length, c0=c0, rho0=rho0),
        "CylindricalDuct": CylindricalDuct(radius=radius, length=length, c0=c0, rho0=rho0),
    }

    model_matrices = {name: duct.matrix(omega) for name, duct in ducts.items()}

    pressures = {}
    for name, duct in ducts.items():
        p_in_slab = duct.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)
        p_end_slab_ptm = duct.p_tm(p_in_slab, z_rigid_like, omega)
        pressures[name] = p_end_slab_ptm

    fig_matrix, axes_matrix = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))
    model_styles = {
        "ViscothermalDuct": {"color": "tab:red", "lw": 2.0},
        "BLIDuct": {"color": "tab:blue", "lw": 2.0},
        "CylindricalDuct": {"color": "tab:gray", "lw": 2.0},
    }

    for row in range(2):
        for col in range(2):
            idx_real = 2 * row
            idx_imag = 2 * row + 1
            ax_real = axes_matrix[idx_real, col]
            ax_imag = axes_matrix[idx_imag, col]
            label = labels[row][col]
            for name, matrix in model_matrices.items():
                style = model_styles[name]
                ax_real.semilogx(freqs, np.real(matrix[:, row, col]), label=name, **style)
                ax_imag.semilogx(freqs, np.imag(matrix[:, row, col]), label=name, **style)
            ax_real.set_ylabel(f"Re({label})")
            ax_imag.set_ylabel(f"Im({label})")
            ax_real.grid(True, which="both", alpha=0.3)
            ax_imag.grid(True, which="both", alpha=0.3)
            ax_real.legend(loc="best")
            ax_imag.legend(loc="best")

    axes_matrix[3, 0].set_xlabel("Frequency [Hz]")
    axes_matrix[3, 1].set_xlabel("Frequency [Hz]")
    fig_matrix.suptitle("Minimal TMM-only duct matrix comparison")
    plt.show()

    fig_pressure, axes_pressure = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    for name, pressure in pressures.items():
        style = model_styles[name]
        axes_pressure[0].semilogx(freqs, np.real(pressure), label=name, **style)
        axes_pressure[1].semilogx(freqs, np.imag(pressure), label=name, **style)

    axes_pressure[0].set_ylabel(r"Re($p_{end}$)")
    axes_pressure[1].set_ylabel(r"Im($p_{end}$)")
    axes_pressure[1].set_xlabel("Frequency [Hz]")
    axes_pressure[0].grid(True, which="both", alpha=0.3)
    axes_pressure[1].grid(True, which="both", alpha=0.3)
    axes_pressure[0].legend(loc="best")
    axes_pressure[1].legend(loc="best")
    fig_pressure.suptitle("Rigid-end pressure comparison")

    plt.show()
