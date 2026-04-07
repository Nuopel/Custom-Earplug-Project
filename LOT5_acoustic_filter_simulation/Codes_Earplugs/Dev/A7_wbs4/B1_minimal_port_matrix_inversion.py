"""Minimal WBS 2 example for decascade by `-` and explicit decascade methods."""

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


def plot_matrix_comparison(
    freqs_hz: np.ndarray,
    true_matrix: np.ndarray,
    recovered_matrix: np.ndarray,
    title: str,
) -> None:
    labels = [["A", "B"], ["C", "D"]]
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    fig.suptitle(title)

    for row in range(2):
        for col in range(2):
            idx_row = row
            idx_col = col
            entry_label = labels[row][col]

            ax_real = axes[2 * row, col]
            ax_imag = axes[2 * row + 1, col]

            ax_real.semilogx(
                freqs_hz,
                np.real(true_matrix[:, idx_row, idx_col]),
                lw=2.0,
                label=f"True {entry_label}",
            )
            ax_real.semilogx(
                freqs_hz,
                np.real(recovered_matrix[:, idx_row, idx_col]),
                "--",
                lw=2.0,
                label=f"Recovered {entry_label}",
            )
            ax_real.set_ylabel(f"Re({entry_label})")
            ax_real.grid(True, which="both", alpha=0.3)
            ax_real.legend(loc="best")

            ax_imag.semilogx(
                freqs_hz,
                np.imag(true_matrix[:, idx_row, idx_col]),
                lw=2.0,
                label=f"True {entry_label}",
            )
            ax_imag.semilogx(
                freqs_hz,
                np.imag(recovered_matrix[:, idx_row, idx_col]),
                "--",
                lw=2.0,
                label=f"Recovered {entry_label}",
            )
            ax_imag.set_ylabel(f"Im({entry_label})")
            ax_imag.grid(True, which="both", alpha=0.3)
            ax_imag.legend(loc="best")

    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    plt.show()


if __name__ == "__main__":
    freqs_hz = np.logspace(np.log10(100.0), np.log10(10000.0), 500)
    omega = 2.0 * np.pi * freqs_hz

    radius = 3.75e-3
    air_length = 6.4e-3
    slab_length = 6.6e-3
    rho_air = 1.2
    c_air = 343.0

    air = CylindricalDuct(radius=radius, length=air_length, c0=c_air, rho0=rho_air)
    slab = ElasticSlab(
        radius=radius,
        length=slab_length,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )

    t_air = air.matrix(omega)
    t_slab = slab.matrix(omega)
    recovered_air_from_air_air = ((air + air) - air).matrix(omega)
    recovered_slab_from_slab_air = (slab + air).decascade_right(air).matrix(omega)
    recovered_air_from_air_slab = (slab + air).decascade_left(slab).matrix(omega)

    cases = [
        ("air + air -> recover first air with `-`", t_air, recovered_air_from_air_air),
        ("slab + air -> recover slab with decascade_right()", t_slab, recovered_slab_from_slab_air),
        ("air + slab -> recover slab-side remainder with decascade_left()", t_air, recovered_air_from_air_slab),
    ]

    print("=== WBS 2 MINIMAL PORT-MATRIX INVERSION ===")
    print(f"Radius      : {radius * 1e3:.2f} mm")
    print(f"Air length  : {air_length * 1e3:.2f} mm")
    print(f"Slab length : {slab_length * 1e3:.2f} mm")
    print()

    entry_names = [("A", (0, 0)), ("B", (0, 1)), ("C", (1, 0)), ("D", (1, 1))]
    tiny = np.finfo(float).tiny

    for case_name, expected_matrix, recovered_matrix in cases:
        print(case_name)
        for entry_name, (i, j) in entry_names:
            diff = recovered_matrix[:, i, j] - expected_matrix[:, i, j]
            ref = np.maximum(np.abs(expected_matrix[:, i, j]), tiny)
            max_abs_err = np.max(np.abs(diff))
            max_rel_err = np.max(np.abs(diff) / ref)
            print(f"  {entry_name}: max abs err = {max_abs_err:.3e}, max rel err = {max_rel_err:.3e}")
        print()

    plot_matrix_comparison(
        freqs_hz=freqs_hz,
        true_matrix=t_air,
        recovered_matrix=recovered_air_from_air_air,
        title="Air + Air: true first-air matrix vs recovered with (air + air) - air",
    )
    plot_matrix_comparison(
        freqs_hz=freqs_hz,
        true_matrix=t_slab,
        recovered_matrix=recovered_slab_from_slab_air,
        title="Slab + Air: true slab matrix vs recovered with decascade_right(air)",
    )
    plot_matrix_comparison(
        freqs_hz=freqs_hz,
        true_matrix=t_air,
        recovered_matrix=recovered_air_from_air_slab,
        title="Air + Slab: true air matrix vs recovered with decascade_left(air)",
    )
