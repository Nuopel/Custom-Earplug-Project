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

from function import build_fem_element_from_sparameters
from toolkitsd.acoustmm import CylindricalDuct, ViscothermalDuct


def plot_matrix_cases_overlay(
    case_data: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    reference_case_name: str,
    reference_tmm: tuple[np.ndarray, np.ndarray] | None = None,
    mode: str = "abs_phase",
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))
    if mode not in {"real_imag", "abs_phase"}:
        raise ValueError("mode must be 'real_imag' or 'abs_phase'")

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            label = labels[row][col]

            for case_name, (freqs, matrix) in case_data.items():
                values = matrix[:, row, col]
                if mode == "real_imag":
                    top_values = np.real(values)
                    bottom_values = np.imag(values)
                    top_ylabel = f"Re({label})"
                    bottom_ylabel = f"Im({label})"
                else:
                    top_values = np.abs(values)
                    bottom_values = np.angle(values)
                    top_ylabel = f"|{label}|"
                    bottom_ylabel = f"Phase({label}) [rad]"

                axes[idx_top, col].semilogx(freqs, top_values, linewidth=1.8, label=case_name)
                axes[idx_bottom, col].semilogx(freqs, bottom_values, linewidth=1.8, label=case_name)

            if reference_tmm is not None:
                freqs_ref, matrix_ref = reference_tmm
                values_ref = matrix_ref[:, row, col]
                if mode == "real_imag":
                    top_ref = np.real(values_ref)
                    bottom_ref = np.imag(values_ref)
                else:
                    top_ref = np.abs(values_ref)
                    bottom_ref = np.angle(values_ref)
                axes[idx_top, col].semilogx(
                    freqs_ref,
                    top_ref,
                    "k--",
                    linewidth=2.2,
                    label=f"{reference_case_name} TMM",
                )
                axes[idx_bottom, col].semilogx(
                    freqs_ref,
                    bottom_ref,
                    "k--",
                    linewidth=2.2,
                    label=f"{reference_case_name} TMM",
                )

            axes[idx_top, col].set_ylabel(top_ylabel)
            axes[idx_bottom, col].set_ylabel(bottom_ylabel)
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)
            if mode == "abs_phase":
                axes[idx_bottom, col].set_ylim([-np.pi, np.pi])
            axes[idx_top, col].legend(loc="best")

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Decascaded FEM slab matrices for all port-load cases")


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5.0e-3
    length_outlet = 5.0e-3
    length_slab = 6.60e-3 + 6.4e-3
    r_tube = 3.5e-3
    r_slab = 3.5e-3

    fem_dir = HERE / "fem_rslt" / "port_load"
    case_files = {
        "Case A: silicone slab filter": fem_dir / "rslt_fem_A0_caseA_silicone_slab_filter_in_duct.txt",
        "Case B: rigid slab filter": fem_dir / "rslt_fem_A0_caseB_rigid_slab_filter_in_duct.txt",
        "Case C: silicone slab": fem_dir / "rslt_fem_A0_caseC_silicone_slab_in_duct.txt",
        "Case D: air slab": fem_dir / "rslt_fem_A0_caseD_air_slab_in_duct.txt",
        "Case E: rigid slab film": fem_dir / "rslt_fem_A0_caseE_rigid_slab_film_in_duct.txt",
        "Case F: silicone slab film": fem_dir / "rslt_fem_A0_caseF_silicone_slab_film_in_duct.txt",
    }

    area_in = np.pi * r_tube**2
    area_out = np.pi * r_tube**2

    inlet = CylindricalDuct(radius=r_tube, length=length_inlet, c0=C0, rho0=RHO0)
    slab = ViscothermalDuct(radius=r_slab, length=length_slab, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_tube, length=length_outlet, c0=C0, rho0=RHO0)

    case_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    reference_tmm: tuple[np.ndarray, np.ndarray] | None = None
    reference_case_name = "Case D: air slab"

    for case_name, fem_file in case_files.items():
        (
            freqs_fem,
            omega_fem,
            _z01_fem,
            _z02_fem,
            _k01_fem,
            _k02_fem,
            _s11,
            _s21,
            _matrix_fem_total,
            fem_element,
        ) = build_fem_element_from_sparameters(
            fem_file,
            area_in,
            area_out,
            RHO0,
            C0,
        )

        matrix_fem = fem_element.decascade_right(outlet).decascade_left(inlet).matrix(omega_fem)
        case_data[case_name] = (freqs_fem, matrix_fem)

        if case_name == "Case D: air slab":
            reference_tmm = (freqs_fem, slab.matrix(omega_fem))

    plot_matrix_cases_overlay(
        case_data,
        reference_case_name=reference_case_name,
        reference_tmm=reference_tmm,
        mode="abs_phase",
    )
    plt.show()
