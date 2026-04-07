from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ACOUSTMM_ROOT = HERE.parents[2]
REFACTOR_ROOT = HERE.parents[3]
candidate_paths = [ACOUSTMM_ROOT / "src"]
candidate_paths.extend(sorted(REFACTOR_ROOT.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from A0_minimal_lined_rectangular_duct import (
    LinedRectangularDuct,
    calculate_kz_one_sided_rectangular_lined_duct,
)
from toolkitsd.acoustmm import FrozenMatrixElement, RectangularDuct
from toolkitsd.porous import JCAMaterial, JCAModel


def load_fem_sparameters(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs = []
    s11 = []
    s21 = []
    s12 = []
    s22 = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        freqs.append(float(parts[0]))
        s11.append(complex(parts[1].replace("i", "j")))
        s21.append(complex(parts[2].replace("i", "j")))
        s22.append(complex(parts[3].replace("i", "j")))
        s12.append(complex(parts[4].replace("i", "j")))
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(s11, dtype=np.complex128),
        np.asarray(s21, dtype=np.complex128),
        np.asarray(s12, dtype=np.complex128),
        np.asarray(s22, dtype=np.complex128),
    )


def abcd_from_sparameters_unequal_z0(
    s11: np.ndarray,
    s21: np.ndarray,
    s12: np.ndarray,
    s22: np.ndarray,
    z01: np.ndarray | complex | float,
    z02: np.ndarray | complex | float,
) -> np.ndarray:
    z01 = np.asarray(z01, dtype=np.complex128)
    z02 = np.asarray(z02, dtype=np.complex128)

    denom = 2.0 * s21
    matrix = np.empty((s11.size, 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = ((1.0 + s11) * (1.0 - s22) + s12 * s21) / denom
    matrix[:, 0, 1] = z02 * ((1.0 + s11) * (1.0 + s22) - s12 * s21) / denom
    matrix[:, 1, 0] = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / (z01 * denom)
    matrix[:, 1, 1] = (z02 / z01) * ((1.0 - s11) * (1.0 + s22) + s12 * s21) / denom
    return matrix


def build_fem_element_from_sparameters(
    fem_file: Path,
    *,
    area_in: float,
    area_out: float,
    rho0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, FrozenMatrixElement]:
    freqs_fem, s11, s21, s12, s22 = load_fem_sparameters(fem_file)
    omega_fem = 2.0 * np.pi * freqs_fem
    z01 = np.full(omega_fem.shape, rho0 * c0 / area_in + 0j, dtype=np.complex128)
    z02 = np.full(omega_fem.shape, rho0 * c0 / area_out + 0j, dtype=np.complex128)
    matrix_fem = abcd_from_sparameters_unequal_z0(s11, s21, s12, s22, z01, z02)
    return freqs_fem, omega_fem, FrozenMatrixElement.from_pu(matrix_fem)


def plot_tl_comparison(
    freqs_hz: np.ndarray,
    tl_tmm_db: np.ndarray,
    tl_fem_zs_db: np.ndarray,
    tl_fem_porous_db: np.ndarray | None = None,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.semilogx(freqs_hz, tl_tmm_db, linewidth=2.0, label="TMM lined rectangular duct")
    plt.semilogx(freqs_hz, tl_fem_zs_db, "--", linewidth=2.0, label="FEM decascaded Zs wall")
    if tl_fem_porous_db is not None:
        plt.semilogx(freqs_hz, tl_fem_porous_db, ":", linewidth=2.0, label="FEM decascaded porous layer")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Transmission Loss [dB]")
    plt.title("Rectangular lined duct: FEM vs TMM")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="best")


def plot_kz_diagnostics(freqs_hz: np.ndarray, kz: np.ndarray, cutoff_hz: float) -> None:
    alpha = -np.imag(kz)
    beta = np.real(kz)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].semilogx(freqs_hz, beta, linewidth=2.0, label=r"$\beta = \Re(k_z)$")
    axes[0].axvline(cutoff_hz, color="k", linestyle="--", linewidth=1.5, label=f"Rigid cutoff = {cutoff_hz:.0f} Hz")
    axes[0].set_ylabel(r"$\beta$ [rad/m]")
    axes[0].set_title("Axial wavenumber diagnostic")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, alpha, linewidth=2.0, label=r"$\alpha = -\Im(k_z)$")
    axes[1].axvline(cutoff_hz, color="k", linestyle="--", linewidth=1.5, label=f"Rigid cutoff = {cutoff_hz:.0f} Hz")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$\alpha$ [Np/m]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")


def plot_matrix_comparison(
    freqs_hz: np.ndarray,
    matrix_tmm: np.ndarray,
    matrix_fem_zs: np.ndarray,
    *,
    matrix_fem_porous: np.ndarray | None = None,
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
            values_tmm = matrix_tmm[:, row, col]
            values_fem_zs = matrix_fem_zs[:, row, col]

            if mode == "real_imag":
                top_tmm = np.real(values_tmm)
                bottom_tmm = np.imag(values_tmm)
                top_fem_zs = np.real(values_fem_zs)
                bottom_fem_zs = np.imag(values_fem_zs)
                top_ylabel = f"Re({label})"
                bottom_ylabel = f"Im({label})"
                if matrix_fem_porous is not None:
                    values_fem_porous = matrix_fem_porous[:, row, col]
                    top_fem_porous = np.real(values_fem_porous)
                    bottom_fem_porous = np.imag(values_fem_porous)
            else:
                top_tmm = np.abs(values_tmm)
                bottom_tmm = np.angle(values_tmm)
                top_fem_zs = np.abs(values_fem_zs)
                bottom_fem_zs = np.angle(values_fem_zs)
                top_ylabel = f"|{label}|"
                bottom_ylabel = f"Phase({label}) [rad]"
                if matrix_fem_porous is not None:
                    values_fem_porous = matrix_fem_porous[:, row, col]
                    top_fem_porous = np.abs(values_fem_porous)
                    bottom_fem_porous = np.angle(values_fem_porous)

            axes[idx_top, col].semilogx(freqs_hz, top_tmm, linewidth=2.0, label="TMM")
            axes[idx_top, col].semilogx(freqs_hz, top_fem_zs, "--", linewidth=2.0, label="FEM Zs wall")
            if matrix_fem_porous is not None:
                axes[idx_top, col].semilogx(
                    freqs_hz,
                    top_fem_porous,
                    ":",
                    linewidth=2.0,
                    label="FEM porous layer",
                )
            axes[idx_bottom, col].semilogx(freqs_hz, bottom_tmm, linewidth=2.0, label="TMM")
            axes[idx_bottom, col].semilogx(freqs_hz, bottom_fem_zs, "--", linewidth=2.0, label="FEM Zs wall")
            if matrix_fem_porous is not None:
                axes[idx_bottom, col].semilogx(
                    freqs_hz,
                    bottom_fem_porous,
                    ":",
                    linewidth=2.0,
                    label="FEM porous layer",
                )
            axes[idx_top, col].set_ylabel(top_ylabel)
            axes[idx_bottom, col].set_ylabel(bottom_ylabel)
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)
            axes[idx_top, col].legend(loc="best")
            if mode == "abs_phase":
                axes[idx_bottom, col].set_ylim([-np.pi, np.pi])

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Rectangular lined duct: matrix comparison (FEM vs TMM)")


if __name__ == "__main__":
    C0 = 344.5
    RHO0 = 1.2

    length_inlet = 5.0e-3
    length_outlet = 5.0e-3
    length_duct = 0.05
    width_x = 0.01
    width_y = 0.01
    rigid_first_mode_cutoff_hz = C0 / (2.0 * max(width_x, width_y))

    fem_file_zs = HERE / "fem_rslt" / "A0_rectduct_with_lining_zsjca.txt"
    fem_file_porous = HERE / "fem_rslt" / "A0_rectduct_with_lining.txt"
    area = width_x * width_y

    freqs_fem, omega_fem, fem_element_zs = build_fem_element_from_sparameters(
        fem_file_zs,
        area_in=area,
        area_out=area,
        rho0=RHO0,
        c0=C0,
    )
    freqs_fem_porous, omega_fem_porous, fem_element_porous = build_fem_element_from_sparameters(
        fem_file_porous,
        area_in=area,
        area_out=area,
        rho0=RHO0,
        c0=C0,
    )
    if not np.allclose(freqs_fem_porous, freqs_fem):
        raise ValueError("Zs-wall and porous-layer FEM files must share the same frequency grid")

    inlet = RectangularDuct(width=width_x, height=width_y, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = RectangularDuct(width=width_x, height=width_y, length=length_outlet, c0=C0, rho0=RHO0)
    decascaded_fem_zs = fem_element_zs.decascade_right(outlet).decascade_left(inlet)
    decascaded_fem_porous = fem_element_porous.decascade_right(outlet).decascade_left(inlet)

    lining_material = JCAMaterial(
        phi=0.93,
        sigma=15000.0,
        tortu=1.10,
        lambda1=6e-5,
        lambdap=1e-4,
        rho0=RHO0,
        c0=C0,
        thickness=0.002,
        name="JCA lining",
    )
    print(f"Rigid rectangular duct first cutoff ≈ {rigid_first_mode_cutoff_hz:.1f} Hz")
    print(f"FEM frequency range: {freqs_fem[0]:.1f} Hz -> {freqs_fem[-1]:.1f} Hz")

    tmm_lined_duct = LinedRectangularDuct(
        width=width_x,
        height=width_y,
        length=length_duct,
        frequencies_hz=freqs_fem,
        wall_mode="one-sided",
        lining_material=lining_material,
        incidence_angle_deg=90.0,
        porous_model=JCAModel(),
        c0=C0,
        rho0=RHO0,
    )

    zc_ref = RHO0 * C0 / area
    matrix_tmm = tmm_lined_duct.matrix(omega_fem)
    matrix_fem_zs = decascaded_fem_zs.matrix(omega_fem)
    matrix_fem_porous = decascaded_fem_porous.matrix(omega_fem)
    tl_tmm_db = tmm_lined_duct.TL(Z_c=zc_ref, omega=omega_fem)
    tl_fem_zs_db = decascaded_fem_zs.TL(Z_c=zc_ref, omega=omega_fem)
    tl_fem_porous_db = decascaded_fem_porous.TL(Z_c=zc_ref, omega=omega_fem)

    # plot_kz_diagnostics(freqs_fem, tmm_lined_duct.kz, rigid_first_mode_cutoff_hz)
    # plt.show()
    plot_matrix_comparison(
        freqs_fem,
        matrix_tmm,
        matrix_fem_zs,
        matrix_fem_porous=matrix_fem_porous,
        mode="abs_phase",
    )
    plt.show()
    plot_tl_comparison(freqs_fem, tl_tmm_db, tl_fem_zs_db, tl_fem_porous_db)
    plt.show()
