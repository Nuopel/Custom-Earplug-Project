from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from toolkitsd.acoustmm import FrozenMatrixElement


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
        s12.append(complex(parts[3].replace("i", "j")))
        s22.append(complex(parts[4].replace("i", "j")))
    return (
        np.asarray(freqs, dtype=np.float64),
        np.asarray(s11, dtype=np.complex128),
        np.asarray(s21, dtype=np.complex128),
        np.asarray(s12, dtype=np.complex128),
        np.asarray(s22, dtype=np.complex128),
    )


def abcd_from_sparameters(
    s11: np.ndarray,
    s21: np.ndarray,
    s12: np.ndarray,
    s22: np.ndarray,
    z0: np.ndarray | complex | float,
) -> np.ndarray:
    z0 = np.asarray(z0, dtype=np.complex128)
    denom = 2.0 * s21
    matrix = np.empty((s11.size, 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = ((1.0 + s11) * (1.0 - s22) + s12 * s21) / denom
    matrix[:, 0, 1] = z0 * ((1.0 + s11) * (1.0 + s22) - s12 * s21) / denom
    matrix[:, 1, 0] = ((1.0 - s11) * (1.0 - s22) - s12 * s21) / (z0 * denom)
    matrix[:, 1, 1] = ((1.0 - s11) * (1.0 + s22) + s12 * s21) / denom
    return matrix


def build_fem_element_from_sparameters(
    fem_file: Path,
    area: float,
    rho0: float,
    c0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FrozenMatrixElement]:
    freqs_fem, s11, s21, s12, s22 = load_fem_sparameters(fem_file)
    omega_fem = 2.0 * np.pi * freqs_fem
    z_w_fem = np.full(omega_fem.shape, rho0 * c0 / area + 0j, dtype=np.complex128)
    k_w_fem = omega_fem / c0
    matrix_fem = abcd_from_sparameters(s11, s21, s12, s22, z_w_fem)
    fem_element = FrozenMatrixElement.from_pu(matrix_fem)
    return freqs_fem, omega_fem, z_w_fem, k_w_fem, s11, s21, matrix_fem, fem_element


def compute_end_pressures(system, omega: np.ndarray, z_source: np.ndarray | complex | float, p_incident: float) -> tuple[np.ndarray, np.ndarray]:
    z_rigid = np.full(np.asarray(omega).shape, np.inf + 0.0j, dtype=np.complex128)
    z_open = np.zeros(np.asarray(omega).shape, dtype=np.complex128)
    p_in_rigid = system.p_in_from_incident_wave(p_incident, z_rigid, z_source, omega)
    p_in_open = system.p_in_from_incident_wave(p_incident, z_open, z_source, omega)
    p_end_rigid = system.p_tm(p_in_rigid, z_rigid, omega)
    p_end_open = system.p_tm(p_in_open, z_open, omega)
    return p_end_rigid, p_end_open


def plot_matrix_comparison(
    freqs: np.ndarray,
    matrix_tmm: np.ndarray,
    freqs_fem: np.ndarray,
    matrix_fem: np.ndarray,
    mode: str = "real_imag",
    title_prefix: str = "Transfer matrix comparison",
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))
    if mode not in {"real_imag", "abs_phase"}:
        raise ValueError("mode must be 'real_imag' or 'abs_phase'")

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            ax_top = axes[idx_top, col]
            ax_bottom = axes[idx_bottom, col]
            label = labels[row][col]
            tmm_values = matrix_tmm[:, row, col]
            fem_values = matrix_fem[:, row, col]

            if mode == "real_imag":
                top_tmm = np.real(tmm_values)
                bottom_tmm = np.imag(tmm_values)
                top_fem = np.real(fem_values)
                bottom_fem = np.imag(fem_values)
                top_ylabel = f"Re({label})"
                bottom_ylabel = f"Im({label})"
                top_label_tmm = f"Re({label}) TMM"
                bottom_label_tmm = f"Im({label}) TMM"
                top_label_fem = f"Re({label}) FEM"
                bottom_label_fem = f"Im({label}) FEM"
            else:
                top_tmm = np.abs(tmm_values)
                bottom_tmm = np.angle(tmm_values)
                top_fem = np.abs(fem_values)
                bottom_fem = np.angle(fem_values)
                top_ylabel = f"|{label}|"
                bottom_ylabel = f"Phase({label}) [rad]"
                top_label_tmm = f"|{label}| TMM"
                bottom_label_tmm = f"Phase({label}) TMM"
                top_label_fem = f"|{label}| FEM"
                bottom_label_fem = f"Phase({label}) FEM"

            ax_top.semilogx(freqs, top_tmm, label=top_label_tmm, linewidth=2.0)
            ax_top.semilogx(freqs_fem, top_fem, "--", label=top_label_fem, linewidth=1.8)
            ax_bottom.semilogx(freqs, bottom_tmm, label=bottom_label_tmm, linewidth=2.0)
            ax_bottom.semilogx(freqs_fem, bottom_fem, "--", label=bottom_label_fem, linewidth=1.8)
            ax_top.set_ylabel(top_ylabel)
            ax_bottom.set_ylabel(bottom_ylabel)
            ax_top.grid(True, which="both", alpha=0.3)
            ax_bottom.grid(True, which="both", alpha=0.3)
            ax_top.legend(loc="best")
            ax_bottom.legend(loc="best")
            ax_top.set_xlim([freqs_fem[0], freqs_fem[-1]])
            ax_bottom.set_xlim([freqs_fem[0], freqs_fem[-1]])
            if mode == "abs_phase":
                ax_bottom.set_ylim([-np.pi, np.pi])

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle(f"{title_prefix} ({mode})")



def plot_end_pressures(
    freqs_tmm: np.ndarray,
    p_end_rigid_tmm: np.ndarray,
    p_end_open_tmm: np.ndarray,
    freqs_fem: np.ndarray,
    p_end_rigid_fem: np.ndarray,
    p_end_open_fem: np.ndarray,
    title_prefix: str = "End pressure comparison",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0, 0].semilogx(freqs_tmm, np.real(p_end_rigid_tmm), label="Re(p_end rigid) TMM", linewidth=2.0)
    axes[0, 0].semilogx(freqs_fem, np.real(p_end_rigid_fem), "--", label="Re(p_end rigid) FEM", linewidth=1.8)
    axes[1, 0].semilogx(freqs_tmm, np.imag(p_end_rigid_tmm), label="Im(p_end rigid) TMM", linewidth=2.0)
    axes[1, 0].semilogx(freqs_fem, np.imag(p_end_rigid_fem), "--", label="Im(p_end rigid) FEM", linewidth=1.8)
    axes[0, 1].semilogx(freqs_tmm, np.real(p_end_open_tmm), label="Re(p_end open) TMM", linewidth=2.0)
    axes[0, 1].semilogx(freqs_fem, np.real(p_end_open_fem), "--", label="Re(p_end open) FEM", linewidth=1.8)
    axes[1, 1].semilogx(freqs_tmm, np.imag(p_end_open_tmm), label="Im(p_end open) TMM", linewidth=2.0)
    axes[1, 1].semilogx(freqs_fem, np.imag(p_end_open_fem), "--", label="Im(p_end open) FEM", linewidth=1.8)

    axes[0, 0].set_title("Rigid load")
    axes[0, 1].set_title("Open load")
    axes[0, 0].set_ylabel("Real part")
    axes[1, 0].set_ylabel("Imaginary part")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(title_prefix)


def compute_rta_from_sparameters(
    s11: np.ndarray,
    s21: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reflection = s11
    transmission = s21
    absorption = 1.0 - np.abs(reflection) ** 2 - np.abs(transmission) ** 2
    return reflection, transmission, absorption


def plot_rta_comparison(
    freqs_tmm: np.ndarray,
    r_tmm: np.ndarray,
    t_tmm: np.ndarray,
    a_tmm: np.ndarray,
    freqs_fem: np.ndarray,
    r_fem_s: np.ndarray,
    t_fem_s: np.ndarray,
    a_fem_s: np.ndarray,
    r_fem_tm: np.ndarray,
    t_fem_tm: np.ndarray,
    a_fem_tm: np.ndarray,
    title_prefix: str = 'R/T/A comparison',
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axs[0].semilogx(freqs_tmm, np.abs(r_tmm), label='|R| TMM', linewidth=2.0)
    axs[0].semilogx(freqs_fem, np.abs(r_fem_s), '--', label='|R| FEM S', linewidth=1.8)
    axs[0].semilogx(freqs_fem, np.abs(r_fem_tm), ':', label='|R| FEM TM', linewidth=1.8)
    axs[0].semilogx(freqs_tmm, np.abs(t_tmm), label='|T| TMM', linewidth=2.0)
    axs[0].semilogx(freqs_fem, np.abs(t_fem_s), '--', label='|T| FEM S', linewidth=1.8)
    axs[0].semilogx(freqs_fem, np.abs(t_fem_tm), ':', label='|T| FEM TM', linewidth=1.8)
    axs[0].set_ylabel('|R|, |T|')
    axs[0].set_ylim([0.0, 1.0])

    axs[1].semilogx(freqs_tmm, a_tmm, label='A TMM', linewidth=2.0)
    axs[1].semilogx(freqs_fem, a_fem_s, '--', label='A FEM S', linewidth=1.8)
    axs[1].semilogx(freqs_fem, a_fem_tm, ':', label='A FEM TM', linewidth=1.8)
    axs[1].set_ylabel('A')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylim([0.0, 1.0])

    for ax in axs:
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim([freqs_fem[0], freqs_fem[-1]])
    fig.suptitle(title_prefix)
    return axs
