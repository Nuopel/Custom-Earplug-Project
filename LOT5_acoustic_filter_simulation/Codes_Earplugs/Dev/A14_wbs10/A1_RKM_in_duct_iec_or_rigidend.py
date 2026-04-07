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
    neck_to_cavity_end_correction,
)

def successive_cone_approx(
    r1: float,
    r2: float,
    length: float,
    n_sub: int,
    *,
    c0: float,
    rho0: float,
):
    radii = np.linspace(r1, r2, n_sub + 1)
    r_mid = 0.5 * (radii[:-1] + radii[1:])
    sub_length = length / n_sub
    segments = [ViscothermalDuct(radius=r, length=sub_length, c0=c0, rho0=rho0) for r in r_mid]
    return sum(segments)


def plot_transfer_matrix(freqs: np.ndarray, matrix: np.ndarray, title_prefix: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            values = matrix[:, row, col]
            label = labels[row][col]

            axes[idx_top, col].semilogx(freqs, np.abs(values), linewidth=2.0)
            axes[idx_bottom, col].semilogx(freqs, np.angle(values), linewidth=2.0)
            axes[idx_top, col].set_ylabel(f"|{label}|")
            axes[idx_bottom, col].set_ylabel(f"Phase({label}) [rad]")
            axes[idx_bottom, col].set_ylim([-np.pi, np.pi])
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle(title_prefix)


def plot_end_pressures_only(
    freqs: np.ndarray,
    p_end_rigid: np.ndarray,
    p_end_open: np.ndarray,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0, 0].semilogx(freqs, np.real(p_end_rigid), label="Re(p_end rigid)", linewidth=2.0)
    axes[1, 0].semilogx(freqs, np.imag(p_end_rigid), label="Im(p_end rigid)", linewidth=2.0)
    axes[0, 1].semilogx(freqs, np.real(p_end_open), label="Re(p_end open)", linewidth=2.0)
    axes[1, 1].semilogx(freqs, np.imag(p_end_open), label="Im(p_end open)", linewidth=2.0)

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


def plot_rta(freqs: np.ndarray, r: np.ndarray, t: np.ndarray, a: np.ndarray, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].semilogx(freqs, np.abs(r), label="|R|", linewidth=2.0)
    axes[0].semilogx(freqs, np.abs(t), label="|T|", linewidth=2.0)
    axes[0].set_ylabel("|R|, |T|")
    axes[0].set_ylim([0.0, 1.05])

    axes[1].semilogx(freqs, a, label="A", linewidth=2.0)
    axes[1].set_ylabel("A")
    axes[1].set_xlabel("Frequency [Hz]")

    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        ax.set_xlim([freqs[0], freqs[-1]])
    fig.suptitle(title_prefix)


def plot_results(
    freqs_hz: np.ndarray,
    p_end_air_rigid: np.ndarray,
    p_end_filter_rigid: np.ndarray,
    il_rigid_db: np.ndarray,
    p_end_air_iec711: np.ndarray,
    p_end_filter_iec711: np.ndarray,
    il_iec711_db: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air_rigid)), lw=2.0, label="Air cavity only, rigid")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_filter_rigid)), lw=2.0, label="Filter + air cavity, rigid")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air_iec711)), "--", lw=2.0, label="Air cavity only, IEC711")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_filter_iec711)), "--", lw=2.0, label="Filter + air cavity, IEC711")
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title("End pressure")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_rigid_db, lw=2.2, label="IL from end-pressure ratio, rigid")
    axes[1].semilogx(freqs_hz, il_iec711_db, "--", lw=2.2, label="IL from end-pressure ratio, IEC711")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Insertion loss from end-pressure ratio")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


def plot_il_case_groups(
    freqs_hz: np.ndarray,
    case_groups: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, constrained_layout=True)

    for row, (group_name, group_results) in enumerate(case_groups.items()):
        for case_name, result in group_results.items():
            axes[row, 0].semilogx(freqs_hz, result["il_rigid_db"], linewidth=2.0, label=case_name)
            axes[row, 1].semilogx(freqs_hz, result["il_iec711_db"], linewidth=2.0, label=case_name)

        axes[row, 0].set_ylabel("IL [dB]")
        axes[row, 0].set_title(f"{group_name}, rigid end")
        axes[row, 1].set_title(f"{group_name}, IEC711 load")

    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")

    for ax in axes.ravel():
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    # %% IL part

    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    r_inlet = 5e-3
    r_outlet = r_inlet
    length_outlet = length_inlet

    r_duct = r_inlet
    length_duct = 2.0e-2

    p_incident = 1.0
    n_sub = 100

    film_case_groups = {
        "R sweep": {
            "R low": {
                "film_resistance": 1.0e5,
                "film_mass": 8.0e2,
                "film_stiffness": 0.0,
            },
            "R mid": {
                "film_resistance": 1.0e7,
                "film_mass": 8.0e2,
                "film_stiffness": 0.0,
            },
            "R high": {
                "film_resistance": 3.0e9,
                "film_mass": 8.0e2,
                "film_stiffness": 0.0,
            },
        },
        "M sweep": {
            "M low": {
                "film_resistance": 1.0e6,
                "film_mass": 1.0e-2,
                "film_stiffness": 0.0,
            },
            "M mid": {
                "film_resistance": 1.0e6,
                "film_mass": 1.0e1,
                "film_stiffness": 0.0,
            },
            "M high": {
                "film_resistance": 1.0e6,
                "film_mass": 8.0e2,
                "film_stiffness": 0.0,
            },
        },
        "K sweep": {
            "K low": {
                "film_resistance": 1.0e6,
                "film_mass": 8.0e2,
                "film_stiffness": 1.0e10,
            },
            "K mid": {
                "film_resistance": 1.0e6,
                "film_mass": 8.0e2,
                "film_stiffness": 1.0e11,
            },
            "K high": {
                "film_resistance": 1.0e6,
                "film_mass": 8.0e2,
                "film_stiffness": 3.2e12,
            },
        },
    }

    s_in = r_inlet**2*np.pi
    s_out = r_outlet**2*np.pi

    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 100)
    omega = 2.0 * np.pi * freqs

    total_length = length_inlet+length_outlet
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)
    k0 = omega / C0

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=length_duct/2, c0=C0, rho0=RHO0)


    system_air_only =  inlet + halfduct + halfduct + outlet




    # %% IL part
    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    z_rigid = np.full(np.asarray(omega).shape, np.inf + 0.0j, dtype=np.complex128)

    p_end_rigid_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:,0]
    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:,0]


    case_group_results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for group_name, film_cases in film_case_groups.items():
        group_results: dict[str, dict[str, np.ndarray]] = {}
        for case_name, case_cfg in film_cases.items():
            film_case = GenericFilmSeriesImpedance(
                resistance=case_cfg["film_resistance"],
                mass=case_cfg["film_mass"],
                stiffness=case_cfg["film_stiffness"],
            )
            system_filter_case = inlet + halfduct + film_case + halfduct + outlet

            p_end_rigid_filter_case = system_filter_case.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]
            p_end_iec711_filter_case = system_filter_case.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

            il_rigid_case_db = 20.0 * np.log10(
                np.maximum(np.abs(p_end_rigid_system_air_only / p_end_rigid_filter_case), np.finfo(float).tiny)
            )
            il_iec711_case_db = 20.0 * np.log10(
                np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_filter_case), np.finfo(float).tiny)
            )

            group_results[case_name] = {
                "il_rigid_db": il_rigid_case_db,
                "il_iec711_db": il_iec711_case_db,
            }
        case_group_results[group_name] = group_results
        print(case_name)

    plot_il_case_groups(freqs, case_group_results)
    plt.show()