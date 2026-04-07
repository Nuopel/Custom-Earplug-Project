from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np
from function import (
    build_fem_element_from_sparameters,
    plot_matrix_comparison,
)
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


def interpolate_complex(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src, dtype=np.complex128)
    y_real = np.interp(x_dst, x_src, np.real(y_src))
    y_imag = np.interp(x_dst, x_src, np.imag(y_src))
    return y_real + 1j * y_imag

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


def plot_iec_tmm_vs_fem(
    freqs_tmm: np.ndarray,
    p_end_air_tmm: np.ndarray,
    p_end_filter_tmm: np.ndarray,
    il_tmm_db: np.ndarray,
    freqs_fem: np.ndarray,
    p_end_air_fem: np.ndarray,
    p_end_filter_fem: np.ndarray,
    il_fem_db: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_tmm, 20.0 * np.log10(np.abs(p_end_air_tmm)), lw=2.0, label="Air cavity only, TMM IEC")
    axes[0].semilogx(freqs_tmm, 20.0 * np.log10(np.abs(p_end_filter_tmm)), lw=2.0, label="Filter + air cavity, TMM IEC")
    axes[0].semilogx(freqs_fem, 20.0 * np.log10(np.abs(p_end_air_fem)), "--", lw=2.0, label="Air cavity only, FEM IEC")
    axes[0].semilogx(freqs_fem, 20.0 * np.log10(np.abs(p_end_filter_fem)), "--", lw=2.0, label="Filter + air cavity, FEM IEC")
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title("IEC711 end pressure: TMM vs FEM")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_tmm, il_tmm_db, lw=2.2, label="IL from end-pressure ratio, TMM IEC")
    axes[1].semilogx(freqs_fem, il_fem_db, "--", lw=2.2, label="IL from end-pressure ratio, FEM IEC")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("IEC711 insertion loss from end-pressure ratio")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


def plot_iec_il_cases(case_results: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    for case_name, result in case_results.items():
        ax.semilogx(result["freqs_tmm"], result["il_tmm_db"], linewidth=2.0, label=f"{case_name} TMM")
        ax.semilogx(result["freqs_fem"], result["il_fem_db"], "--", linewidth=1.8, label=f"{case_name} FEM")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("IEC711 insertion loss for all RKM film families")
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

    film_cases = {
        "family_A_resistive_dominated": {
            "film_resistance": 3.0e9,
            "film_mass": 1.0e-2,
            "film_stiffness": 0.0,
            "fem_file": HERE / "fem_rslt" / "rslt_fem_rkm_case_a_rf3e9_mf1em2_kf0.txt",
        },
        "family_B_mass_dominated_scaled": {
            "film_resistance": 1.0e4,
            "film_mass": 8.0e2,
            "film_stiffness": 0.0,
            "fem_file": HERE / "fem_rslt" / "rslt_fem_rkm_scaled_case_b_rf1e4_mf8e2_kf0.txt",
        },
        "family_C_resonant_scaled": {
            "film_resistance": 1.0e6,
            "film_mass": 8.0e2,
            "film_stiffness": 3.2e10,
            "fem_file": HERE / "fem_rslt" / "rslt_fem_rkm_scaled_case_c_rf1e6_mf8e2_kf3p2e10.txt",
        },
    }
    active_case = "family_C_resonant_scaled"
    case_cfg = film_cases[active_case]
    film_resistance = case_cfg["film_resistance"]
    film_mass = case_cfg["film_mass"]
    film_stiffness = case_cfg["film_stiffness"]
    FEM_FILE = case_cfg["fem_file"]

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
    film = GenericFilmSeriesImpedance(
        resistance=film_resistance,
        mass=film_mass,
        stiffness=film_stiffness,
    )



    system_filter =   inlet + halfduct + film + halfduct + outlet
    system_air_only =  inlet + halfduct + halfduct + outlet

    matrix_tmm = system_filter.matrix(omega)
    r_tmm, t_tmm, a_tmm = system_filter.reflection_transmission_absorption_unequal_refs(
        Z_in=z0_in,
        Z_out=z0_in,
        omega=omega,
        k_ref=k0,
        length=total_length,
    )

    freqs_fem, omega_fem, z01_fem, z02_fem, k01_fem, k02_fem, s11, s21, matrix_fem, fem_element = build_fem_element_from_sparameters(
        FEM_FILE,
        s_in,
        s_out,
        RHO0,
        C0,
    )

    plot_matrix_comparison(
        freqs,
        matrix_tmm,
        freqs_fem,
        matrix_fem,
        mode="abs_phase",
        title_prefix="Circular cone + cav : transfer matrix comparison",
    )
    plt.show()


    plot_rta(freqs, r_tmm, t_tmm, a_tmm, title_prefix="Conical duct with losses: R/T/A")
    plt.show()

    # %% IL part
    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    z_rigid = np.full(np.asarray(omega).shape, np.inf + 0.0j, dtype=np.complex128)

    p_end_rigid_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:,0]
    p_end_rigid_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:,0]

    p_end_iec711_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:,0]
    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:,0]

    z_711_fem = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega_fem)
    p_end_iec711_system_filter_fem = fem_element.state_tm_from_incident_wave(p_incident, z_711_fem, z01_fem, omega_fem)[:,0]
    p_end_iec711_air_fem = interpolate_complex(freqs, p_end_iec711_system_air_only, freqs_fem)

    il_rigid_db = 20.0 * np.log10(np.maximum(np.abs(p_end_rigid_system_air_only / p_end_rigid_system_filter), np.finfo(float).tiny))
    il_iec711_db = 20.0 * np.log10(np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_system_filter), np.finfo(float).tiny))
    il_iec711_fem_db = 20.0 * np.log10(np.maximum(np.abs(p_end_iec711_air_fem / p_end_iec711_system_filter_fem), np.finfo(float).tiny))

    plot_results(
        freqs_hz=freqs,
        p_end_air_rigid=p_end_rigid_system_air_only,
        p_end_filter_rigid=p_end_rigid_system_filter,
        il_rigid_db=il_rigid_db,
        p_end_air_iec711=p_end_iec711_system_air_only,
        p_end_filter_iec711=p_end_iec711_system_filter,
        il_iec711_db=il_iec711_db,
    )

    plot_iec_tmm_vs_fem(
        freqs_tmm=freqs,
        p_end_air_tmm=p_end_iec711_system_air_only,
        p_end_filter_tmm=p_end_iec711_system_filter,
        il_tmm_db=il_iec711_db,
        freqs_fem=freqs_fem,
        p_end_air_fem=p_end_iec711_air_fem,
        p_end_filter_fem=p_end_iec711_system_filter_fem,
        il_fem_db=il_iec711_fem_db,
    )

    case_results = {}
    for case_name, case_cfg in film_cases.items():
        film = GenericFilmSeriesImpedance(
            resistance=case_cfg["film_resistance"],
            mass=case_cfg["film_mass"],
            stiffness=case_cfg["film_stiffness"],
        )
        system_filter = inlet + halfduct + film + halfduct + outlet

        p_end_iec711_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
        il_iec711_case_db = 20.0 * np.log10(
            np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_filter), np.finfo(float).tiny)
        )

        (
            freqs_case_fem,
            omega_case_fem,
            z01_case_fem,
            _z02_case_fem,
            _k01_case_fem,
            _k02_case_fem,
            _s11_case,
            _s21_case,
            _matrix_case_fem,
            fem_case_element,
        ) = build_fem_element_from_sparameters(
            case_cfg["fem_file"],
            s_in,
            s_out,
            RHO0,
            C0,
        )
        z_711_case_fem = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega_case_fem)
        p_end_iec711_filter_fem = fem_case_element.state_tm_from_incident_wave(
            p_incident,
            z_711_case_fem,
            z01_case_fem,
            omega_case_fem,
        )[:, 0]
        p_end_iec711_air_case_fem = interpolate_complex(freqs, p_end_iec711_system_air_only, freqs_case_fem)
        il_iec711_case_fem_db = 20.0 * np.log10(
            np.maximum(np.abs(p_end_iec711_air_case_fem / p_end_iec711_filter_fem), np.finfo(float).tiny)
        )

        case_results[case_name] = {
            "freqs_tmm": freqs,
            "il_tmm_db": il_iec711_case_db,
            "freqs_fem": freqs_case_fem,
            "il_fem_db": il_iec711_case_fem_db,
        }

    plot_iec_il_cases(case_results)
