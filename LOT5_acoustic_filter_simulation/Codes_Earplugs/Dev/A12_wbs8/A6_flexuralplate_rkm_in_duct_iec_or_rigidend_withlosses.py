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

from function import build_fem_element_from_sparameters, plot_matrix_comparison
from toolkitsd.acoustmm import (
    CylindricalDuct,
    ExactFlexuralPlateSeriesImpedance,
    IEC711Coupler,
    LowFrequencyFlexuralPlateSeriesImpedance,
    ViscothermalDuct,
)


def interpolate_complex(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src, dtype=np.complex128)
    y_real = np.interp(x_dst, x_src, np.real(y_src))
    y_imag = np.interp(x_dst, x_src, np.imag(y_src))
    return y_real + 1j * y_imag


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    r_inlet = 5e-3
    r_outlet = r_inlet
    length_outlet = length_inlet

    r_duct = r_inlet
    cavity_each_side = 1.0e-2
    p_incident = 1.0

    s_in = np.pi * r_inlet**2
    s_out = np.pi * r_outlet**2

    plate_cases = {
        "plate_exact_d1": {
            "rho_plate": 1000.0,        # kg/m^3
            "thickness": 5e-5,         # m
            "young": 2.0e9,             # Pa
            "poisson": 0.30,
            "fem_file": HERE / "fem_rslt" / "rslt_fem_plate_first_mode_fit.txt",
        },
    "plate_lower_f_25pct": {
        "rho_plate": 1000.0,
        "thickness": 1.25e-5,   # 0.25 x original
        "young": 2.0e9,
        "poisson": 0.30,
        "fem_file": HERE / "fem_rslt" / "rslt_fem_plate_lower_f_25pct.txt",
    },
    }

    active_case = "plate_lower_f_25pct"
    case_cfg = plate_cases[active_case]
    FEM_FILE = case_cfg["fem_file"]

    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 400)
    omega = 2.0 * np.pi * freqs
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=cavity_each_side, c0=C0, rho0=RHO0)
    system_air_only = inlet + halfduct + halfduct + outlet

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    p_end_iec711_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    exact_plate = ExactFlexuralPlateSeriesImpedance(
        radius=r_duct,
        rho_plate=case_cfg["rho_plate"],
        h=case_cfg["thickness"],
        E=case_cfg["young"],
        nu=case_cfg["poisson"],
    )
    low_frequency_plate = LowFrequencyFlexuralPlateSeriesImpedance(
        radius=r_duct,
        rho_plate=case_cfg["rho_plate"],
        h=case_cfg["thickness"],
        E=case_cfg["young"],
        nu=case_cfg["poisson"],
    )
    exact_system = inlet + halfduct + exact_plate + halfduct + outlet
    low_frequency_system = inlet + halfduct + low_frequency_plate + halfduct + outlet
    matrix_tmm = exact_system.matrix(omega)
    matrix_tmm_low_frequency = low_frequency_system.matrix(omega)

    plot_matrix_comparison(
        freqs,
        matrix_tmm,
        freqs,
        matrix_tmm_low_frequency,
        mode="abs_phase",
        title_prefix="Exact D1 vs low-frequency D2 flexural plate",
    )
    plt.show()

    p_end_iec711_filter_exact = exact_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_iec711_filter_low_frequency = low_frequency_system.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    il_iec711_exact_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_air_only / p_end_iec711_filter_exact), np.finfo(float).tiny)
    )
    il_iec711_low_frequency_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_air_only / p_end_iec711_filter_low_frequency), np.finfo(float).tiny)
    )

    if FEM_FILE.exists():
        (
            freqs_fem,
            omega_fem,
            z01_fem,
            _z02_fem,
            _k01_fem,
            _k02_fem,
            _s11,
            _s21,
            matrix_fem,
            fem_element,
        ) = build_fem_element_from_sparameters(
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
            title_prefix=f"Flexural plate RKM surrogate vs FEM: {active_case}",
        )
        plt.show()

        z_711_fem = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega_fem)
        p_end_iec711_filter_fem = fem_element.state_tm_from_incident_wave(
            p_incident,
            z_711_fem,
            z01_fem,
            omega_fem,
        )[:, 0]
        p_end_iec711_air_fem = interpolate_complex(freqs, p_end_iec711_air_only, freqs_fem)
        il_iec711_fem_db = 20.0 * np.log10(
            np.maximum(np.abs(p_end_iec711_air_fem / p_end_iec711_filter_fem), np.finfo(float).tiny)
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.semilogx(freqs, il_iec711_exact_db, linewidth=2.2, label="Exact D1 plate TMM")
        ax.semilogx(freqs, il_iec711_low_frequency_db, linewidth=2.0, label="Low-frequency D2 plate TMM")
        ax.semilogx(freqs_fem, il_iec711_fem_db, "--", linewidth=1.8, label="FEM plate")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("IL [dB]")
        ax.set_title("IEC711 insertion loss: exact and low-frequency flexural plate")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
        ax.semilogx(freqs, il_iec711_exact_db, linewidth=2.2, label="Exact D1 plate TMM")
        ax.semilogx(freqs, il_iec711_low_frequency_db, linewidth=2.0, label="Low-frequency D2 plate TMM")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("IL [dB]")
        ax.set_title("IEC711 insertion loss: exact and low-frequency flexural plate")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        plt.show()

    surface_density = case_cfg["rho_plate"] * case_cfg["thickness"]
    print(f"surface_density = {surface_density:.3e} kg/m^2")
    print(f"bending_stiffness = {exact_plate.bending_stiffness:.3e} N.m")
    print(f"plate_mass = {exact_plate.plate_mass:.3e} kg")
