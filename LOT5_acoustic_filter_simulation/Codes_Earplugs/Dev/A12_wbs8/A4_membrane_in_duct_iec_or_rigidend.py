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
from toolkitsd.acoustmm import CylindricalDuct, IEC711Coupler, MembraneSeriesImpedance, ViscothermalDuct


def interpolate_complex(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src, dtype=np.complex128)
    y_real = np.interp(x_dst, x_src, np.real(y_src))
    y_imag = np.interp(x_dst, x_src, np.imag(y_src))
    return y_real + 1j * y_imag


def plot_iec_il_cases(case_results: dict[str, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for case_name, result in case_results.items():
        ax.semilogx(result["freqs_tmm"], result["il_tmm_db"], linewidth=2.0, label=f"{case_name} TMM")
        if "freqs_fem" in result:
            ax.semilogx(result["freqs_fem"], result["il_fem_db"], "--", linewidth=1.8, label=f"{case_name} FEM")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("IEC711 insertion loss for membrane surrogate cases")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()


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

    membrane_cases = {
        "silicone_membrane_soft": {
            "surface_density": 0.97,  # mu = rho*h
            "tension": 8.0,  # effective tension [N/m]
            "resistance": 2.0e5,  # fitted damping
            "geometry_constant": 1e-3,  # absorb Ct into tension
            "fem_file": HERE / "fem_rslt" / "rslt_fem_silicone_membrane_soft.txt",
        },
        "polymer_membrane_test": {
            "surface_density": 0.06,  # mu = rho*h = 1200 * 5e-5
            "tension": 10.0,  # N/m = FEM initial in-plane force
            "resistance": 1.0e4,  # fitted effective damping
            "geometry_constant": 0.0001,  # first circular membrane mode factor
            "fem_file": HERE / "fem_rslt" / "rslt_fem_silicone_membrane.txt",

        }

    }

    active_case = "polymer_membrane_test"
    case_cfg = membrane_cases[active_case]
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

    active_membrane = MembraneSeriesImpedance(
        radius=r_duct,
        surface_density=case_cfg["surface_density"],
        tension=case_cfg["tension"],
        resistance=case_cfg["resistance"],
        geometry_constant=case_cfg["geometry_constant"],
    )
    active_system = inlet + halfduct + active_membrane + halfduct + outlet
    matrix_tmm = active_system.matrix(omega)

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
            title_prefix=f"Membrane surrogate vs FEM: {active_case}",
        )
        plt.show()

    case_results = {}
    for case_name, case_cfg in membrane_cases.items():
        membrane = MembraneSeriesImpedance(
            radius=r_duct,
            surface_density=case_cfg["surface_density"],
            tension=case_cfg["tension"],
            resistance=case_cfg["resistance"],
            geometry_constant=case_cfg["geometry_constant"],
        )
        system_filter = inlet + halfduct + membrane + halfduct + outlet
        p_end_iec711_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
        il_iec711_case_db = 20.0 * np.log10(
            np.maximum(np.abs(p_end_iec711_air_only / p_end_iec711_filter), np.finfo(float).tiny)
        )

        case_results[case_name] = {
            "freqs_tmm": freqs,
            "il_tmm_db": il_iec711_case_db,
        }

        fem_file = case_cfg["fem_file"]
        if fem_file.exists():
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
                fem_file,
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
            p_end_iec711_air_case_fem = interpolate_complex(freqs, p_end_iec711_air_only, freqs_case_fem)
            il_iec711_case_fem_db = 20.0 * np.log10(
                np.maximum(np.abs(p_end_iec711_air_case_fem / p_end_iec711_filter_fem), np.finfo(float).tiny)
            )
            case_results[case_name]["freqs_fem"] = freqs_case_fem
            case_results[case_name]["il_fem_db"] = il_iec711_case_fem_db

    for case_name, case_cfg in membrane_cases.items():
        membrane = MembraneSeriesImpedance(
            radius=r_duct,
            surface_density=case_cfg["surface_density"],
            tension=case_cfg["tension"],
            resistance=case_cfg["resistance"],
            geometry_constant=case_cfg["geometry_constant"],
        )
        omega_ref = 2.0 * np.pi * 1000.0
        z_ref = membrane.acoustic_series_impedance(np.array([omega_ref]))[0]
        print(
            f"{case_name}: mu={case_cfg['surface_density']:.3e} kg/m^2, "
            f"T={case_cfg['tension']:.3e} N/m, Rm={case_cfg['resistance']:.3e} Pa.s/m^3, "
            f"|Zm(1 kHz)|={abs(z_ref):.3e} Pa.s/m^3"
        )

    plot_iec_il_cases(case_results)
