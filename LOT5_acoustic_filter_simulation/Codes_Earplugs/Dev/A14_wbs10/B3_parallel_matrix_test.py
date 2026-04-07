from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from function import build_fem_element_from_sparameters, plot_matrix_comparison
from toolkitsd.acoustmm import CylindricalDuct, FrozenMatrixElement, IEC711Coupler


def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def compute_insertion_loss(
    system_air_only: FrozenMatrixElement,
    system_filter: FrozenMatrixElement,
    *,
    omega: np.ndarray,
    z0_in: np.ndarray,
    c0: float,
    rho0: float,
    p_incident: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    z_711 = IEC711Coupler(c0=c0, rho0=rho0).Z(omega)
    z_rigid = np.full(np.asarray(omega).shape, np.inf + 0.0j, dtype=np.complex128)

    p_end_rigid_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]
    p_end_rigid_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_rigid, z0_in, omega)[:, 0]

    p_end_iec711_system_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
    p_end_iec711_system_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    il_rigid_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_rigid_system_air_only / p_end_rigid_system_filter), np.finfo(float).tiny)
    )
    il_iec711_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_system_air_only / p_end_iec711_system_filter), np.finfo(float).tiny)
    )
    return il_rigid_db, il_iec711_db


def plot_insertion_loss_comparison(
    freqs_hz: np.ndarray,
    il_rigid_case_c_db: np.ndarray,
    il_rigid_parallel_db: np.ndarray,
    il_rigid_case_f_db: np.ndarray,
    il_iec711_case_c_db: np.ndarray,
    il_iec711_parallel_db: np.ndarray,
    il_iec711_case_f_db: np.ndarray,
    *,
    il_rigid_full_pressure_case_f_db: np.ndarray | None = None,
    freqs_full_pressure_hz: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, il_rigid_case_c_db, linewidth=2.0, label="Case C")
    axes[0].semilogx(freqs_hz, il_rigid_parallel_db, linewidth=2.2, label="Parallel (Case C || Case E)")
    axes[0].semilogx(freqs_hz, il_rigid_case_f_db, "--", linewidth=2.2, label="Case F")
    if il_rigid_full_pressure_case_f_db is not None and freqs_full_pressure_hz is not None:
        axes[0].semilogx(
            freqs_full_pressure_hz,
            il_rigid_full_pressure_case_f_db,
            ":",
            linewidth=2.4,
            label="Case F full-pressure FEM",
        )
    axes[0].set_ylabel("IL [dB]")
    axes[0].set_title("Rigid-end insertion loss")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_iec711_case_c_db, linewidth=2.0, label="Case C")
    axes[1].semilogx(freqs_hz, il_iec711_parallel_db, linewidth=2.2, label="Parallel (Case C || Case E)")
    axes[1].semilogx(freqs_hz, il_iec711_case_f_db, "--", linewidth=2.2, label="Case F")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("IEC711-load insertion loss")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")


def load_rigid_end_pressure(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequencies_hz = []
    end_pressure = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            frequencies_hz.append(float(parts[0]))
            end_pressure.append(parse_fem_complex(parts[1]))

    return np.asarray(frequencies_hz, dtype=float), np.asarray(end_pressure, dtype=np.complex128)


def load_decascaded_port_case_matrices(
    *,
    c0: float,
    rho0: float,
    length_inlet: float,
    length_outlet: float,
    r_tube: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fem_dir = HERE / "fem_rslt" / "port_load"

    case_files = {
        "Case C: silicone slab": fem_dir / "rslt_fem_A0_caseC_silicone_slab_in_duct.txt",
        "Case D: air slab": fem_dir / "rslt_fem_A0_caseD_air_slab_in_duct.txt",
        "Case E: rigid slab film": fem_dir / "rslt_fem_A0_caseE_rigid_slab_film_in_duct.txt",
        "Case F: silicone slab film": fem_dir / "rslt_fem_A0_caseF_silicone_slab_film_in_duct.txt",
    }

    area_in = np.pi * r_tube**2
    area_out = np.pi * r_tube**2
    inlet = CylindricalDuct(radius=r_tube, length=length_inlet, c0=c0, rho0=rho0)
    outlet = CylindricalDuct(radius=r_tube, length=length_outlet, c0=c0, rho0=rho0)

    case_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
            rho0,
            c0,
        )

        matrix_fem = fem_element.decascade_right(outlet).decascade_left(inlet).matrix(omega_fem)
        case_data[case_name] = (freqs_fem, matrix_fem)

    return case_data


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043
    LENGTH_INLET = 5.0e-3
    LENGTH_OUTLET = 5.0e-3
    R_TUBE = 3.5e-3
    rigid_end_dir = HERE / "fem_rslt" / "rigid_end"

    case_data = load_decascaded_port_case_matrices(
        c0=C0,
        rho0=RHO0,
        length_inlet=LENGTH_INLET,
        length_outlet=LENGTH_OUTLET,
        r_tube=R_TUBE,
    )

    freqs_c, matrix_c = case_data["Case C: silicone slab"]
    freqs_d, matrix_d = case_data["Case D: air slab"]
    freqs_e, matrix_e = case_data["Case E: rigid slab film"]
    freqs_f, matrix_f = case_data["Case F: silicone slab film"]

    if not (np.allclose(freqs_c, freqs_d) and np.allclose(freqs_c, freqs_e) and np.allclose(freqs_c, freqs_f)):
        raise ValueError("Frequency grids do not match between Cases C, D, E, and F")

    omega = 2.0 * np.pi * freqs_c
    system_case_c = FrozenMatrixElement.from_pu(matrix_c)
    system_case_e = FrozenMatrixElement.from_pu(matrix_e)
    system_parallel = system_case_c // system_case_e
    # Equivalent: system_case_c.in_parallel_with(system_case_e)
    matrix_parallel = system_parallel.matrix(omega)
    area_in = np.pi * R_TUBE**2
    z0_in = np.full(omega.shape, RHO0 * C0 / area_in + 0j, dtype=np.complex128)

    system_air_only = FrozenMatrixElement.from_pu(matrix_d)
    system_case_f = FrozenMatrixElement.from_pu(matrix_f)

    il_rigid_case_c_db, il_iec711_case_c_db = compute_insertion_loss(
        system_air_only,
        system_case_c,
        omega=omega,
        z0_in=z0_in,
        c0=C0,
        rho0=RHO0,
    )
    il_rigid_parallel_db, il_iec711_parallel_db = compute_insertion_loss(
        system_air_only,
        system_parallel,
        omega=omega,
        z0_in=z0_in,
        c0=C0,
        rho0=RHO0,
    )
    il_rigid_case_f_db, il_iec711_case_f_db = compute_insertion_loss(
        system_air_only,
        system_case_f,
        omega=omega,
        z0_in=z0_in,
        c0=C0,
        rho0=RHO0,
    )
    freqs_rigid_d, p_end_rigid_case_d = load_rigid_end_pressure(
        rigid_end_dir / "rslt_fem_A2_caseD_air_slab_in_duct_rigidend.txt"
    )
    freqs_rigid_f, p_end_rigid_case_f = load_rigid_end_pressure(
        rigid_end_dir / "rslt_fem_A2_caseF_silicone_slab_filter_filmrkm20_in_duct_rigidend.txt"
    )
    if not np.allclose(freqs_rigid_d, freqs_rigid_f):
        raise ValueError("Rigid-end pressure frequency grids do not match between Cases D and F")
    il_rigid_full_pressure_case_f_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_rigid_case_d / p_end_rigid_case_f), np.finfo(float).tiny)
    )

    plot_matrix_comparison(
        freqs_c,
        matrix_parallel,
        freqs_f,
        matrix_f,
        mode="abs_phase",
        title_prefix="Parallel test: (Case C || Case E) vs Case F",
    )
    plt.show()

    plot_insertion_loss_comparison(
        freqs_c,
        il_rigid_case_c_db,
        il_rigid_parallel_db,
        il_rigid_case_f_db,
        il_iec711_case_c_db,
        il_iec711_parallel_db,
        il_iec711_case_f_db,
        il_rigid_full_pressure_case_f_db=il_rigid_full_pressure_case_f_db,
        freqs_full_pressure_hz=freqs_rigid_d,
    )
    plt.show()
