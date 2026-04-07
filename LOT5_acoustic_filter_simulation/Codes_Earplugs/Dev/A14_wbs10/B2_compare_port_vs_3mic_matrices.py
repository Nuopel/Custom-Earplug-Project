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

from B1_wb5_3mic_method_cases import (
    build_case_measurements,
    build_geometry,
    compute_transfer_functions,
    load_fem_double_load_pressures,
    select_pressure_points,
)
from function import build_fem_element_from_sparameters
from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, ThreeMicPostProcessor, ViscothermalDuct


def build_case_port_files() -> dict[str, Path]:
    base_dir = HERE / "fem_rslt" / "port_load"
    return {
        "Case A: silicone slab filter": base_dir / "rslt_fem_A0_caseA_silicone_slab_filter_in_duct.txt",
        "Case B: rigid slab filter": base_dir / "rslt_fem_A0_caseB_rigid_slab_filter_in_duct.txt",
        "Case C: silicone slab": base_dir / "rslt_fem_A0_caseC_silicone_slab_in_duct.txt",
        "Case D: air slab": base_dir / "rslt_fem_A0_caseD_air_slab_in_duct.txt",
        "Case E: rigid slab film": base_dir / "rslt_fem_A0_caseE_rigid_slab_film_in_duct.txt",
        "Case F: silicone slab film": base_dir / "rslt_fem_A0_caseF_silicone_slab_film_in_duct.txt",
    }


def load_port_case_matrices(
    *,
    c0: float,
    rho0: float,
    length_inlet: float,
    length_outlet: float,
    r_tube: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    area_in = np.pi * r_tube**2
    area_out = np.pi * r_tube**2
    inlet = CylindricalDuct(radius=r_tube, length=length_inlet, c0=c0, rho0=rho0)
    outlet = CylindricalDuct(radius=r_tube, length=length_outlet, c0=c0, rho0=rho0)

    case_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case_name, fem_file in build_case_port_files().items():
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


def load_three_mic_case_matrices(
    *,
    c0: float,
    rho0: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    geometry = build_geometry()
    case_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for case_name, measurement in build_case_measurements().items():
        data = load_fem_double_load_pressures(measurement.data_path)
        pressures_by_load = select_pressure_points(data, (measurement.mic_1, measurement.mic_2, measurement.mic_3))
        h12, h13 = compute_transfer_functions(pressures_by_load)

        params = AcousticParameters(data.frequencies_hz, c0=c0, rho0=rho0)
        post_processor = ThreeMicPostProcessor(params=params, geometry=geometry)
        identified_matrix_sh_pu = post_processor.identify_transfer_element_from_h_two_loads(
            h12,
            h13,
            k_tube=params.wavenumbers,
            z_tube=np.full(data.frequencies_hz.shape, params.z0, dtype=np.complex128),
            l1=geometry.l1,
            l2_by_load=np.full(2, geometry.l2, dtype=float),
            l3_by_load=np.array([geometry.l_load_a, geometry.l_load_b], dtype=float),
            s_tube=geometry.s_tube,
            s_eff=geometry.s_slab,
            return_basis="pu",
        )
        case_data[case_name] = (
            data.frequencies_hz,
            identified_matrix_sh_pu.matrix(2.0 * np.pi * data.frequencies_hz),
        )

    return case_data


def plot_case_comparison(
    case_name: str,
    port_case: tuple[np.ndarray, np.ndarray],
    three_mic_case: tuple[np.ndarray, np.ndarray],
    *,
    reference_tmm: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    labels = (("A", "B"), ("C", "D"))
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    freqs_port, matrix_port = port_case
    freqs_3mic, matrix_3mic = three_mic_case

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            label = labels[row][col]

            axes[idx_top, col].semilogx(freqs_port, np.abs(matrix_port[:, row, col]), linewidth=2.0, label="Port-load")
            axes[idx_top, col].semilogx(
                freqs_3mic,
                np.abs(matrix_3mic[:, row, col]),
                "--",
                linewidth=2.0,
                label="3-mic / 2-load",
            )
            axes[idx_bottom, col].semilogx(freqs_port, np.angle(matrix_port[:, row, col]), linewidth=2.0, label="Port-load")
            axes[idx_bottom, col].semilogx(
                freqs_3mic,
                np.angle(matrix_3mic[:, row, col]),
                "--",
                linewidth=2.0,
                label="3-mic / 2-load",
            )

            if reference_tmm is not None:
                freqs_ref, matrix_ref = reference_tmm
                axes[idx_top, col].semilogx(
                    freqs_ref,
                    np.abs(matrix_ref[:, row, col]),
                    "k--",
                    linewidth=2.2,
                    label="TMM",
                )
                axes[idx_bottom, col].semilogx(
                    freqs_ref,
                    np.angle(matrix_ref[:, row, col]),
                    "k--",
                    linewidth=2.2,
                    label="TMM",
                )

            axes[idx_top, col].set_ylabel(f"|{label}|")
            axes[idx_bottom, col].set_ylabel(f"Phase({label}) [rad]")
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].set_ylim([-np.pi, np.pi])
            axes[idx_top, col].legend(loc="best")

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle(f"{case_name}: decascaded port-load matrix vs identified 3-mic matrix")


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043
    geometry = build_geometry()

    port_case_matrices = load_port_case_matrices(
        c0=C0,
        rho0=RHO0,
        length_inlet=5.0e-3,
        length_outlet=5.0e-3,
        r_tube=geometry.r_tube,
    )
    three_mic_case_matrices = load_three_mic_case_matrices(c0=C0, rho0=RHO0)

    for case_name in build_case_measurements():
        reference_tmm = None
        if case_name == "Case D: air slab":
            freqs_ref, _ = port_case_matrices[case_name]
            omega_ref = 2.0 * np.pi * freqs_ref
            reference_tmm = (
                freqs_ref,
                ViscothermalDuct(
                    radius=geometry.r_slab,
                    length=geometry.l_slab,
                    c0=C0,
                    rho0=RHO0,
                ).matrix(omega_ref),
            )

        plot_case_comparison(
            case_name,
            port_case_matrices[case_name],
            three_mic_case_matrices[case_name],
            reference_tmm=reference_tmm,
        )

        plt.show()
