"""Three-mic/two-load identification for the A14 air-slab FEM case."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import AcousticParameters, GeometryConfig, ThreeMicPostProcessor, ViscothermalDuct


@dataclass(frozen=True)
class MeasurementConfig:
    data_path: Path
    mic_1: int
    mic_2: int
    mic_3: int


@dataclass(frozen=True)
class FemDoubleLoadPressureData:
    point_ids: tuple[int, ...]
    load_values_m: np.ndarray
    frequencies_hz: np.ndarray
    pressures_by_load: np.ndarray


def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def parse_fem_point_ids(path: Path) -> tuple[int, ...]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("%") or "Total acoustic pressure" not in line:
                continue
            pressure_point_ids = [int(match.group(1)) for match in re.finditer(r"Total acoustic pressure \(Pa\), Point:\s*(\d+)", line)]
            if not pressure_point_ids:
                raise ValueError(f"Unexpected point-id header in {path}")
            return tuple(pressure_point_ids)
    raise ValueError(f"Could not find FEM pressure header in {path}")


def split_array_by_load(loads_m: np.ndarray, freqs_hz: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_loads = np.unique(loads_m)
    if unique_loads.size != 2:
        raise ValueError(f"Expected exactly 2 load values, got {unique_loads.size}: {unique_loads}")

    freq_blocks = []
    value_blocks = []
    for load_value in unique_loads:
        mask = np.isclose(loads_m, load_value)
        freq_block = freqs_hz[mask]
        value_block = values[mask]
        order = np.argsort(freq_block)
        freq_blocks.append(freq_block[order])
        value_blocks.append(value_block[order])

    if not np.allclose(freq_blocks[0], freq_blocks[1]):
        raise ValueError("Frequency mismatch between load a and load b")

    return unique_loads, freq_blocks[0], np.stack(value_blocks, axis=-1)


def load_fem_double_load_pressures(path: Path) -> FemDoubleLoadPressureData:
    point_ids = parse_fem_point_ids(path)
    n_points = len(point_ids)
    loads_m = []
    freqs_hz = []
    pressures = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            expected_columns = 2 + n_points
            if len(parts) < expected_columns:
                raise ValueError(
                    f"Unexpected FEM table format in {path}: expected at least {expected_columns} columns, got {len(parts)}"
                )
            loads_m.append(float(parts[0]))
            freqs_hz.append(float(parts[1]))
            pressures.append([parse_fem_complex(token) for token in parts[2 : 2 + n_points]])

    load_values_m, freqs_by_load, pressures_by_load = split_array_by_load(
        np.asarray(loads_m, dtype=float),
        np.asarray(freqs_hz, dtype=float),
        np.asarray(pressures, dtype=np.complex128),
    )

    return FemDoubleLoadPressureData(
        point_ids=point_ids,
        load_values_m=load_values_m,
        frequencies_hz=freqs_by_load,
        pressures_by_load=pressures_by_load,
    )


def select_pressure_points(data: FemDoubleLoadPressureData, point_ids: tuple[int, ...]) -> np.ndarray:
    selected = []
    for point_id in point_ids:
        point_index = data.point_ids.index(point_id)
        selected.append(data.pressures_by_load[:, point_index, :])
    return np.stack(selected, axis=1)


def compute_transfer_functions(pressures_by_load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p1 = pressures_by_load[:, 0, :]
    p2 = pressures_by_load[:, 1, :]
    p3 = pressures_by_load[:, 2, :]
    return p2 / p1, p3 / p1


def build_geometry() -> GeometryConfig:
    geometry = GeometryConfig(
        l1=20.0e-3,
        l2=25.0e-3,
        l_slab=6.60e-3+6.4e-3,
        l_cav=0,
        l_load_a=11e-3,
        l_load_b=36e-3,
        r_tube=3.5e-3,
        r_slab=3.5e-3,
    )
    return geometry


def build_case_measurements() -> dict[str, MeasurementConfig]:
    base_dir = HERE / "fem_rslt" / "3mics_2loads"
    return {
        "Case A: silicone slab filter": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseA_silicone_slab_filter_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=7,
        ),
        "Case B: rigid slab filter": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseB_rigid_slab_filter_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=7,
        ),
        "Case C: silicone slab": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseC_silicone_slab_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=7,
        ),
        "Case D: air slab": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseD_air_slab_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=7,
        ),
        "Case E: rigid slab film": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseE_rigid_slab_film_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=8,
        ),
        "Case F: silicone slab film": MeasurementConfig(
            data_path=base_dir / "rslt_fem_A1_caseF_silicone_slab_film_in_duct_3mic_2loads.txt",
            mic_1=2,
            mic_2=3,
            mic_3=8,
        ),
    }


def plot_matrix_comparison_overlay(
    case_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    reference_case_name: str,
    geometry: GeometryConfig,
    params: AcousticParameters,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))
    reference_freqs_hz, _ = case_matrices[reference_case_name]
    reference_omega = 2.0 * np.pi * reference_freqs_hz
    reference_matrix = ViscothermalDuct(
        radius=geometry.r_slab,
        length=geometry.l_slab,
        c0=params.c0,
        rho0=params.rho0,
    ).matrix(reference_omega)

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            label = labels[row][col]
            for case_name, (freqs_hz, identified_matrix) in case_matrices.items():
                values_ident = identified_matrix[:, row, col]
                axes[idx_top, col].semilogx(freqs_hz, np.abs(values_ident), linewidth=1.8, label=case_name)
                axes[idx_bottom, col].semilogx(freqs_hz, np.angle(values_ident), linewidth=1.8, label=case_name)

            axes[idx_top, col].semilogx(
                reference_freqs_hz,
                np.abs(reference_matrix[:, row, col]),
                "--",
                linewidth=2.0,
                label=f"{reference_case_name} TMM",
            )
            axes[idx_bottom, col].semilogx(
                reference_freqs_hz,
                np.angle(reference_matrix[:, row, col]),
                "--",
                linewidth=2.0,
                label=f"{reference_case_name} TMM",
            )
            axes[idx_top, col].set_ylabel(f"|{label}|")
            axes[idx_bottom, col].set_ylabel(f"Phase({label}) [rad]")
            axes[idx_bottom, col].set_ylim([-np.pi, np.pi])
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)
            axes[idx_top, col].legend(loc="best")

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Identified matrices SH(p,U) for all 3-mic / 2-load cases")
    plt.show()


if __name__ == "__main__":
    geometry = build_geometry()
    case_measurements = build_case_measurements()
    C0 = 343.2
    RHO0 = 1.2043
    case_matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for case_name, measurement in case_measurements.items():
        data = load_fem_double_load_pressures(measurement.data_path)
        pressures_by_load = select_pressure_points(data, (measurement.mic_1, measurement.mic_2, measurement.mic_3))
        h12, h13 = compute_transfer_functions(pressures_by_load)

        params = AcousticParameters(data.frequencies_hz, c0=C0, rho0=RHO0)
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
        case_matrices[case_name] = (
            data.frequencies_hz,
            identified_matrix_sh_pu.matrix(2.0 * np.pi * data.frequencies_hz),
        )

    plot_matrix_comparison_overlay(
        case_matrices,
        reference_case_name="Case D: air slab",
        geometry=geometry,
        params=AcousticParameters(next(iter(case_matrices.values()))[0], c0=C0, rho0=RHO0),
    )
