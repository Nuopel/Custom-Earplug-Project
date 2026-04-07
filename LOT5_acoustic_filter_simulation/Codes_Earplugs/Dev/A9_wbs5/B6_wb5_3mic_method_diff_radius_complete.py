"""Minimal WBS5 three-mic FEM setup using the simulated measurement export."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import (
    AcousticParameters,
    CylindricalDuct,
    EardrumImpedance,
    FrozenMatrixElement,
    GeometryConfig,
    RadiationImpedance,
    ThreeMicPostProcessor,
)


@dataclass(frozen=True)
class SlabConfig:
    rho: float
    young: float
    poisson: float


@dataclass(frozen=True)
class MeasurementConfig:
    data_path: Path
    mic_1: int
    mic_2: int
    mic_3: int
    point_labels: dict[int, str]
    upstream_boundary_point: int
    downstream_boundary_point: int


@dataclass(frozen=True)
class ComsolDoubleLoadData:
    point_ids: tuple[int, ...]
    load_values_m: np.ndarray
    frequencies_hz: np.ndarray
    pressures_by_load: np.ndarray
    velocities_by_load: np.ndarray


def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def parse_fem_point_ids(path: Path) -> tuple[int, ...]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("%") or "Total acoustic pressure" not in line:
                continue

            pressure_point_ids = [int(match.group(1)) for match in re.finditer(r"Total acoustic pressure \(Pa\), Point:\s*(\d+)", line)]
            velocity_point_ids = [int(match.group(1)) for match in re.finditer(r"Total acoustic velocity, z-component \(m/s\), Point:\s*(\d+)", line)]
            if not pressure_point_ids or pressure_point_ids != velocity_point_ids:
                raise ValueError(f"Unexpected point-id header in {path}")
            return tuple(pressure_point_ids)

    raise ValueError(f"Could not find FEM header in {path}")


def split_array_by_load(
    loads_m: np.ndarray,
    freqs_hz: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    values_by_load = np.stack(value_blocks, axis=-1)
    return unique_loads, freq_blocks[0], values_by_load


def load_fem_double_load_data(path: Path) -> ComsolDoubleLoadData:
    point_ids = parse_fem_point_ids(path)
    n_points = len(point_ids)
    loads_m = []
    freqs_hz = []
    pressures = []
    velocities = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            expected_columns = 2 + 2 * n_points
            if len(parts) < expected_columns:
                raise ValueError(
                    f"Unexpected FEM table format in {path}: expected at least {expected_columns} columns, got {len(parts)}"
                )

            loads_m.append(float(parts[0]))
            freqs_hz.append(float(parts[1]))
            pressures.append([parse_fem_complex(token) for token in parts[2 : 2 + n_points]])
            velocities.append([parse_fem_complex(token) for token in parts[2 + n_points : 2 + 2 * n_points]])

    loads_array = np.asarray(loads_m, dtype=float)
    freqs_array = np.asarray(freqs_hz, dtype=float)
    pressures_array = np.asarray(pressures, dtype=np.complex128)
    velocities_array = np.asarray(velocities, dtype=np.complex128)

    load_values_m, freqs_by_load, pressures_by_load = split_array_by_load(loads_array, freqs_array, pressures_array)
    _, _, velocities_by_load = split_array_by_load(loads_array, freqs_array, velocities_array)

    return ComsolDoubleLoadData(
        point_ids=point_ids,
        load_values_m=load_values_m,
        frequencies_hz=freqs_by_load,
        pressures_by_load=pressures_by_load,
        velocities_by_load=velocities_by_load,
    )


def load_simulated_measurement(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loads_m = []
    freqs_hz = []
    pressures = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            loads_m.append(float(parts[0]))
            freqs_hz.append(float(parts[1]))
            pressures.append([parse_fem_complex(parts[idx]) for idx in range(2, 5)])
    return (
        np.asarray(loads_m, dtype=float),
        np.asarray(freqs_hz, dtype=float),
        np.asarray(pressures, dtype=np.complex128),
    )


def split_by_load(
    loads_m: np.ndarray,
    freqs_hz: np.ndarray,
    pressures: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_loads = np.unique(loads_m)
    if unique_loads.size != 2:
        raise ValueError(f"Expected exactly 2 load values, got {unique_loads.size}: {unique_loads}")

    freq_blocks = []
    pressure_blocks = []
    for load_value in unique_loads:
        mask = np.isclose(loads_m, load_value)
        freq_block = freqs_hz[mask]
        pressure_block = pressures[mask]
        order = np.argsort(freq_block)
        freq_blocks.append(freq_block[order])
        pressure_blocks.append(pressure_block[order])

    if not np.allclose(freq_blocks[0], freq_blocks[1]):
        raise ValueError("Frequency mismatch between load a and load b")

    pressures_by_load = np.stack(pressure_blocks, axis=-1)
    return unique_loads, freq_blocks[0], pressures_by_load


def compute_transfer_functions(pressures_by_load: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p1 = pressures_by_load[:, 0, :]
    p2 = pressures_by_load[:, 1, :]
    p3 = pressures_by_load[:, 2, :]
    h12 = p2 / p1
    h13 = p3 / p1
    return h12, h13


def load_b4_identified_matrices(
    freqs_by_load: np.ndarray,
    *,
    area_m2: float,
    npz_path: Path = HERE / "diff_radius_identified_matrices_from_fem.npz",
) -> tuple[np.ndarray | None, np.ndarray | None]:
    identified_matrix_b4_pu = None
    identified_matrix_b4_pv = None

    if not npz_path.exists():
        print(f"B4 comparison skipped: {npz_path.name} not found.")
        return identified_matrix_b4_pu, identified_matrix_b4_pv

    with np.load(npz_path) as data_b4:
        freqs_b4 = np.asarray(data_b4["frequencies_hz"], dtype=float)
        if not np.allclose(freqs_b4, freqs_by_load):
            print(
                "B4 comparison skipped: frequency mismatch between "
                f"{npz_path.name} and current B6 data."
            )
            return identified_matrix_b4_pu, identified_matrix_b4_pv

        identified_matrix_b4_pu = np.asarray(data_b4["identified_total"], dtype=np.complex128)
        identified_matrix_b4_pv = FrozenMatrixElement.from_pu(identified_matrix_b4_pu).to_pv(area_m2).matrices

    return identified_matrix_b4_pu, identified_matrix_b4_pv








def build_config() -> tuple[MeasurementConfig, GeometryConfig, SlabConfig]:
    measurement = MeasurementConfig(
        data_path=HERE / "fem_rslt" / "diff_radius_silicone_air_double_load_miclsh.txt",
        mic_1=7,
        mic_2=8,
        mic_3=12,
        point_labels={
            7: "Mic 1",
            8: "Mic 2",
            9: "In slab",
            10: "Out slab",
            11: "lsh",
            12: "Rigid end/mic3",
        },
        upstream_boundary_point=9,
        downstream_boundary_point=11,
    )
    geometry = GeometryConfig(
        l1=25.0e-3,
        l2=45.0e-3,
        l_slab=6.6e-3,
        l_cav=6.4e-3,
        l_load_a=11e-3,
        l_load_b=36e-3,
        r_tube=14.5e-3,
        r_slab=3.75e-3,
    )
    slab = SlabConfig(
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
    )
    return measurement, geometry, slab


def select_pressure_points(
    data: ComsolDoubleLoadData,
    point_ids: tuple[int, ...],
) -> np.ndarray:
    selected = []
    for point_id in point_ids:
        point_index = data.point_ids.index(point_id)
        selected.append(data.pressures_by_load[:, point_index, :])
    return np.stack(selected, axis=1)


def extract_direct_boundary_states(
    data: ComsolDoubleLoadData,
    *,
    upstream_point: int,
    downstream_point: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upstream_index = data.point_ids.index(upstream_point)
    downstream_index = data.point_ids.index(downstream_point)

    p0 = data.pressures_by_load[:, upstream_index, :]
    v0 = data.velocities_by_load[:, upstream_index, :]
    pl = data.pressures_by_load[:, downstream_index, :]
    vl = data.velocities_by_load[:, downstream_index, :]

    p0_a, p0_b = p0[:, 0], p0[:, 1]
    v0_a, v0_b = v0[:, 0], v0[:, 1]
    pl_a, pl_b = pl[:, 0], pl[:, 1]
    vl_a, vl_b = vl[:, 0], vl[:, 1]

    return p0_a, v0_a, pl_a, vl_a, p0_b, v0_b, pl_b, vl_b





def plot_measurement(load_values_m: np.ndarray, freqs_hz: np.ndarray, pressures_by_load: np.ndarray, measurement: MeasurementConfig) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    mic_ids = (measurement.mic_1, measurement.mic_2, measurement.mic_3)

    for axis, pressure_index, mic_id in zip(axes, range(3), mic_ids):
        for load_idx, (load_value, linestyle) in enumerate(zip(load_values_m, ("-", "--"))):
            axis.semilogx(
                freqs_hz,
                level_db(pressures_by_load[:, pressure_index, load_idx]),
                linestyle,
                lw=2.0,
                label=f"load = {load_value * 1e3:.1f} mm",
            )
        axis.set_ylabel(r"$20 \log_{10}|p|$ [dB]")
        axis.set_title(measurement.point_labels[mic_id])
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Frequency [Hz]")
    plt.show()


def plot_transfer_functions(load_values_m: np.ndarray, freqs_hz: np.ndarray, h12: np.ndarray, h13: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    for load_idx, (load_value, linestyle) in enumerate(zip(load_values_m, ("-", "--"))):
        axes[0].semilogx(
            freqs_hz,
            level_db(h12[:, load_idx]),
            linestyle,
            lw=2.0,
            label=f"H12, load = {load_value * 1e3:.1f} mm",
        )
        axes[1].semilogx(
            freqs_hz,
            level_db(h13[:, load_idx]),
            linestyle,
            lw=2.0,
            label=f"H13, load = {load_value * 1e3:.1f} mm",
        )

    axes[0].set_ylabel(r"$20 \log_{10}|H_{12}|$ [dB]")
    axes[0].set_title("Transfer function H12 = p2 / p1")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$20 \log_{10}|H_{13}|$ [dB]")
    axes[1].set_title("Transfer function H13 = p3 / p1")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()




def plot_identified_matrix(
    freqs_hz: np.ndarray,
    identified_matrix_direct: np.ndarray,
    identified_matrix_reconstructed: np.ndarray,
    condition_numbers_direct: np.ndarray,
    condition_numbers_reconstructed: np.ndarray,
    identified_matrix_b4: np.ndarray | None = None,
) -> None:
    fig_matrix, axes_matrix = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    labels = [["A", "B"], ["C", "D"]]

    for i in range(2):
        for j in range(2):
            ax = axes_matrix[i, j]
            label = labels[i][j]
            ax.semilogx(freqs_hz, np.abs(identified_matrix_direct[:, i, j]), lw=2.0, label=f"{label} direct")
            ax.semilogx(
                freqs_hz,
                np.abs(identified_matrix_reconstructed[:, i, j]),
                "--",
                lw=1.8,
                label=f"{label} reconstructed",
            )
            if identified_matrix_b4 is not None:
                ax.semilogx(
                    freqs_hz,
                    np.abs(identified_matrix_b4[:, i, j]),
                    ":",
                    lw=1.8,
                    label=f"{label} B4",
                )
            ax.set_ylabel(f"|{label}|")
            ax.set_title(f"Identified {label}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best")
            # ax.set_xlim([50,3000])
            # ax.set_ylim([0,0.0001])

    for ax in axes_matrix[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    fig_matrix.suptitle("Identified T_EP,SH coefficients")
    plt.show()

    # fig_condition, ax_condition = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=True)
    # ax_condition.semilogx(freqs_hz, condition_numbers_direct, lw=2.0, label="cond direct")
    # ax_condition.semilogx(
    #     freqs_hz,
    #     condition_numbers_reconstructed,
    #     "--",
    #     lw=1.8,
    #     color="#ea580c",
    #     label="cond reconstructed",
    # )
    # ax_condition.set_xlabel("Frequency [Hz]")
    # ax_condition.set_ylabel("Condition number")
    # ax_condition.set_title("Two-load inversion conditioning")
    # ax_condition.grid(True, which="both", alpha=0.3)
    # ax_condition.legend(loc="best")
    # plt.show()


def plot_il_comparison(
    freqs_hz: np.ndarray,
    identified_matrix_direct_pv: np.ndarray,
    identified_matrix_reconstructed_pv: np.ndarray,
    identified_matrix_b4_pu: np.ndarray | None,
    *,
    r_tube: float,
    l_slab: float,
    l_cav: float,
    c0: float,
    rho0: float,
    tl_ep_db: np.ndarray | None = None,
    ilc_db: np.ndarray | None = None,
    il_reduced_db: np.ndarray | None = None,
    il_tm_db: np.ndarray | None = None,
) -> None:
    area_m2 = np.pi * r_tube**2
    omega = 2.0 * np.pi * freqs_hz
    p0 = 1.0
    z_source = rho0 * c0 / area_m2
    z_rigid = np.inf

    air_equivalent = CylindricalDuct(radius=r_tube, length=l_slab, c0=c0, rho0=rho0)
    cavity = CylindricalDuct(radius=r_tube, length=l_cav, c0=c0, rho0=rho0)
    air_air_total = air_equivalent + cavity
    p_end_air_air_rigid = air_air_total.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]

    direct_element = FrozenMatrixElement.from_pv_converted_to_pu(identified_matrix_direct_pv, area_m2)
    reconstructed_element = FrozenMatrixElement.from_pv_converted_to_pu(identified_matrix_reconstructed_pv, area_m2)

    p_end_direct_rigid = direct_element.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]
    p_end_reconstructed_rigid = reconstructed_element.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]

    il_direct_db = level_db(p_end_air_air_rigid) - level_db(p_end_direct_rigid)
    il_reconstructed_db = level_db(p_end_air_air_rigid) - level_db(p_end_reconstructed_rigid)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.semilogx(freqs_hz, il_direct_db, lw=2.0, label="IL direct")
    ax.semilogx(freqs_hz, il_reconstructed_db, "--", lw=1.8, label="IL reconstructed")

    if identified_matrix_b4_pu is not None:
        b4_element = FrozenMatrixElement.from_pu(identified_matrix_b4_pu)
        p_end_b4_rigid = b4_element.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]
        il_b4_db = level_db(p_end_air_air_rigid) - level_db(p_end_b4_rigid)
        ax.semilogx(freqs_hz, il_b4_db, ":", lw=2.0, label="IL B4")

    if tl_ep_db is not None:
        ax.semilogx(freqs_hz, tl_ep_db, lw=1.6, color="#16a34a", label="TL EP")
    if ilc_db is not None:
        ax.semilogx(freqs_hz, ilc_db, lw=1.6, color="#ea580c", label="ILc")
    if il_reduced_db is not None:
        ax.semilogx(freqs_hz, il_reduced_db, lw=2.0, color="#7c3aed", label="IL reduced")
    if il_tm_db is not None:
        ax.semilogx(freqs_hz, il_tm_db, lw=1.8, color="#0891b2", label="IL TM ratio")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("Rigid-end IL from identified matrices")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()


def plot_pressure_and_il_comparison(
    freqs_hz: np.ndarray,
    *,
    p_open_tm: np.ndarray,
    p_occl_tm: np.ndarray,
    p_end_air_air_rigid: np.ndarray,
    p_end_reconstructed_rigid: np.ndarray,
    il_tm_db: np.ndarray,
    il_reconstructed_db: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, level_db(p_open_tm), lw=2.0, label="|p_open(Lec)|")
    axes[0].semilogx(freqs_hz, level_db(p_occl_tm), lw=2.0, label="|p_occl(Lec)|")
    axes[0].semilogx(freqs_hz, level_db(p_end_air_air_rigid), "--", lw=1.8, label="|p_end open rigid|")
    axes[0].semilogx(freqs_hz, level_db(p_end_reconstructed_rigid), "--", lw=1.8, label="|p_end occl rigid|")
    axes[0].set_ylabel(r"$20 \log_{10}|p|$ [dB]")
    axes[0].set_title("Pressure Comparison")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_tm_db, lw=2.0, label="IL TM ratio")
    axes[1].semilogx(freqs_hz, il_reconstructed_db, "--", lw=1.8, label="IL reconstructed")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("IL Comparison")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


def compare_boundary_states(
    freqs_hz: np.ndarray,
    p0_a: np.ndarray,
    v0_a: np.ndarray,
    pl_a: np.ndarray,
    vl_a: np.ndarray,
    p0_b: np.ndarray,
    v0_b: np.ndarray,
    pl_b: np.ndarray,
    vl_b: np.ndarray,
    p0_a_direct: np.ndarray,
    v0_a_direct: np.ndarray,
    pl_a_direct: np.ndarray,
    vl_a_direct: np.ndarray,
    p0_b_direct: np.ndarray,
    v0_b_direct: np.ndarray,
    pl_b_direct: np.ndarray,
    vl_b_direct: np.ndarray,
) -> None:
    reconstructed = {
        "p0_a": p0_a,
        "v0_a": v0_a,
        "pl_a": pl_a,
        "vl_a": vl_a,
        "p0_b": p0_b,
        "v0_b": v0_b,
        "pl_b": pl_b,
        "vl_b": vl_b,
    }
    direct = {
        "p0_a": p0_a_direct,
        "v0_a": v0_a_direct,
        "pl_a": pl_a_direct,
        "vl_a": vl_a_direct,
        "p0_b": p0_b_direct,
        "v0_b": v0_b_direct,
        "pl_b": pl_b_direct,
        "vl_b": vl_b_direct,
    }

    print("=== BOUNDARY STATE COMPARISON ===")
    for name in ("p0_a", "v0_a", "pl_a", "vl_a", "p0_b", "v0_b", "pl_b", "vl_b"):
        gap_db = level_db(reconstructed[name]) - level_db(direct[name])
        print(
            f"{name:>5} | median |gap| [dB] = {np.median(np.abs(gap_db)):.3e} | "
            f"max |gap| [dB] = {np.max(np.abs(gap_db)):.3e}"
        )
    print()

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    plot_order = (
        ("p0_a", "p0_a"),
        ("v0_a", "v0_a"),
        ("pl_a", "pl_a"),
        ("vl_a", "vl_a"),
        ("p0_b", "p0_b"),
        ("v0_b", "v0_b"),
        ("pl_b", "pl_b"),
        ("vl_b", "vl_b"),
    )

    for ax, (title, key) in zip(axes.flat, plot_order):
        ax.semilogx(freqs_hz, level_db(reconstructed[key]), lw=2.0, label=f"{key} reconstructed")
        ax.semilogx(freqs_hz, level_db(direct[key]), "--", lw=1.8, label=f"{key} direct")
        ax.set_title(title)
        ax.set_ylabel(r"$20 \log_{10}|.|$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best")
        # ax.set_ylim([5.8,6.2])
    for ax in axes[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    plt.show()




if __name__ == "__main__":
    measurement, geometry, slab = build_config()
    data = load_fem_double_load_data(measurement.data_path)
    load_values_m = data.load_values_m
    freqs_by_load = data.frequencies_hz
    pressures_by_load = select_pressure_points(
        data,
        (measurement.mic_1, measurement.mic_2, measurement.mic_3),
    )
    params = AcousticParameters(freqs_by_load, c0=344.96, rho0=1.2)

    print("=== WBS5 CONFIG ===")
    print(f"data file    : {measurement.data_path.name}")
    print(f"point ids    : {data.point_ids}")
    print(f"mic_1       : {measurement.mic_1}")
    print(f"mic_2       : {measurement.mic_2}")
    print(f"mic_3       : {measurement.mic_3}")
    print(f"upstream pt : {measurement.upstream_boundary_point}")
    print(f"downstream pt: {measurement.downstream_boundary_point}")
    print(f"l1          : {geometry.l1 * 1e3:.2f} mm")
    print(f"l2          : {geometry.l2 * 1e3:.2f} mm")
    print(f"l_slab      : {geometry.l_slab * 1e3:.2f} mm")
    print(f"l_cav       : {geometry.l_cav * 1e3:.2f} mm")
    print(f"l_load_a    : {geometry.l_load_a * 1e3:.2f} mm")
    print(f"l_load_b    : {geometry.l_load_b * 1e3:.2f} mm")
    print(f"r_tube      : {geometry.r_tube * 1e3:.2f} mm")
    print(f"r_slab      : {geometry.r_slab * 1e3:.2f} mm")
    print(f"rho         : {slab.rho:.1f} kg/m^3")
    print(f"young       : {slab.young:.3e} Pa")
    print(f"poisson     : {slab.poisson:.3f}")
    print(f"loads [mm]   : {', '.join(f'{value * 1e3:.3f}' for value in load_values_m)}")
    print(f"n_rows       : {data.load_values_m.size * data.frequencies_hz.size}")
    print(f"n_freqs      : {params.frequencies.size}")
    print(f"f_min        : {params.frequencies.min():.3f} Hz")
    print(f"f_max        : {params.frequencies.max():.3f} Hz")




    plot_measurement(load_values_m, freqs_by_load, pressures_by_load, measurement)

    # begining of the measurement post processing
    h12, h13 = compute_transfer_functions(pressures_by_load)
    plot_transfer_functions(load_values_m, freqs_by_load, h12, h13)
    post_processor = ThreeMicPostProcessor(params=params, geometry=geometry)


    #identify matrix
    k_tube = params.wavenumbers
    z_tube = np.full(freqs_by_load.shape, params.z0, dtype=np.complex128)
    l2_by_load = np.full(2, geometry.l2, dtype=float)
    l3_by_load = np.array([geometry.l_load_a, geometry.l_load_b], dtype=float)

    p0_a, v0_a, pl_a, vl_a, p0_b, v0_b, pl_b, vl_b = post_processor.reconstruct_boundary_states_from_h(
        h12,
        h13,
        k_tube=k_tube,
        z_tube=z_tube,
        l1=geometry.l1,
        l2_by_load=l2_by_load,
        l3_by_load=l3_by_load,
        s_tube=geometry.s_tube,
        s_eff=geometry.s_slab,
    )

    p0_a_direct, v0_a_direct, pl_a_direct, vl_a_direct, p0_b_direct, v0_b_direct, pl_b_direct, vl_b_direct = extract_direct_boundary_states(
        data,
        upstream_point=measurement.upstream_boundary_point,
        downstream_point=measurement.downstream_boundary_point,
    )

    identified_matrix_sh_direct = post_processor.identify_transfer_matrix_two_loads(
        p0_a_direct,
        v0_a_direct,
        pl_a_direct,
        vl_a_direct,
        p0_b_direct,
        v0_b_direct,
        pl_b_direct,
        vl_b_direct,
    )
    delta_direct = pl_a_direct * vl_b_direct - pl_b_direct * vl_a_direct
    condition_numbers_direct = 1.0 / np.maximum(np.abs(delta_direct), np.finfo(float).tiny)

    identified_matrix_sh = post_processor.identify_transfer_matrix_two_loads(
        p0_a,
        v0_a,
        pl_a,
        vl_a,
        p0_b,
        v0_b,
        pl_b,
        vl_b,
    )
    delta = pl_a * vl_b - pl_b * vl_a
    condition_numbers = 1.0 / np.maximum(np.abs(delta), np.finfo(float).tiny)

    print(f"|p0_a|(fmin) : {np.abs(p0_a[0]):.6e}")
    print(f"|v0_a|(fmin) : {np.abs(v0_a[0]):.6e}")
    print(f"|pl_a|(fmin) : {np.abs(pl_a[0]):.6e}")
    print(f"|vl_a|(fmin) : {np.abs(vl_a[0]):.6e}")
    print(f"|p0_b|(fmin) : {np.abs(p0_b[0]):.6e}")
    print(f"|v0_b|(fmin) : {np.abs(v0_b[0]):.6e}")
    print(f"|pl_b|(fmin) : {np.abs(pl_b[0]):.6e}")
    print(f"|vl_b|(fmin) : {np.abs(vl_b[0]):.6e}")
    print("--- direct FEM at points 8 and 9 ---")
    print(f"|p0_a_direct|(fmin) : {np.abs(p0_a_direct[0]):.6e}")
    print(f"|v0_a_direct|(fmin) : {np.abs(v0_a_direct[0]):.6e}")
    print(f"|pl_a_direct|(fmin) : {np.abs(pl_a_direct[0]):.6e}")
    print(f"|vl_a_direct|(fmin) : {np.abs(vl_a_direct[0]):.6e}")
    print(f"|p0_b_direct|(fmin) : {np.abs(p0_b_direct[0]):.6e}")
    print(f"|v0_b_direct|(fmin) : {np.abs(v0_b_direct[0]):.6e}")
    print(f"|pl_b_direct|(fmin) : {np.abs(pl_b_direct[0]):.6e}")
    print(f"|vl_b_direct|(fmin) : {np.abs(vl_b_direct[0]):.6e}")
    print(
        "|T_EP,SH_direct|(fmin): "
        f"{np.abs(identified_matrix_sh_direct[0, 0, 0]):.6e}, "
        f"{np.abs(identified_matrix_sh_direct[0, 0, 1]):.6e}, "
        f"{np.abs(identified_matrix_sh_direct[0, 1, 0]):.6e}, "
        f"{np.abs(identified_matrix_sh_direct[0, 1, 1]):.6e}"
    )
    print(f"cond-direct(fmin): {condition_numbers_direct[0]:.6e}")
    print(
        "|T_EP,SH|(fmin): "
        f"{np.abs(identified_matrix_sh[0, 0, 0]):.6e}, "
        f"{np.abs(identified_matrix_sh[0, 0, 1]):.6e}, "
        f"{np.abs(identified_matrix_sh[0, 1, 0]):.6e}, "
        f"{np.abs(identified_matrix_sh[0, 1, 1]):.6e}"
    )
    print(f"cond-indicator(fmin): {condition_numbers[0]:.6e}")
#%%
    compare_boundary_states(
        freqs_by_load,
        p0_a,
        v0_a,
        pl_a,
        vl_a,
        p0_b,
        v0_b,
        pl_b,
        vl_b,
        p0_a_direct,
        v0_a_direct,
        pl_a_direct,
        vl_a_direct,
        p0_b_direct,
        v0_b_direct,
        pl_b_direct,
        vl_b_direct,
    )
    #%%
    omega = 2.0 * np.pi * freqs_by_load
    cavity = CylindricalDuct(radius=geometry.r_slab, length=geometry.l_cav, c0=params.c0, rho0=params.rho0)
    z_tm = np.full_like(omega, 1e20, dtype=np.complex128)

    z_ec = params.z0 / geometry.s_slab
    z_r = params.z0 / geometry.s_slab

    identified_matrix_ep_pu = FrozenMatrixElement.from_pv_converted_to_pu(identified_matrix_sh, geometry.s_slab).decascade_right(cavity).matrix(omega)

    tl_ep_db, ilc_db, il_reduced_db = post_processor.compute_reduced_il_from_matrix(
        identified_matrix_ep_pu,
        z_r=z_r,
        z_tm=z_tm,
        z_ec=z_ec,
    )
    p_open_tm, p_occl_tm, il_tm_db = post_processor.compute_tm_pressure_il_from_matrix(
        identified_matrix_ep_pu,
        z_r=z_r,
        z_tm=z_tm,
        z_ec=z_ec,
    )

    print(f"TL_EP(fmin) [dB]: {tl_ep_db[0]:.6e}")
    print(f"ILc(fmin) [dB]  : {ilc_db[0]:.6e}")
    print(f"IL_red(fmin) [dB]: {il_reduced_db[0]:.6e}")
    print(f"|p_open(Lec)|(fmin): {np.abs(p_open_tm[0]):.6e}")
    print(f"|p_occl(Lec)|(fmin): {np.abs(p_occl_tm[0]):.6e}")
    print(f"IL_TM(fmin) [dB] : {il_tm_db[0]:.6e}")

    identified_matrix_b4_pu, identified_matrix_b4_pv = load_b4_identified_matrices(
        freqs_by_load,
        area_m2=geometry.s_slab,
    )

    plot_identified_matrix(
        freqs_by_load,
        identified_matrix_sh_direct,
        identified_matrix_sh,
        condition_numbers_direct,
        condition_numbers,
        identified_matrix_b4_pv,
    )

    plot_il_comparison(
        freqs_by_load,
        identified_matrix_sh_direct,
        identified_matrix_sh,
        identified_matrix_b4_pu,
        r_tube=geometry.r_tube,
        l_slab=geometry.l_slab,
        l_cav=geometry.l_cav,
        c0=params.c0,
        rho0=params.rho0,
        tl_ep_db=tl_ep_db,
        ilc_db=ilc_db,
        il_reduced_db=il_reduced_db,
        il_tm_db=il_tm_db,
    )
