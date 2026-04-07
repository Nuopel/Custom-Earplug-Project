"""Minimal diff-radius WBS5 three-mic workflow using the post-processor wrapper."""

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

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, FrozenMatrixElement, GeometryConfig, ThreeMicPostProcessor


@dataclass(frozen=True)
class MeasurementConfig:
    data_path: Path
    mic_1: int
    mic_2: int
    mic_3: int


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

    load_values_m, freqs_by_load, pressures_by_load = split_array_by_load(
        np.asarray(loads_m, dtype=float),
        np.asarray(freqs_hz, dtype=float),
        np.asarray(pressures, dtype=np.complex128),
    )
    _, _, velocities_by_load = split_array_by_load(
        np.asarray(loads_m, dtype=float),
        np.asarray(freqs_hz, dtype=float),
        np.asarray(velocities, dtype=np.complex128),
    )

    return ComsolDoubleLoadData(
        point_ids=point_ids,
        load_values_m=load_values_m,
        frequencies_hz=freqs_by_load,
        pressures_by_load=pressures_by_load,
        velocities_by_load=velocities_by_load,
    )


def select_pressure_points(data: ComsolDoubleLoadData, point_ids: tuple[int, ...]) -> np.ndarray:
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


def load_b4_identified_matrices(
    freqs_hz: np.ndarray,
    *,
    npz_path: Path = HERE / "diff_radius_identified_matrices_from_fem.npz",
) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    with np.load(npz_path) as data:
        freqs_b4 = np.asarray(data["frequencies_hz"], dtype=float)
        if not np.allclose(freqs_b4, freqs_hz):
            return None
        return np.asarray(data["identified_total"], dtype=np.complex128)


def build_config() -> tuple[MeasurementConfig, GeometryConfig]:
    measurement = MeasurementConfig(
        data_path=HERE / "fem_rslt" / "diff_radius_silicone_air_double_load_miclsh.txt",
        mic_1=7,
        mic_2=8,
        mic_3=12,
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
    return measurement, geometry



def plot_il_comparison(
    freqs_hz: np.ndarray,
    identified_matrix_sh_pu: FrozenMatrixElement,
    identified_matrix_ep_pu: np.ndarray,
    identified_matrix_b4_pu: np.ndarray | None,
    *,
    geometry: GeometryConfig,
    params: AcousticParameters,
    tl_ep_db: np.ndarray,
    ilc_db: np.ndarray,
    il_reduced_db: np.ndarray,
    il_tm_db: np.ndarray,
) -> None:
    omega = 2.0 * np.pi * freqs_hz
    p0 = 1.0
    z_source = params.z0 / geometry.s_slab
    z_rigid = np.inf

    cavity = CylindricalDuct(radius=geometry.r_slab, length=geometry.l_cav, c0=params.c0, rho0=params.rho0)
    air_equivalent = CylindricalDuct(radius=geometry.r_slab, length=geometry.l_slab, c0=params.c0, rho0=params.rho0)
    air_air_total = air_equivalent + cavity
    p_end_air_air_rigid = air_air_total.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]

    p_end_sh_rigid = identified_matrix_sh_pu.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]
    p_end_ep_rebuilt_rigid = (FrozenMatrixElement.from_pu(identified_matrix_ep_pu) + cavity).state_tm_from_incident_wave(
        p0,
        z_rigid,
        z_source,
        omega,
    )[:, 0]

    il_sh_db = level_db(p_end_air_air_rigid) - level_db(p_end_sh_rigid)
    il_ep_rebuilt_db = level_db(p_end_air_air_rigid) - level_db(p_end_ep_rebuilt_rigid)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax.semilogx(freqs_hz, il_sh_db, lw=2.0, label="IL identified SH")
    ax.semilogx(freqs_hz, il_ep_rebuilt_db, "--", lw=1.8, label="IL EP + cavity")

    if identified_matrix_b4_pu is not None:
        p_end_b4_rigid = FrozenMatrixElement.from_pu(identified_matrix_b4_pu).state_tm_from_incident_wave(
            p0,
            z_rigid,
            z_source,
            omega,
        )[:, 0]
        il_b4_db = level_db(p_end_air_air_rigid) - level_db(p_end_b4_rigid)
        ax.semilogx(freqs_hz, il_b4_db, ":", lw=2.0, label="IL B4")

    ax.semilogx(freqs_hz, tl_ep_db, lw=1.6, color="#16a34a", label="TL EP")
    ax.semilogx(freqs_hz, ilc_db, lw=1.6, color="#ea580c", label="ILc")
    ax.semilogx(freqs_hz, il_reduced_db, lw=2.0, color="#7c3aed", label="IL reduced")
    ax.semilogx(freqs_hz, il_tm_db, lw=1.8, color="#0891b2", label="IL TM ratio")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("Diff-radius IL comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()





if __name__ == "__main__":
    measurement, geometry = build_config()
    data = load_fem_double_load_data(measurement.data_path)
    pressures_by_load = select_pressure_points(data, (measurement.mic_1, measurement.mic_2, measurement.mic_3))
    h12, h13 = compute_transfer_functions(pressures_by_load)

    params = AcousticParameters(data.frequencies_hz, c0=344.96, rho0=1.2)
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

    omega = 2.0 * np.pi * data.frequencies_hz
    cavity = CylindricalDuct(radius=geometry.r_slab, length=geometry.l_cav, c0=params.c0, rho0=params.rho0)
    identified_matrix_ep_pu = identified_matrix_sh_pu.decascade_right(cavity).matrix(omega)

    z_tm = np.full_like(omega, 1e20, dtype=np.complex128)
    z_ec = params.z0 / geometry.s_slab
    z_r = params.z0 / geometry.s_slab

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

    print(f"|H12|(fmin, load a)   : {np.abs(h12[0, 0]):.6e}")
    print(f"|H13|(fmin, load a)   : {np.abs(h13[0, 0]):.6e}")
    print(f"|T_SH|(fmin)          : {np.abs(identified_matrix_sh_pu.matrix(omega)[0, 0, 0]):.6e}")
    print(f"TL_EP(fmin) [dB]      : {tl_ep_db[0]:.6e}")
    print(f"ILc(fmin) [dB]        : {ilc_db[0]:.6e}")
    print(f"IL_reduced(fmin) [dB] : {il_reduced_db[0]:.6e}")
    print(f"|p_open|(fmin)        : {np.abs(p_open_tm[0]):.6e}")
    print(f"|p_occl|(fmin)        : {np.abs(p_occl_tm[0]):.6e}")
    print(f"IL_TM(fmin) [dB]      : {il_tm_db[0]:.6e}")

    plot_il_comparison(
        data.frequencies_hz,
        identified_matrix_sh_pu,
        identified_matrix_ep_pu,
        load_b4_identified_matrices(data.frequencies_hz),
        geometry=geometry,
        params=params,
        tl_ep_db=tl_ep_db,
        ilc_db=ilc_db,
        il_reduced_db=il_reduced_db,
        il_tm_db=il_tm_db,
    )
