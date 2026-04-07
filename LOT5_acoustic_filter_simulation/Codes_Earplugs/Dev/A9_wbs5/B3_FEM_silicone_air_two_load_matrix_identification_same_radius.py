"""Two-load FEM matrix identification for the WBS5 same-radius verification case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
VERIF_DIR = HERE / "fem_rslt"
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import CylindricalDuct, FrozenMatrixElement, ViscothermalDuct

OPEN_PATH = VERIF_DIR / "same_radius_silicone_air_open.txt"
RIGID_PATH = VERIF_DIR / "same_radius_silicone_air_rigidend.txt"

POINT_LABELS = {
    6: "Mic 1",
    7: "Mic 2",
    8: "In slab",
    9: "Out slab",
    10: "Rigid end/mic3",
}

POINT_TOTAL_LEFT = 8
POINT_TOTAL_RIGHT = 10
POINT_SLAB_LEFT = 8
POINT_SLAB_RIGHT = 9

RADIUS_M = 3.75e-3
SLAB_LENGTH_M = 6.6e-3
CAVITY_LENGTH_M = 6.4e-3
RHO_AIR = 1.2
C_AIR = 343
AREA_M2 = np.pi * RADIUS_M**2


@dataclass(frozen=True)
class FemCase:
    frequencies_hz: np.ndarray
    point_ids: tuple[int, ...]
    pressures: np.ndarray
    velocities: np.ndarray



def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def parse_point_ids(path: Path) -> tuple[int, ...]:
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


def load_fem_case(path: Path) -> FemCase:
    point_ids = parse_point_ids(path)
    n_points = len(point_ids)
    frequencies_hz = []
    pressures = []
    velocities = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            expected_columns = 1 + 2 * n_points
            if len(parts) < expected_columns:
                raise ValueError(
                    f"Unexpected FEM table format in {path}: expected at least {expected_columns} columns, got {len(parts)}"
                )

            frequencies_hz.append(float(parts[0]))
            pressures.append([parse_fem_complex(token) for token in parts[1 : 1 + n_points]])
            velocities.append([parse_fem_complex(token) for token in parts[1 + n_points : 1 + 2 * n_points]])

    if not frequencies_hz:
        raise ValueError(f"No FEM data rows found in {path}")

    return FemCase(
        frequencies_hz=np.asarray(frequencies_hz, dtype=float),
        point_ids=point_ids,
        pressures=np.asarray(pressures, dtype=np.complex128).T,
        velocities=np.asarray(velocities, dtype=np.complex128).T,
    )


def point_index(case: FemCase, point_id: int) -> int:
    try:
        return case.point_ids.index(point_id)
    except ValueError as exc:
        raise ValueError(f"Point {point_id} not found in FEM file. Available points: {case.point_ids}") from exc


def state_at(case: FemCase, point_id: int) -> np.ndarray:
    idx = point_index(case, point_id)
    return np.column_stack((case.pressures[idx], AREA_M2 * case.velocities[idx]))


def identify_two_port(
    right_state_a: np.ndarray,
    right_state_b: np.ndarray,
    left_state_a: np.ndarray,
    left_state_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    right_states = np.stack((right_state_a, right_state_b), axis=-1)
    left_states = np.stack((left_state_a, left_state_b), axis=-1)
    matrices = np.einsum("nij,njk->nik", left_states, np.linalg.inv(right_states))
    return matrices, np.linalg.cond(right_states)


def print_summary(
    frequencies_hz: np.ndarray,
    identified_total: np.ndarray,
    identified_slab_direct: np.ndarray,
    identified_slab_decascaded: np.ndarray,
    total_condition_numbers: np.ndarray,
    slab_direct_condition_numbers: np.ndarray,
    p_end_fem_rigid: np.ndarray,
    p_end_identified_rigid: np.ndarray,
    p_end_fem_open: np.ndarray,
    p_end_identified_open: np.ndarray,
    p_end_air_air_rigid: np.ndarray,
) -> None:
    determinant_total = np.linalg.det(identified_total)
    determinant_slab_direct = np.linalg.det(identified_slab_direct)
    determinant_slab_decascaded = np.linalg.det(identified_slab_decascaded)
    rigid_pressure_error_db = np.max(np.abs(level_db(p_end_fem_rigid) - level_db(p_end_identified_rigid)))
    open_pressure_error_db = np.max(np.abs(level_db(p_end_fem_open) - level_db(p_end_identified_open)))
    il_rigid_db = level_db(p_end_air_air_rigid) - level_db(p_end_identified_rigid)
    slab_compare_error = np.linalg.norm(identified_slab_direct - identified_slab_decascaded, axis=(1, 2)) / np.maximum(
        np.linalg.norm(identified_slab_direct, axis=(1, 2)),
        np.finfo(float).tiny,
    )

    print("=== WBS5 FEM TWO-LOAD MATRIX IDENTIFICATION ===")
    print(f"Open load file               : {OPEN_PATH.name}")
    print(f"Rigid load file              : {RIGID_PATH.name}")
    print(f"Total matrix                 : point {POINT_TOTAL_LEFT} -> point {POINT_TOTAL_RIGHT}")
    print(f"Slab matrix direct           : point {POINT_SLAB_LEFT} -> point {POINT_SLAB_RIGHT}")
    print(f"Slab matrix decascaded       : total decascaded by cavity {POINT_SLAB_RIGHT} -> {POINT_TOTAL_RIGHT}")
    print(f"Frequency range [Hz]         : {frequencies_hz[0]:.3f} -> {frequencies_hz[-1]:.3f}")
    print(
        f"Condition number range total : {np.min(total_condition_numbers):.3e} to {np.max(total_condition_numbers):.3e}"
    )
    print(
        f"Condition number range slab  : {np.min(slab_direct_condition_numbers):.3e} to {np.max(slab_direct_condition_numbers):.3e}"
    )
    print(
        f"|det(T_total)| range         : {np.min(np.abs(determinant_total)):.3e} to {np.max(np.abs(determinant_total)):.3e}"
    )
    print(
        f"|det(T_slab direct)| range   : {np.min(np.abs(determinant_slab_direct)):.3e} to {np.max(np.abs(determinant_slab_direct)):.3e}"
    )
    print(
        f"|det(T_slab decas)| range    : {np.min(np.abs(determinant_slab_decascaded)):.3e} to {np.max(np.abs(determinant_slab_decascaded)):.3e}"
    )
    print(f"Max rigid-end pressure error : {rigid_pressure_error_db:.3e} dB")
    print(f"Max open-end pressure error  : {open_pressure_error_db:.3e} dB")
    print(f"Rigid-end IL range           : {np.min(il_rigid_db):.3f} to {np.max(il_rigid_db):.3f} dB")
    print(f"Median slab matrix mismatch  : {np.median(slab_compare_error):.3e}")
    print(f"Max slab matrix mismatch     : {np.max(slab_compare_error):.3e}")
    print()
    print(
        "f [Hz] | |A_total| | |B_total| | |C_total| | |D_total| | "
        "|A_slab_dir| |A_slab_decas| | rel mismatch |"
    )

    for target_hz in (50.0, 100.0, 1000.0, 5000.0, 10000.0):
        idx = int(np.argmin(np.abs(frequencies_hz - target_hz)))
        total = identified_total[idx]
        slab_direct = identified_slab_direct[idx]
        slab_decascaded = identified_slab_decascaded[idx]
        print(
            f"{frequencies_hz[idx]:7.1f} | "
            f"{abs(total[0, 0]):9.3e} | {abs(total[0, 1]):9.3e} | {abs(total[1, 0]):9.3e} | {abs(total[1, 1]):9.3e} | "
            f"{abs(slab_direct[0, 0]):12.3e} | {abs(slab_decascaded[0, 0]):13.3e} | {slab_compare_error[idx]:12.3e}"
        )
    print()
    print("f [Hz] | |p_air+air_rigid| [Pa] | |p_slab_rigid| [Pa] | IL_rigid [dB]")
    for target_hz in (50.0, 100.0, 1000.0, 5000.0, 10000.0):
        idx = int(np.argmin(np.abs(frequencies_hz - target_hz)))
        print(
            f"{frequencies_hz[idx]:7.1f} | "
            f"{np.abs(p_end_air_air_rigid[idx]):20.6e} | "
            f"{np.abs(p_end_identified_rigid[idx]):17.6e} | "
            f"{il_rigid_db[idx]:12.3f}"
        )


def plot_results(
    frequencies_hz: np.ndarray,
    identified_total: np.ndarray,
    identified_slab_direct: np.ndarray,
    identified_slab_decascaded: np.ndarray,
    total_condition_numbers: np.ndarray,
    slab_direct_condition_numbers: np.ndarray,
    p_end_fem_rigid: np.ndarray,
    p_end_identified_rigid: np.ndarray,
    p_end_identified_direct_rigid: np.ndarray,
    p_end_fem_open: np.ndarray,
    p_end_identified_open: np.ndarray,
    p_end_identified_direct_open: np.ndarray,
    p_end_air_air_rigid: np.ndarray,
) -> None:
    fig_total, axes_total = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))

    for row in range(2):
        for col in range(2):
            label = labels[row][col]
            ax_total = axes_total[row, col]
            ax_total.semilogx(frequencies_hz, np.abs(identified_total[:, row, col]), lw=2.0, label=f"Total {label}")
            ax_total.set_ylabel(f"|{label}|")
            ax_total.grid(True, which="both", alpha=0.3)
            ax_total.legend(loc="best")

    axes_total[2, 0].semilogx(frequencies_hz, total_condition_numbers, lw=2.0, label="cond(total)")
    axes_total[2, 0].set_ylabel("Condition number")
    axes_total[2, 0].set_xlabel("Frequency [Hz]")
    axes_total[2, 0].grid(True, which="both", alpha=0.3)
    axes_total[2, 0].legend(loc="best")

    axes_total[2, 1].semilogx(frequencies_hz, level_db(np.linalg.det(identified_total)), lw=2.0, label="det(total)")
    axes_total[2, 1].set_ylabel(r"$20 \log_{10}|\det(T)|$")
    axes_total[2, 1].set_xlabel("Frequency [Hz]")
    axes_total[2, 1].grid(True, which="both", alpha=0.3)
    axes_total[2, 1].legend(loc="best")
    fig_total.suptitle("Identified Total Matrix")
    plt.show()

    fig_slab, axes_slab = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for row in range(2):
        for col in range(2):
            label = labels[row][col]
            ax_slab = axes_slab[row, col]
            ax_slab.semilogx(
                frequencies_hz,
                np.abs(identified_slab_direct[:, row, col]),
                lw=2.0,
                label=f"Slab direct {label}",
            )
            ax_slab.semilogx(
                frequencies_hz,
                np.abs(identified_slab_decascaded[:, row, col]),
                "--",
                lw=1.8,
                label=f"Slab decas {label}",
            )
            ax_slab.set_ylabel(f"|{label}|")
            ax_slab.grid(True, which="both", alpha=0.3)
            ax_slab.legend(loc="best")
    slab_compare_error = np.linalg.norm(identified_slab_direct - identified_slab_decascaded, axis=(1, 2)) / np.maximum(
        np.linalg.norm(identified_slab_direct, axis=(1, 2)),
        np.finfo(float).tiny,
    )
    axes_slab[2, 0].semilogx(frequencies_hz, slab_direct_condition_numbers, lw=2.0, label="cond(slab direct)")
    axes_slab[2, 0].set_ylabel("Condition number")
    axes_slab[2, 0].set_xlabel("Frequency [Hz]")
    axes_slab[2, 0].grid(True, which="both", alpha=0.3)
    axes_slab[2, 0].legend(loc="best")

    axes_slab[2, 1].semilogx(
        frequencies_hz,
        slab_compare_error,
        lw=2.0,
        color="#dc2626",
        label="Relative mismatch",
    )
    axes_slab[2, 1].set_ylabel("Relative mismatch")
    axes_slab[2, 1].set_xlabel("Frequency [Hz]")
    axes_slab[2, 1].grid(True, which="both", alpha=0.3)
    axes_slab[2, 1].legend(loc="best")
    fig_slab.suptitle("Slab Matrix: Direct vs Decascaded")
    plt.show()

    fig_pressures, axes_pressures = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes_pressures[0].semilogx(frequencies_hz, level_db(p_end_fem_rigid), lw=2.0, label="FEM rigid end")
    axes_pressures[0].semilogx(
        frequencies_hz,
        level_db(p_end_identified_rigid),
        "--",
        lw=1.8,
        label="Identified rigid end",
    )
    axes_pressures[0].set_ylabel(r"$20 \log_{10}|p_{end}|$")
    axes_pressures[0].grid(True, which="both", alpha=0.3)
    axes_pressures[0].legend(loc="best")

    il_rigid_db = level_db(p_end_air_air_rigid) - level_db(p_end_identified_rigid)
    il_rigid_direct_db = level_db(p_end_air_air_rigid) - level_db(p_end_identified_direct_rigid)
    axes_pressures[1].semilogx(frequencies_hz, level_db(p_end_air_air_rigid), lw=2.0, label="Air-air rigid ref")
    axes_pressures[1].semilogx(
        frequencies_hz,
        level_db(p_end_identified_rigid),
        "--",
        lw=1.8,
        label="Decascaded slab rigid",
    )
    axes_pressures[1].semilogx(
        frequencies_hz,
        level_db(p_end_identified_direct_rigid),
        ":",
        lw=1.8,
        label="Direct slab rigid",
    )
    axes_pressures[1].set_ylabel(r"$20 \log_{10}|p_{end}|$")
    axes_pressures[1].grid(True, which="both", alpha=0.3)
    axes_pressures[1].legend(loc="best")

    axes_pressures[2].semilogx(frequencies_hz, level_db(p_end_fem_open), lw=2.0, label="FEM open end")
    axes_pressures[2].semilogx(
        frequencies_hz,
        level_db(p_end_identified_open),
        "--",
        lw=1.8,
        label="Decascaded slab open end",
    )
    axes_pressures[2].semilogx(
        frequencies_hz,
        level_db(p_end_identified_direct_open),
        ":",
        lw=1.8,
        label="Direct slab open end",
    )
    axes_pressures[2].semilogx(frequencies_hz, il_rigid_db, lw=2.0, color="#ea580c", label="IL rigid end")
    axes_pressures[2].semilogx(
        frequencies_hz,
        il_rigid_direct_db,
        ":",
        lw=1.8,
        color="#7c3aed",
        label="IL rigid end direct slab",
    )
    axes_pressures[2].set_xlabel("Frequency [Hz]")
    axes_pressures[2].set_ylabel("Open / IL [dB]")
    axes_pressures[2].grid(True, which="both", alpha=0.3)
    axes_pressures[2].legend(loc="best")
    fig_pressures.suptitle("End Pressure and Rigid-End IL")
    plt.show()






if __name__ == "__main__":
    open_case = load_fem_case(OPEN_PATH)
    rigid_case = load_fem_case(RIGID_PATH)

    if open_case.point_ids != rigid_case.point_ids:
        raise ValueError(
            f"Point-id mismatch between open and rigid cases: {open_case.point_ids} vs {rigid_case.point_ids}")
    if not np.allclose(open_case.frequencies_hz, rigid_case.frequencies_hz):
        raise ValueError("Frequency mismatch between open and rigid FEM cases")

    omega = 2.0 * np.pi * open_case.frequencies_hz
    p0 = 1.0
    z_source = RHO_AIR * C_AIR / AREA_M2
    z_open = z_source
    z_rigid = np.inf

    p_end_fem_rigid = rigid_case.pressures[point_index(rigid_case, POINT_TOTAL_RIGHT)]
    p_end_fem_open = open_case.pressures[point_index(open_case, POINT_TOTAL_RIGHT)]
    air_equivalent = CylindricalDuct(radius=RADIUS_M, length=SLAB_LENGTH_M, c0=C_AIR, rho0=RHO_AIR)
    cavity = CylindricalDuct(radius=RADIUS_M, length=CAVITY_LENGTH_M, c0=C_AIR, rho0=RHO_AIR)
    air_air_total = air_equivalent + cavity
    p_end_air_air_rigid = air_air_total.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]

    left_state_open_total = state_at(open_case, POINT_TOTAL_LEFT)
    left_state_rigid_total = state_at(rigid_case, POINT_TOTAL_LEFT)
    right_state_open_total = state_at(open_case, POINT_TOTAL_RIGHT)
    right_state_rigid_total = state_at(rigid_case, POINT_TOTAL_RIGHT)

    left_state_open_slab = state_at(open_case, POINT_SLAB_LEFT)
    left_state_rigid_slab = state_at(rigid_case, POINT_SLAB_LEFT)
    right_state_open_slab = state_at(open_case, POINT_SLAB_RIGHT)
    right_state_rigid_slab = state_at(rigid_case, POINT_SLAB_RIGHT)

    identified_total, total_condition_numbers = identify_two_port(
        right_state_a=right_state_rigid_total,
        right_state_b=right_state_open_total,
        left_state_a=left_state_rigid_total,
        left_state_b=left_state_open_total,
    )
    identified_slab_direct, slab_direct_condition_numbers = identify_two_port(
        right_state_a=right_state_rigid_slab,
        right_state_b=right_state_open_slab,
        left_state_a=left_state_rigid_slab,
        left_state_b=left_state_open_slab,
    )
    identified_total_element = FrozenMatrixElement(identified_total)
    identified_slab_decascaded = identified_total_element.decascade_right(cavity).matrix(omega)

    identified_slab_direct_element = FrozenMatrixElement(identified_slab_direct)
    identified_slab_direct_total = identified_slab_direct_element + cavity

    p_end_identified_rigid = identified_total_element.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]
    p_end_identified_open = identified_total_element.state_tm_from_incident_wave(p0, z_open, z_source, omega)[:, 0]
    p_end_identified_direct_rigid = identified_slab_direct_total.state_tm_from_incident_wave(p0, z_rigid, z_source, omega)[:, 0]
    p_end_identified_direct_open = identified_slab_direct_total.state_tm_from_incident_wave(p0, z_open, z_source, omega)[:, 0]

    print_summary(
        open_case.frequencies_hz,
        identified_total,
        identified_slab_direct,
        identified_slab_decascaded,
        total_condition_numbers,
        slab_direct_condition_numbers,
        p_end_fem_rigid,
        p_end_identified_rigid,
        p_end_fem_open,
        p_end_identified_open,
        p_end_air_air_rigid,
    )

    np.savez(
        HERE / "same_radius_identified_matrices_from_fem.npz",
        frequencies_hz=open_case.frequencies_hz,
        point_ids=np.asarray(open_case.point_ids, dtype=int),
        identified_total=identified_total,
        identified_slab_direct=identified_slab_direct,
        identified_slab_decascaded=identified_slab_decascaded,
        total_condition_numbers=total_condition_numbers,
        slab_direct_condition_numbers=slab_direct_condition_numbers,
        p_end_fem_rigid=p_end_fem_rigid,
        p_end_identified_rigid=p_end_identified_rigid,
        p_end_fem_open=p_end_fem_open,
        p_end_identified_open=p_end_identified_open,
        p_end_air_air_rigid=p_end_air_air_rigid,
    )

    plot_results(
        open_case.frequencies_hz,
        identified_total,
        identified_slab_direct,
        identified_slab_decascaded,
        total_condition_numbers,
        slab_direct_condition_numbers,
        p_end_fem_rigid,
        p_end_identified_rigid,
        p_end_identified_direct_rigid,
        p_end_fem_open,
        p_end_identified_open,
        p_end_identified_direct_open,
        p_end_air_air_rigid,
    )

    # interestingly the identified_slab_direct and identified_slab_decascaded are different
    # and it seems that identified_slab_direct is the faulty one, propagating evanescent stuff?
