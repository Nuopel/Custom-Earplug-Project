"""Compare WBS6 FEM-identified lossy-duct matrices against TMM models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import BLIDuct, CylindricalDuct, ViscothermalDuct

RESULTS_DIR = HERE / "fem_rslt"
FIGURES_DIR = HERE / "figures"

POINT_INLET = 8
POINT_OUTLET = 9
POINT_LOAD = 10

# WBS6 current bench assumption. Change here if the FEM duct length changes.
DUCT_LENGTH_M = 6.6e-3
C0 = 343.0
RHO0 = 1.2


@dataclass(frozen=True)
class FemCase:
    frequencies_hz: np.ndarray
    point_ids: tuple[int, ...]
    pressures_pa: np.ndarray
    velocities_ms: np.ndarray


@dataclass(frozen=True)
class RadiusCase:
    label: str
    radius_m: float
    rigid_path: Path
    open_path: Path

    @property
    def area_m2(self) -> float:
        return np.pi * self.radius_m**2


CASES = (
    RadiusCase(
        label="r = 0.5 mm",
        radius_m=0.5e-3,
        rigid_path=RESULTS_DIR / "lossyduct_radius0p5mm_rigid_end.txt",
        open_path=RESULTS_DIR / "lossyduct_radius0p5mm_open_end.txt",
    ),
    RadiusCase(
        label="r = 3.75 mm",
        radius_m=3.75e-3,
        rigid_path=RESULTS_DIR / "lossyduct_radius3p75mm_rigid_end.txt",
        open_path=RESULTS_DIR / "lossyduct_radius3p75mm_open_end.txt",
    ),
)


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
        pressures_pa=np.asarray(pressures, dtype=np.complex128).T,
        velocities_ms=np.asarray(velocities, dtype=np.complex128).T,
    )


def point_index(case: FemCase, point_id: int) -> int:
    try:
        return case.point_ids.index(point_id)
    except ValueError as exc:
        raise ValueError(f"Point {point_id} not found in FEM file. Available points: {case.point_ids}") from exc


def state_at(case: FemCase, point_id: int, *, area_m2: float) -> np.ndarray:
    idx = point_index(case, point_id)
    return np.column_stack((case.pressures_pa[idx], area_m2 * case.velocities_ms[idx]))


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


def relative_matrix_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.linalg.norm(candidate - reference, axis=(1, 2)) / np.maximum(
        np.linalg.norm(reference, axis=(1, 2)),
        np.finfo(float).tiny,
    )


def predict_left_states(matrix_stack: np.ndarray, right_state: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", matrix_stack, right_state)


def summarize_case(
    radius_case: RadiusCase,
    frequencies_hz: np.ndarray,
    identified: np.ndarray,
    cond_numbers: np.ndarray,
    model_matrices: dict[str, np.ndarray],
    left_state_rigid: np.ndarray,
    left_state_open: np.ndarray,
    right_state_rigid: np.ndarray,
    right_state_open: np.ndarray,
) -> None:
    predicted_left_ident_rigid = predict_left_states(identified, right_state_rigid)
    predicted_left_ident_open = predict_left_states(identified, right_state_open)

    print(f"=== WBS6 LOSSY DUCT: {radius_case.label} ===")
    print(f"Rigid file                     : {radius_case.rigid_path.name}")
    print(f"Open file                      : {radius_case.open_path.name}")
    print(f"Duct length used in TMM        : {DUCT_LENGTH_M * 1e3:.3f} mm")
    print(f"Frequency range [Hz]           : {frequencies_hz[0]:.3f} -> {frequencies_hz[-1]:.3f}")
    print(f"Condition number range         : {np.min(cond_numbers):.3e} to {np.max(cond_numbers):.3e}")
    print(f"|det(T_identified)| range      : {np.min(np.abs(np.linalg.det(identified))):.3e} to {np.max(np.abs(np.linalg.det(identified))):.3e}")
    print(
        f"Max inlet pressure err rigid   : {np.max(np.abs(level_db(predicted_left_ident_rigid[:, 0]) - level_db(left_state_rigid[:, 0]))):.3e} dB"
    )
    print(
        f"Max inlet pressure err open    : {np.max(np.abs(level_db(predicted_left_ident_open[:, 0]) - level_db(left_state_open[:, 0]))):.3e} dB"
    )

    for model_name, matrix_stack in model_matrices.items():
        rel_error = relative_matrix_error(identified, matrix_stack)
        predicted_rigid = predict_left_states(matrix_stack, right_state_rigid)
        predicted_open = predict_left_states(matrix_stack, right_state_open)
        rigid_pressure_error_db = np.max(np.abs(level_db(predicted_rigid[:, 0]) - level_db(left_state_rigid[:, 0])))
        open_pressure_error_db = np.max(np.abs(level_db(predicted_open[:, 0]) - level_db(left_state_open[:, 0])))
        print(
            f"{model_name:30s}: median rel |T-Tid|={np.median(rel_error):.3e}, "
            f"max inlet p err rigid={rigid_pressure_error_db:.3e} dB, "
            f"open={open_pressure_error_db:.3e} dB"
        )

    print()


def plot_case(
    radius_case: RadiusCase,
    frequencies_hz: np.ndarray,
    identified: np.ndarray,
    cond_numbers: np.ndarray,
    model_matrices: dict[str, np.ndarray],
    left_state_rigid: np.ndarray,
    left_state_open: np.ndarray,
    right_state_rigid: np.ndarray,
    right_state_open: np.ndarray,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = radius_case.label.replace(" ", "_").replace("=", "").replace(".", "p")

    labels = (("A", "B"), ("C", "D"))
    fig_matrix, axes_matrix = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for row in range(2):
        for col in range(2):
            ax = axes_matrix[row, col]
            label = labels[row][col]
            ax.semilogx(frequencies_hz, np.abs(identified[:, row, col]), lw=2.5, label="FEM identified")
            for model_name, matrix_stack in model_matrices.items():
                ax.semilogx(frequencies_hz, np.abs(matrix_stack[:, row, col]), lw=1.7, label=model_name)
            ax.set_ylabel(f"|{label}|")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best")

    axes_matrix[2, 0].semilogx(frequencies_hz, cond_numbers, lw=2.0, color="black", label="cond(R)")
    axes_matrix[2, 0].set_ylabel("Condition number")
    axes_matrix[2, 0].set_xlabel("Frequency [Hz]")
    axes_matrix[2, 0].grid(True, which="both", alpha=0.3)
    axes_matrix[2, 0].legend(loc="best")

    axes_matrix[2, 1].semilogx(frequencies_hz, np.abs(np.linalg.det(identified)), lw=2.0, color="black", label="|det(T_ident)|")
    for model_name, matrix_stack in model_matrices.items():
        axes_matrix[2, 1].semilogx(frequencies_hz, np.abs(np.linalg.det(matrix_stack)), lw=1.6, label=f"|det({model_name})|")
    axes_matrix[2, 1].set_ylabel(r"$|\det(T)|$")
    axes_matrix[2, 1].set_xlabel("Frequency [Hz]")
    axes_matrix[2, 1].grid(True, which="both", alpha=0.3)
    axes_matrix[2, 1].legend(loc="best")
    fig_matrix.suptitle(f"WBS6 matrix comparison ({radius_case.label})")
    plt.show()

    fig_pressure, axes_pressure = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes_pressure[0].semilogx(frequencies_hz, level_db(left_state_rigid[:, 0]), lw=2.5, label="FEM rigid")
    axes_pressure[0].semilogx(
        frequencies_hz,
        level_db(predict_left_states(identified, right_state_rigid)[:, 0]),
        lw=2.0,
        label="Identified rigid",
    )
    axes_pressure[0].semilogx(frequencies_hz, level_db(left_state_open[:, 0]), lw=2.5, ls="--", label="FEM open")
    axes_pressure[0].semilogx(
        frequencies_hz,
        level_db(predict_left_states(identified, right_state_open)[:, 0]),
        lw=2.0,
        ls="--",
        label="Identified open",
    )
    for model_name, matrix_stack in model_matrices.items():
        axes_pressure[0].semilogx(
            frequencies_hz,
            level_db(predict_left_states(matrix_stack, right_state_rigid)[:, 0]),
            lw=1.6,
            label=f"{model_name} rigid",
        )
        axes_pressure[0].semilogx(
            frequencies_hz,
            level_db(predict_left_states(matrix_stack, right_state_open)[:, 0]),
            lw=1.6,
            ls="--",
            label=f"{model_name} open",
        )
    axes_pressure[0].set_ylabel(r"$20 \log_{10}|p_{in}|$ [dB re 1 Pa]")
    axes_pressure[0].grid(True, which="both", alpha=0.3)
    axes_pressure[0].legend(loc="best", ncol=2)

    axes_pressure[1].semilogx(frequencies_hz, np.abs(left_state_rigid[:, 1]), lw=2.5, label="FEM rigid")
    axes_pressure[1].semilogx(
        frequencies_hz,
        np.abs(predict_left_states(identified, right_state_rigid)[:, 1]),
        lw=2.0,
        label="Identified rigid",
    )
    axes_pressure[1].semilogx(frequencies_hz, np.abs(left_state_open[:, 1]), lw=2.5, ls="--", label="FEM open")
    axes_pressure[1].semilogx(
        frequencies_hz,
        np.abs(predict_left_states(identified, right_state_open)[:, 1]),
        lw=2.0,
        ls="--",
        label="Identified open",
    )
    for model_name, matrix_stack in model_matrices.items():
        axes_pressure[1].semilogx(
            frequencies_hz,
            np.abs(predict_left_states(matrix_stack, right_state_rigid)[:, 1]),
            lw=1.6,
            label=f"{model_name} rigid",
        )
        axes_pressure[1].semilogx(
            frequencies_hz,
            np.abs(predict_left_states(matrix_stack, right_state_open)[:, 1]),
            lw=1.6,
            ls="--",
            label=f"{model_name} open",
        )
    axes_pressure[1].set_ylabel(r"$|U_{in}|$ [m$^3$/s]")
    axes_pressure[1].set_xlabel("Frequency [Hz]")
    axes_pressure[1].grid(True, which="both", alpha=0.3)
    axes_pressure[1].legend(loc="best", ncol=2)
    fig_pressure.suptitle(f"WBS6 inlet-state reconstruction ({radius_case.label})")
    plt.show()




def ensure_compatible_cases(rigid_case: FemCase, open_case: FemCase) -> None:
    if rigid_case.point_ids != open_case.point_ids:
        raise ValueError(f"Rigid/open point ids differ: {rigid_case.point_ids} vs {open_case.point_ids}")
    if rigid_case.frequencies_hz.shape != open_case.frequencies_hz.shape or not np.allclose(
        rigid_case.frequencies_hz,
        open_case.frequencies_hz,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Rigid/open frequency grids differ")



def run_case(radius_case: RadiusCase) -> None:
    rigid_case = load_fem_case(radius_case.rigid_path)
    open_case = load_fem_case(radius_case.open_path)
    ensure_compatible_cases(rigid_case, open_case)

    freqs_hz = rigid_case.frequencies_hz
    omega = 2.0 * np.pi * freqs_hz
    area_m2 = radius_case.area_m2

    left_state_rigid = state_at(rigid_case, POINT_INLET, area_m2=area_m2)
    right_state_rigid = state_at(rigid_case, POINT_OUTLET, area_m2=area_m2)
    left_state_open = state_at(open_case, POINT_INLET, area_m2=area_m2)
    right_state_open = state_at(open_case, POINT_OUTLET, area_m2=area_m2)

    identified, cond_numbers = identify_two_port(
        right_state_a=right_state_rigid,
        right_state_b=right_state_open,
        left_state_a=left_state_rigid,
        left_state_b=left_state_open,
    )

    model_matrices = {
        "ViscothermalDuct": ViscothermalDuct(radius=radius_case.radius_m, length=DUCT_LENGTH_M, c0=C0, rho0=RHO0).matrix(omega),
        "BLIDuct": BLIDuct(radius=radius_case.radius_m, length=DUCT_LENGTH_M, c0=C0, rho0=RHO0).matrix(omega),
        "CylindricalDuct": CylindricalDuct(radius=radius_case.radius_m, length=DUCT_LENGTH_M, c0=C0, rho0=RHO0).matrix(omega),
    }

    summarize_case(
        radius_case=radius_case,
        frequencies_hz=freqs_hz,
        identified=identified,
        cond_numbers=cond_numbers,
        model_matrices=model_matrices,
        left_state_rigid=left_state_rigid,
        left_state_open=left_state_open,
        right_state_rigid=right_state_rigid,
        right_state_open=right_state_open,
    )
    plot_case(
        radius_case=radius_case,
        frequencies_hz=freqs_hz,
        identified=identified,
        cond_numbers=cond_numbers,
        model_matrices=model_matrices,
        left_state_rigid=left_state_rigid,
        left_state_open=left_state_open,
        right_state_rigid=right_state_rigid,
        right_state_open=right_state_open,
    )



def main() -> None:
    for radius_case in CASES:
        run_case(radius_case)
    plt.show()


if __name__ == "__main__":
    main()
