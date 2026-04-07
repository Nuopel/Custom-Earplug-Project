"""Incremental FEM pressure loader for the WBS5 three-mic exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FEM_DIR = HERE / "fem_rslt"

POINT_IDS = (6, 7, 8, 9, 10)
POINT_LABELS = {
    6: "Mic 1",
    7: "Mic 2",
    8: "Mic 3",
    9: "Cavity side",
    10: "Rigid end",
    11: "Rigid end + 1",
}


@dataclass(frozen=True)
class CaseConfig:
    name: str
    air_path: Path
    silicone_path: Path


@dataclass(frozen=True)
class PressureCase:
    name: str
    frequencies_hz: np.ndarray
    point_ids: tuple[int, ...]
    pressures_air: np.ndarray
    pressures_silicone: np.ndarray
    air_source: Path
    silicone_source: Path


CASE_CONFIGS = (
    CaseConfig(
        name="same_radius",
        air_path=FEM_DIR / "same_radius_air_air_rigidend.txt",
        silicone_path=FEM_DIR / "same_radius_silicone_air_rigidend.txt",
    ),
    CaseConfig(
        name="diff_radius",
        air_path=FEM_DIR / "diff_radius_air_air_rigidend.txt",
        silicone_path=FEM_DIR / "diff_radius_silicone_air_rigidend.txt",
    ),
)


def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def parse_pressure_point_ids(path: Path) -> tuple[int, ...]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("%") or "freq" not in line or "Total acoustic pressure" not in line:
                continue

            point_ids = []
            for match in re.finditer(r"Total acoustic pressure \(Pa\), Point:\s*(\d+)", line):
                point_ids.append(int(match.group(1)))

            if not point_ids:
                raise ValueError(f"No pressure point ids found in header of {path}")

            return tuple(point_ids)

    raise ValueError(f"Could not find FEM column header in {path}")


def load_fem_pressures(path: Path) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    point_ids = parse_pressure_point_ids(path)
    n_pressure_points = len(point_ids)
    frequencies_hz = []
    pressures = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            expected_min_columns = 1 + 2 * n_pressure_points
            if len(parts) < expected_min_columns:
                raise ValueError(
                    f"Unexpected FEM table format in {path}: expected at least "
                    f"{expected_min_columns} columns, got {len(parts)}"
                )

            frequencies_hz.append(float(parts[0]))
            pressures.append([parse_fem_complex(parts[idx]) for idx in range(1, 1 + n_pressure_points)])

    if not frequencies_hz:
        raise ValueError(f"No FEM data rows found in {path}")

    return (
        np.asarray(frequencies_hz, dtype=float),
        point_ids,
        np.asarray(pressures, dtype=np.complex128).T,
    )


def load_pressure_case(config: CaseConfig) -> PressureCase:
    frequencies_air, point_ids_air, pressures_air = load_fem_pressures(config.air_path)
    frequencies_silicone, point_ids_silicone, pressures_silicone = load_fem_pressures(config.silicone_path)

    if not np.allclose(frequencies_air, frequencies_silicone):
        raise ValueError(f"Frequency mismatch in case {config.name}")
    if point_ids_air != point_ids_silicone:
        raise ValueError(f"Point-id mismatch in case {config.name}: {point_ids_air} vs {point_ids_silicone}")

    return PressureCase(
        name=config.name,
        frequencies_hz=frequencies_air,
        point_ids=point_ids_air,
        pressures_air=pressures_air,
        pressures_silicone=pressures_silicone,
        air_source=config.air_path,
        silicone_source=config.silicone_path,
    )


def print_case_summary(case: PressureCase) -> None:
    print(f"=== {case.name.upper()} ===")
    print(f"air source: {case.air_source.name}")
    print(f"silicone source: {case.silicone_source.name}")
    print(f"n_freqs: {case.frequencies_hz.size}")
    print(
        f"freq range [Hz]: {case.frequencies_hz[0]:.3f} -> {case.frequencies_hz[-1]:.3f}"
    )
    print(f"pressure point ids: {case.point_ids}")
    print("Point | Label        | min |p_air| [Pa] | min |p_silicone| [Pa] | max IL [dB]")

    for point_id, pressure_air, pressure_silicone in zip(case.point_ids, case.pressures_air, case.pressures_silicone):
        il_db = level_db(pressure_air) - level_db(pressure_silicone)
        print(
            f"{point_id:>5} | {POINT_LABELS.get(point_id, f'Point {point_id}'):<12} | "
            f"{np.min(np.abs(pressure_air)):15.6e} | "
            f"{np.min(np.abs(pressure_silicone)):20.6e} | "
            f"{np.max(np.abs(il_db)):11.3f}"
        )

    print("Sample values:")
    point_8_index = min(2, len(case.point_ids) - 1)
    point_10_index = len(case.point_ids) - 1
    point_8_id = case.point_ids[point_8_index]
    point_10_id = case.point_ids[point_10_index]
    print(
        f"f [Hz] | point {point_8_id} |p_air| [Pa] | "
        f"point {point_8_id} |p_silicone| [Pa] | point {point_10_id} IL [dB]"
    )
    for target_hz in (50.0, 100.0, 1000.0, 5000.0, 10000.0):
        idx = int(np.argmin(np.abs(case.frequencies_hz - target_hz)))
        point_8_air = case.pressures_air[point_8_index, idx]
        point_8_silicone = case.pressures_silicone[point_8_index, idx]
        point_10_il_db = (
            level_db(case.pressures_air[point_10_index, idx : idx + 1])
            - level_db(case.pressures_silicone[point_10_index, idx : idx + 1])
        )[0]
        print(
            f"{case.frequencies_hz[idx]:7.1f} | "
            f"{np.abs(point_8_air):18.6e} | "
            f"{np.abs(point_8_silicone):23.6e} | "
            f"{point_10_il_db:14.3f}"
        )
    print()


def plot_cases(cases: list[PressureCase]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    colors = {
        "same_radius": "#2563eb",
        "diff_radius": "#dc2626",
    }
    for case in cases:
        color = colors.get(case.name, None)
        point_8_index = min(2, len(case.point_ids) - 1)
        point_10_index = len(case.point_ids) - 1
        point_8_id = case.point_ids[point_8_index]
        point_10_id = case.point_ids[point_10_index]
        axes[0].semilogx(
            case.frequencies_hz,
            level_db(case.pressures_air[point_8_index]),
            lw=2.0,
            color=color,
            label=f"{case.name} air, point {point_8_id}",
        )
        axes[0].semilogx(
            case.frequencies_hz,
            level_db(case.pressures_silicone[point_8_index]),
            "--",
            lw=2.0,
            color=color,
            label=f"{case.name} silicone, point {point_8_id}",
        )

        il_db = level_db(case.pressures_air[point_10_index]) - level_db(case.pressures_silicone[point_10_index])
        axes[1].semilogx(
            case.frequencies_hz,
            il_db,
            lw=2.0,
            color=color,
            label=f"{case.name} rigid-end IL, point {point_10_id}",
        )

    axes[0].set_ylabel(r"$20 \log_{10}|p|$ [dB re 1 Pa]")
    axes[0].set_title("FEM loaded pressures")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Air vs silicone pressure ratio at rigid end")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


def main() -> None:
    cases = []
    failures: list[tuple[str, str]] = []

    for config in CASE_CONFIGS:
        try:
            cases.append(load_pressure_case(config))
        except ValueError as exc:
            failures.append((config.name, str(exc)))

    for case in cases:
        print_case_summary(case)

    if failures:
        print("=== UNAVAILABLE CASES ===")
        for case_name, message in failures:
            print(f"{case_name}: {message}")
        print()

    if not cases:
        raise RuntimeError("No valid FEM pressure cases could be loaded")

    plot_cases(cases)


if __name__ == "__main__":
    main()
