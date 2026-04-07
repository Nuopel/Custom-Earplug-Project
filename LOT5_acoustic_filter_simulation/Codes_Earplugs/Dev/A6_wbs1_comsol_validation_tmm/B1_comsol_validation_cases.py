"""Plot COMSOL validation cases with points ordered like TMM states."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COMSOL_RESULTS_DIR = HERE / "comsol_rslt"

# COMSOL exports points as 5, 6, 7, 8. Reorder them to match the TMM cascade p0 -> p3.
COMSOL_POINT_IDS = (5, 6, 7, 8)
POINT_IDS = (8, 7, 6, 5)
POINT_TITLES = {
    8: "Point 0: in air",
    7: "Point 1: in slab",
    6: "Point 2: out slab",
    5: "Point 3: end",
}
POINT_ORDER = tuple(COMSOL_POINT_IDS.index(point_id) for point_id in POINT_IDS)


def parse_comsol_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def level_db(values: np.ndarray, ref: float) -> np.ndarray:
    magnitude = np.maximum(np.abs(values), np.finfo(float).tiny)
    return 20.0 * np.log10(magnitude / ref)


def insertion_loss_db(reference: np.ndarray, comparison: np.ndarray) -> np.ndarray:
    return level_db(reference, 1.0) - level_db(comparison, 1.0)


def reorder_to_tmm(points_by_row: np.ndarray) -> np.ndarray:
    return points_by_row[np.asarray(POINT_ORDER)]


def load_point_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies_hz = []
    pressures = []
    velocities = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f"Expected 9 columns in {path}, got {len(parts)}")

            frequencies_hz.append(float(parts[0]))
            pressures.append([parse_comsol_complex(parts[idx]) for idx in range(1, 5)])
            velocities.append([parse_comsol_complex(parts[idx]) for idx in range(5, 9)])

    return (
        np.asarray(frequencies_hz, dtype=float),
        reorder_to_tmm(np.asarray(pressures, dtype=np.complex128).T),
        reorder_to_tmm(np.asarray(velocities, dtype=np.complex128).T),
    )


def load_comsol_point_cases(
    open_path: Path,
    rigid_end_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freq_open, p_open, v_open = load_point_file(open_path)
    freq_rigid, p_rigid, v_rigid = load_point_file(rigid_end_path)

    if not np.allclose(freq_open, freq_rigid):
        raise ValueError(f"Frequency mismatch between {open_path.name} and {rigid_end_path.name}")

    return freq_open, p_open, p_rigid, v_open, v_rigid


def compute_shared_limits(first: np.ndarray, second: np.ndarray, ref: float) -> tuple[float, float]:
    stacked = np.concatenate([level_db(first, ref).ravel(), level_db(second, ref).ravel()])
    ymin = np.floor(stacked.min() / 5.0) * 5.0
    ymax = np.ceil(stacked.max() / 5.0) * 5.0
    if np.isclose(ymin, ymax):
        ymin -= 1.0
        ymax += 1.0
    return ymin, ymax


def plot_four_points(
    frequencies_hz: np.ndarray,
    values_open: np.ndarray,
    values_rigid: np.ndarray,
    *,
    ref: float,
    ylabel: str,
    figure_title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    y_limits = compute_shared_limits(values_open, values_rigid, ref)

    for axis, point_id, open_curve, rigid_curve in zip(axes.flat, POINT_IDS, values_open, values_rigid):
        axis.semilogx(
            frequencies_hz,
            level_db(open_curve, ref),
            lw=2.0,
            color="#2563eb",
            label="Open",
        )
        axis.semilogx(
            frequencies_hz,
            level_db(rigid_curve, ref),
            lw=2.0,
            ls="--",
            color="#dc2626",
            label="Rigid end",
        )
        axis.set_title(POINT_TITLES[point_id])
        axis.set_ylim(*y_limits)
        axis.grid(True, which="both", alpha=0.3)

    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    for axis in axes[-1, :]:
        axis.set_xlabel("Frequency [Hz]")

    axes[1, 1].legend(loc="lower right")
    fig.suptitle(figure_title)
    plt.show()


def plot_case(case_label: str, open_filename: str, rigid_filename: str) -> None:
    frequencies_hz, p_open, p_rigid, v_open, v_rigid = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / open_filename,
        COMSOL_RESULTS_DIR / rigid_filename,
    )

    plot_four_points(
        frequencies_hz,
        p_open,
        p_rigid,
        ref=1.0,
        ylabel="Pressure level [dB re 1 Pa]",
        figure_title=f"COMSOL {case_label}: pressure",
    )
    plot_four_points(
        frequencies_hz,
        v_open,
        v_rigid,
        ref=1.0,
        ylabel="Velocity level [dB re 1 m/s]",
        figure_title=f"COMSOL {case_label}: velocity",
    )


def plot_end_insertion_loss(
    frequencies_hz: np.ndarray,
    p_reference: np.ndarray,
    p_freeslab: np.ndarray,
    p_fixedslab: np.ndarray,
    *,
    case_label: str,
) -> None:
    point3_index = POINT_IDS.index(5)

    il_freeslab = insertion_loss_db(
        p_reference[point3_index],
        p_freeslab[point3_index],
    )
    il_fixedslab = insertion_loss_db(
        p_reference[point3_index],
        p_fixedslab[point3_index],
    )

    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.semilogx(
        frequencies_hz,
        il_freeslab,
        lw=2.0,
        color="#16a34a",
        label=f"IL {case_label}: air-air vs air-freeslab-air",
    )
    axis.semilogx(
        frequencies_hz,
        il_fixedslab,
        lw=2.0,
        color="#ea580c",
        label=f"IL {case_label}: air-air vs air-fixedslab-air",
    )
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Insertion loss [dB]")
    axis.set_title(f"End pressure insertion loss at Point 3 ({case_label})")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best")
    plt.show()






if __name__ == "__main__":
    air_air_freq, air_air_p_open, air_air_p_rigid, _, _ = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_air_air_open.txt",
        COMSOL_RESULTS_DIR / "air_air_air_rigidend.txt",
    )
    freeslab_freq, freeslab_p_open, freeslab_p_rigid, _, _ = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_freeslab_air_open.txt",
        COMSOL_RESULTS_DIR / "air_freeslab_air_rigidend.txt",
    )
    fixedslab_freq, fixedslab_p_open, fixedslab_p_rigid, _, _ = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_fixedslab_air_open.txt",
        COMSOL_RESULTS_DIR / "air_fixedslab_air_rigidend.txt",
    )

    if not np.allclose(air_air_freq, freeslab_freq):
        raise ValueError("Frequency mismatch between air-air and air-freeslab-air cases")
    if not np.allclose(air_air_freq, fixedslab_freq):
        raise ValueError("Frequency mismatch between air-air and air-fixedslab-air cases")

    plot_case(
        "air-air case",
        "air_air_air_open.txt",
        "air_air_air_rigidend.txt",
    )
    plot_case(
        "air-freeslab-air case",
        "air_freeslab_air_open.txt",
        "air_freeslab_air_rigidend.txt",
    )
    plot_case(
        "air-fixedslab-air case",
        "air_fixedslab_air_open.txt",
        "air_fixedslab_air_rigidend.txt",
    )
    plot_end_insertion_loss(
        air_air_freq,
        air_air_p_rigid,
        freeslab_p_rigid,
        fixedslab_p_rigid,
        case_label="rigid end",
    )
    plot_end_insertion_loss(
        air_air_freq,
        air_air_p_open,
        freeslab_p_open,
        fixedslab_p_open,
        case_label="open end",
    )
