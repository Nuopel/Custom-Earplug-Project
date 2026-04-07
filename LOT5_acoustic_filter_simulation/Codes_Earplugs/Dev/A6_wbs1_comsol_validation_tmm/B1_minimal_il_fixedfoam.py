"""Minimal IL plot for rigid-end air-air vs air-fixedfoam-air at the end point."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COMSOL_RESULTS_DIR = HERE / "comsol_rslt"
POINT3_COMSOL_INDEX = 0  # COMSOL Point 5 == TMM end point p3


def parse_comsol_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def level_db(values: np.ndarray) -> np.ndarray:
    magnitude = np.maximum(np.abs(values), np.finfo(float).tiny)
    return 20.0 * np.log10(magnitude)


def load_end_pressure(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frequencies_hz = []
    end_pressure = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()
            frequencies_hz.append(float(parts[0]))
            end_pressure.append(parse_comsol_complex(parts[1 + POINT3_COMSOL_INDEX]))

    return np.asarray(frequencies_hz, dtype=float), np.asarray(end_pressure, dtype=np.complex128)


if __name__ == "__main__":
    air_air_freq, air_air_p_end = load_end_pressure(COMSOL_RESULTS_DIR / "air_air_air_rigidend.txt")
    fixedfoam_freq, fixedfoam_p_end = load_end_pressure(COMSOL_RESULTS_DIR / "air_fixedfoam_air_rigidend.txt")

    if not np.allclose(air_air_freq, fixedfoam_freq):
        raise ValueError("Frequency mismatch between air-air and air-fixedfoam-air rigid-end cases")

    il_db = level_db(air_air_p_end) - level_db(fixedfoam_p_end)

    fig, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.semilogx(air_air_freq, il_db, lw=2.0, color="#ea580c")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("Insertion loss [dB]")
    axis.set_title("IL at end point: air-air rigid end vs air-fixedfoam-air rigid end")
    axis.grid(True, which="both", alpha=0.3)
    plt.ylim(50, -10)  # inverted axis
    plt.xlim(100, 6000)  # inverted axis

    plt.show()
