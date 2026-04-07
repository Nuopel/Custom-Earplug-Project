"""Plot end-point SPL and insertion loss from FEM rigid-end microphone results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FEM_RESULTS_DIR = HERE / "fem_rslt/rigid_end/"
P_REF = 20.0e-6

CASE_FILES = {
    "Case A: silicone slab filter": "rslt_fem_A2_caseA_silicone_slab_filter_in_duct_rigidend.txt",
    "Case B: rigid slab filter": "rslt_fem_A2_caseB_rigid_slab_filter_in_duct_rigidend.txt",
    "Case C: silicone slab": "rslt_fem_A2_caseC_silicone_slab_in_duct_rigidend.txt",
    "Case D: air slab": "rslt_fem_A2_caseD_air_slab_in_duct_rigidend.txt",
    "Case E: rigid slab filter film R=5e8, K=3e12, M=10": "rslt_fem_A2_caseE_rigid_slab_filter_filmrkm20_in_duct_rigidend.txt",
    "Case F: silicone slab filter film R=5e8, K=3e12, M=10": "rslt_fem_A2_caseF_silicone_slab_filter_filmrkm20_in_duct_rigidend.txt",
}

def parse_fem_complex(token: str) -> complex:
    return complex(token.replace("i", "j"))


def spl_db(values: np.ndarray) -> np.ndarray:
    magnitude = np.maximum(np.abs(values), np.finfo(float).tiny)
    return 20.0 * np.log10(magnitude / P_REF)


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
            end_pressure.append(parse_fem_complex(parts[1]))

    return np.asarray(frequencies_hz, dtype=float), np.asarray(end_pressure, dtype=np.complex128)


def load_case_pressures() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference_freq: np.ndarray | None = None
    pressures: dict[str, np.ndarray] = {}

    for label, filename in CASE_FILES.items():
        freqs, pressure = load_end_pressure(FEM_RESULTS_DIR / filename)
        if reference_freq is None:
            reference_freq = freqs
        elif not np.allclose(reference_freq, freqs):
            raise ValueError(f"Frequency mismatch for {label}")
        pressures[label] = pressure

    if reference_freq is None:
        raise ValueError("No FEM result files were loaded")
    return reference_freq, pressures


if __name__ == "__main__":
    freqs, case_pressures = load_case_pressures()

    case_spl = {label: spl_db(pressure) for label, pressure in case_pressures.items()}
    reference_label = "Case D: air slab"
    reference_spl = case_spl[reference_label]
    il_db = {
        label: reference_spl - level_db
        for label, level_db in case_spl.items()
        if label != reference_label
    }

    fig_spl, axis_spl = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for label, level_db in case_spl.items():
        axis_spl.semilogx(freqs, level_db, lw=2.0, label=label)
    axis_spl.set_xlabel("Frequency [Hz]")
    axis_spl.set_ylabel("Pressure level [dB SPL]")
    axis_spl.set_title("Rigid-end microphone pressure for cases A-F")
    axis_spl.grid(True, which="both", alpha=0.3)
    axis_spl.legend(loc="best")
    plt.show()

    fig_il, axis_il = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for label, values_db in il_db.items():
        axis_il.semilogx(freqs, values_db, lw=2.0, label=f"IL = Case D - {label}")
    axis_il.set_xlabel("Frequency [Hz]")
    axis_il.set_ylabel("Insertion loss [dB]")
    axis_il.set_title("Insertion loss relative to Case D air slab")
    axis_il.grid(True, which="both", alpha=0.3)
    axis_il.legend(loc="best")
    axis_il.set_ylim(-10,50)

    plt.show()
