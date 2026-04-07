"""Compare COMSOL and TMM pressure/velocity for the open-end air-air case."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from B1_comsol_validation_cases import COMSOL_RESULTS_DIR, load_comsol_point_cases
from toolkitsd.acoustmm import CylindricalDuct, RadiationImpedance


POINT_END = 3


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def open_end_tmm_end_state(
    frequencies_hz: np.ndarray,
    *,
    radius: float = 3.75e-3,
    air_only_length: float = 13.0e-3,
    rho_air: float = 1.2,
    c_air: float = 343.0,
    p0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * frequencies_hz
    area = np.pi * radius**2
    z_source = rho_air * c_air / area
    z_open = rho_air * c_air / area

    air_only = CylindricalDuct(radius=radius, length=air_only_length, c0=c_air, rho0=rho_air)
    state_end = air_only.state_tm_from_incident_wave(p0, z_open, z_source, omega)
    return state_end[:, 0], state_end[:, 1] / area


def plot_results(
    frequencies_hz: np.ndarray,
    p_end_comsol: np.ndarray,
    p_end_tmm: np.ndarray,
    v_end_comsol: np.ndarray,
    v_end_tmm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(frequencies_hz, level_db(p_end_comsol), lw=2.0, color="#111827", label="COMSOL")
    axes[0].semilogx(frequencies_hz, level_db(p_end_tmm), "--", lw=2.0, color="#2563eb", label="TMM")
    axes[0].set_ylabel(r"$20 \log_{10}|p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title("Open-end pressure: air-air")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(frequencies_hz, level_db(v_end_comsol), lw=2.0, color="#111827", label="COMSOL")
    axes[1].semilogx(frequencies_hz, level_db(v_end_tmm), "--", lw=2.0, color="#2563eb", label="TMM")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$20 \log_{10}|v_{end}|$ [dB re 1 m/s]")
    axes[1].set_title("Open-end particle velocity: air-air")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


if __name__ == "__main__":
    freqs_hz, p_open, _, v_open, _ = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_air_air_open.txt",
        COMSOL_RESULTS_DIR / "air_air_air_rigidend.txt",
    )

    p_end_comsol = p_open[POINT_END]
    v_end_comsol = v_open[POINT_END]
    p_end_tmm, v_end_tmm = open_end_tmm_end_state(freqs_hz)

    print("=== OPEN-END AIR-AIR: COMSOL VS TMM ===")
    print("Compared quantity             : end pressure and end particle velocity")
    print(f"Max pressure difference       : {np.max(np.abs(level_db(p_end_comsol) - level_db(p_end_tmm))):.3e} dB")
    print(f"Max velocity difference       : {np.max(np.abs(level_db(v_end_comsol) - level_db(v_end_tmm))):.3e} dB")

    plot_results(
        frequencies_hz=freqs_hz,
        p_end_comsol=p_end_comsol,
        p_end_tmm=p_end_tmm,
        v_end_comsol=v_end_comsol,
        v_end_tmm=v_end_tmm,
    )
