from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
candidate_paths.extend(sorted((ROOT / "toolkitsd").glob("Toolkitsd_*/src")))
candidate_paths.extend(sorted((ROOT.parent / "toolkitsd").glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import (
    AcousticParameters,
    CylindricalDuct,
    ElasticSlab,
    GeometryConfig,
    ThreeMicPostProcessor,
)


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def phase_deg(values: np.ndarray) -> np.ndarray:
    return np.angle(values, deg=True)


def pressure_from_state_at_distance(
    p_in: np.ndarray,
    U_in: np.ndarray,
    *,
    z_char: np.ndarray,
    k: np.ndarray,
    x: float,
) -> np.ndarray:
    p_plus = 0.5 * (p_in + z_char * U_in)
    p_minus = 0.5 * (p_in - z_char * U_in)
    return p_plus * np.exp(-1j * k * x) + p_minus * np.exp(1j * k * x)


def plot_mic_pressures(
    freqs_hz: np.ndarray,
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p3_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    p3_b: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    for axis, title, pa, pb in (
        (axes[0], "mic1", p1_a, p1_b),
        (axes[1], "mic2", p2_a, p2_b),
        (axes[2], "mic3", p3_a, p3_b),
    ):
        axis.semilogx(freqs_hz, level_db(pa), lw=2.0, label=f"{title}, load A")
        axis.semilogx(freqs_hz, level_db(pb), "--", lw=2.0, label=f"{title}, load B")
        axis.set_ylabel("|p| [dB]")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(loc="best")

    axes[0].set_title("Synthetic microphone pressures")
    axes[-1].set_xlabel("Frequency [Hz]")
    plt.show()


def plot_matrix_comparison(freqs_hz: np.ndarray, matrix_id: np.ndarray, matrix_true: np.ndarray) -> None:
    labels = (("T11", 0, 0), ("T12", 0, 1), ("T21", 1, 0), ("T22", 1, 1))
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True, constrained_layout=True)

    for row, (label, i, j) in enumerate(labels):
        ax_mag = axes[row, 0]
        ax_phase = axes[row, 1]

        ax_mag.semilogx(freqs_hz, level_db(matrix_true[:, i, j]), lw=2.0, label="Direct TMM")
        ax_mag.semilogx(freqs_hz, level_db(matrix_id[:, i, j]), "--", lw=2.0, label="3 mics / 2 loads")
        ax_mag.set_ylabel(f"{label} [dB]")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend(loc="best")

        ax_phase.semilogx(freqs_hz, phase_deg(matrix_true[:, i, j]), lw=2.0, label="Direct TMM")
        ax_phase.semilogx(freqs_hz, phase_deg(matrix_id[:, i, j]), "--", lw=2.0, label="3 mics / 2 loads")
        ax_phase.set_ylabel(f"{label} [deg]")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.legend(loc="best")

    axes[0, 0].set_title("Magnitude")
    axes[0, 1].set_title("Phase")
    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    plt.show()


if __name__ == "__main__":
    freqs_hz = np.logspace(np.log10(50.0), np.log10(15811.388), 300)
    params = AcousticParameters(freqs_hz, c0=344.96, rho0=1.2)
    geometry = GeometryConfig(
        l1=25.0e-3,
        l2=45.0e-3,
        l_slab=6.6e-3,
        l_cav=6.4e-3,
        l_load_a=11.0e-3,
        l_load_b=36.0e-3,
        r_tube=3.75e-3,
        r_slab=3.75e-3,
    )
    post_processor = ThreeMicPostProcessor(params=params, geometry=geometry)

    omega = 2.0 * np.pi * freqs_hz
    z_source = np.full(freqs_hz.shape, params.z0 / geometry.s_tube + 0j, dtype=np.complex128)
    z_load = np.full(freqs_hz.shape, 1e20 + 0j, dtype=np.complex128)
    z_char = np.full(freqs_hz.shape, params.z0 / geometry.s_tube + 0j, dtype=np.complex128)

    upstream_air = CylindricalDuct(
        radius=geometry.r_tube,
        length=geometry.l1 + geometry.l2,
        c0=params.c0,
        rho0=params.rho0,
    )
    slab = ElasticSlab(
        radius=geometry.r_slab,
        length=geometry.l_slab,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )

    mic_pressures: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for load_length in (geometry.l_load_a, geometry.l_load_b):
        load_duct = CylindricalDuct(
            radius=geometry.r_tube,
            length=load_length,
            c0=params.c0,
            rho0=params.rho0,
        )
        total_system = upstream_air + slab + load_duct
        state_in = total_system.state_in_from_incident_wave(1.0, z_load, z_source, omega)
        state_tm = total_system.state_tm_from_incident_wave(1.0, z_load, z_source, omega)

        p1 = state_in[:, 0]
        U1 = state_in[:, 1]
        p2 = pressure_from_state_at_distance(
            p1,
            U1,
            z_char=z_char,
            k=params.wavenumbers,
            x=geometry.l1,
        )
        p3 = state_tm[:, 0]
        mic_pressures.append((p1, p2, p3))

    (p1_a, p2_a, p3_a), (p1_b, p2_b, p3_b) = mic_pressures

    h12 = np.column_stack((p2_a / p1_a, p2_b / p1_b))
    h13 = np.column_stack((p3_a / p1_a, p3_b / p1_b))

    identified_slab = post_processor.identify_transfer_element_from_h_two_loads(
        h12,
        h13,
        k_tube=params.wavenumbers,
        z_tube=np.full(freqs_hz.shape, params.z0 + 0j, dtype=np.complex128),
        l1=geometry.l1,
        l2_by_load=np.full(2, geometry.l2, dtype=float),
        l3_by_load=np.array([geometry.l_load_a, geometry.l_load_b], dtype=float),
        s_tube=geometry.s_tube,
        s_eff=geometry.s_slab,
        return_basis="pu",
    )

    matrix_direct = slab.matrix(omega)
    matrix_identified = identified_slab.matrix(omega)
    rel_error = np.linalg.norm(matrix_identified - matrix_direct, axis=(1, 2)) / np.maximum(
        np.linalg.norm(matrix_direct, axis=(1, 2)),
        np.finfo(float).tiny,
    )

    print("=== Minimal pure TMM 3-mics / 2-load identification ===")
    print(f"median relative error : {np.median(rel_error):.6e}")
    print(f"max relative error    : {np.max(rel_error):.6e}")

    plot_mic_pressures(freqs_hz, p1_a, p2_a, p3_a, p1_b, p2_b, p3_b)
    plot_matrix_comparison(freqs_hz, matrix_identified, matrix_direct)
