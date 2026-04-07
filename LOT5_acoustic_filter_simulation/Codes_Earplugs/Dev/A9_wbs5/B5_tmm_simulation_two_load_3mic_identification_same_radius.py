"""Pure TMM three-mic / two-load slab identification.

This script does not load FEM data. It:
1. builds a full TMM model for two load lengths,
2. synthesizes mic1, mic2, and mic3 pressures,
3. reconstructs boundary states with the 3-mic post-processor,
4. identifies the slab transfer matrix,
5. compares the identified matrix against the true TMM slab matrix in the
   same identification convention, in both magnitude and phase,
6. compares reconstructed boundary states against the direct TMM boundary states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, ElasticSlab, GeometryConfig, ThreeMicPostProcessor


FREQ_MIN_HZ = 50.0
FREQ_MAX_HZ = 15811.388
N_FREQS = 300

P0 = 1.0
RIGID_IMPEDANCE = 1e20
SOURCE_IMPEDANCE_MODE = "tube"  # "tube" or "same_as_load"

C_AIR = 344.96
RHO_AIR = 1.2

SLAB_RHO = 1500.0
SLAB_YOUNG = 2.9e6
SLAB_POISSON = 0.49
SLAB_LOSS_FACTOR = 0.20


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def phase_deg(values: np.ndarray) -> np.ndarray:
    return (np.angle(values)) * 180.0 / np.pi


def build_geometry() -> GeometryConfig:
    return GeometryConfig(
        l1=25.0e-3,
        l2=45.0e-3,
        l_slab=6.6e-3,
        l_cav=6.4e-3,
        l_load_a=11e-3,
        l_load_b=36e-3,
        r_tube=3.75e-3,
        r_slab=3.75e-3,
    )


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


def slab_matrix_pu_to_identification_basis(matrix_pu: np.ndarray, area_m2: float) -> np.ndarray:
    matrix_id = np.array(matrix_pu, copy=True)
    matrix_id[:, 0, 1] = matrix_id[:, 0, 1] * area_m2
    matrix_id[:, 1, 0] =  matrix_id[:, 1, 0] / area_m2
    return matrix_id


def solve_right_state_from_left(matrix: np.ndarray, left_state: np.ndarray) -> np.ndarray:
    right_state = np.empty_like(left_state)
    for idx in range(matrix.shape[0]):
        right_state[idx] = np.linalg.solve(matrix[idx], left_state[idx])
    return right_state


def simulate_three_mic_pressures_and_boundary_states(
    *,
    load_length: float,
    geometry: GeometryConfig,
    params: AcousticParameters,
    z_source: np.ndarray,
    z_load: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * params.frequencies
    k = params.wavenumbers
    z_char = np.full_like(omega, params.z0 / geometry.s_tube, dtype=np.complex128)

    upstream_air = CylindricalDuct(
        radius=geometry.r_tube,
        length=geometry.l1 + geometry.l2,
        c0=params.c0,
        rho0=params.rho0,
    )
    slab = ElasticSlab(
        radius=geometry.r_slab,
        length=geometry.l_slab,
        rho=SLAB_RHO,
        young=SLAB_YOUNG,
        poisson=SLAB_POISSON,
        loss_factor=SLAB_LOSS_FACTOR,
    )
    load_duct = CylindricalDuct(
        radius=geometry.r_tube,
        length=load_length,
        c0=params.c0,
        rho0=params.rho0,
    )
    total = upstream_air + slab + load_duct

    state_in_total = total.state_in_from_incident_wave(P0, z_load, z_source, omega)
    state_tm_total = total.state_tm_from_incident_wave(P0, z_load, z_source, omega)

    p1 = state_in_total[:, 0]
    U1 = state_in_total[:, 1]
    p2 = pressure_from_state_at_distance(p1, U1, z_char=z_char, k=k, x=geometry.l1)
    p3 = state_tm_total[:, 0]

    t_up = upstream_air.matrix(omega)
    t_load = load_duct.matrix(omega)
    entrance_state = solve_right_state_from_left(t_up, state_in_total)
    exit_state = np.einsum("nij,nj->ni", t_load, state_tm_total)

    p0 = entrance_state[:, 0]
    v0 = entrance_state[:, 1] / geometry.s_tube
    pl = exit_state[:, 0]
    vl = exit_state[:, 1] / geometry.s_tube

    return p1, p2, p3, p0, v0, pl, vl


def plot_mic_pressures(
    freqs_hz: np.ndarray,
    p1_a: np.ndarray,
    p2_a: np.ndarray,
    p3_a: np.ndarray,
    p1_b: np.ndarray,
    p2_b: np.ndarray,
    p3_b: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)

    for axis, (title, pa, pb) in zip(
        axes,
        (
            ("mic1", p1_a, p1_b),
            ("mic2", p2_a, p2_b),
            ("mic3", p3_a, p3_b),
        ),
    ):
        phase_axis = axis.twinx()

        mag_a, = axis.semilogx(freqs_hz, level_db(pa), lw=2.0, color="#1f77b4", label=f"{title} |A|")
        mag_b, = axis.semilogx(freqs_hz, level_db(pb), "--", lw=2.0, color="#ff7f0e", label=f"{title} |B|")
        ph_a, = phase_axis.semilogx(freqs_hz, phase_deg(pa), lw=1.5, color="#2ca02c", alpha=0.9, label=f"{title} phase A")
        ph_b, = phase_axis.semilogx(freqs_hz, phase_deg(pb), ":", lw=1.5, color="#d62728", alpha=0.9, label=f"{title} phase B")

        axis.set_ylabel(f"{title} |p| [dB]")
        phase_axis.set_ylabel(f"{title} phase [deg]")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend([mag_a, mag_b, ph_a, ph_b], [h.get_label() for h in (mag_a, mag_b, ph_a, ph_b)], loc="best")

    axes[-1].set_xlabel("Frequency [Hz]")
    plt.show()


def plot_boundary_state_comparison(
    freqs_hz: np.ndarray,
    reconstructed: dict[str, np.ndarray],
    direct: dict[str, np.ndarray],
) -> None:
    order = ("p0_a", "v0_a", "pl_a", "vl_a", "p0_b", "v0_b", "pl_b", "vl_b")
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, constrained_layout=True)

    for ax, key in zip(axes.flat, order):
        phase_axis = ax.twinx()
        mag_rec, = ax.semilogx(freqs_hz, level_db(reconstructed[key]), lw=2.0, color="#1f77b4", label=f"{key} reconstructed |.|")
        mag_dir, = ax.semilogx(freqs_hz, level_db(direct[key]), "--", lw=2.0, color="#ff7f0e", label=f"{key} direct |.|")
        ph_rec, = phase_axis.semilogx(freqs_hz, phase_deg(reconstructed[key]), lw=1.5, color="#2ca02c", alpha=0.9, label=f"{key} reconstructed phase")
        ph_dir, = phase_axis.semilogx(freqs_hz, phase_deg(direct[key]), ":", lw=1.5, color="#d62728", alpha=0.9, label=f"{key} direct phase")
        ax.set_title(key)
        ax.set_ylabel("Magnitude [dB]")
        phase_axis.set_ylabel("Phase [deg]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend([mag_rec, mag_dir, ph_rec, ph_dir], [h.get_label() for h in (mag_rec, mag_dir, ph_rec, ph_dir)], loc="best")

    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    plt.show()


def plot_matrix_comparison(
    freqs_hz: np.ndarray,
    identified_matrix: np.ndarray,
    true_matrix: np.ndarray,
) -> None:
    labels = (("T11", 0, 0), ("T12", 0, 1), ("T21", 1, 0), ("T22", 1, 1))
    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True, constrained_layout=True)

    for row, (label, i, j) in enumerate(labels):
        ax_mag = axes[row, 0]
        ax_phase = axes[row, 1]

        ax_mag.semilogx(freqs_hz, level_db(true_matrix[:, i, j]), lw=2.0, label=f"true {label}")
        ax_mag.semilogx(freqs_hz, level_db(identified_matrix[:, i, j]), "--", lw=2.0, label=f"identified {label}")
        ax_mag.set_ylabel(f"{label} [dB]")
        ax_mag.grid(True, which="both", alpha=0.3)
        ax_mag.legend(loc="best")

        ax_phase.semilogx(freqs_hz, phase_deg(true_matrix[:, i, j]), lw=2.0, label=f"true {label}")
        ax_phase.semilogx(freqs_hz, phase_deg(identified_matrix[:, i, j]), "--", lw=2.0, label=f"identified {label}")
        ax_phase.set_ylabel(f"{label} [deg]")
        ax_phase.grid(True, which="both", alpha=0.3)
        ax_phase.legend(loc="best")

    axes[-1, 0].set_xlabel("Frequency [Hz]")
    axes[-1, 1].set_xlabel("Frequency [Hz]")
    axes[0, 0].set_title("Magnitude")
    axes[0, 1].set_title("Phase")
    plt.show()


if __name__ == "__main__":
    freqs_hz = np.logspace(np.log10(FREQ_MIN_HZ), np.log10(FREQ_MAX_HZ), N_FREQS)
    params = AcousticParameters(freqs_hz, c0=C_AIR, rho0=RHO_AIR)
    geometry = build_geometry()
    post_processor = ThreeMicPostProcessor(params=params, geometry=geometry)

    omega = 2.0 * np.pi * freqs_hz
    z_load = np.full_like(omega, RIGID_IMPEDANCE, dtype=np.complex128)
    z_source = np.full_like(omega, params.z0 / geometry.s_tube, dtype=np.complex128)
    if SOURCE_IMPEDANCE_MODE == "same_as_load":
        z_source = z_load.copy()

    p1_a, p2_a, p3_a, p0_a_direct, v0_a_direct, pl_a_direct, vl_a_direct = simulate_three_mic_pressures_and_boundary_states(
        load_length=geometry.l_load_a,
        geometry=geometry,
        params=params,
        z_source=z_source,
        z_load=z_load,
    )
    p1_b, p2_b, p3_b, p0_b_direct, v0_b_direct, pl_b_direct, vl_b_direct = simulate_three_mic_pressures_and_boundary_states(
        load_length=geometry.l_load_b,
        geometry=geometry,
        params=params,
        z_source=z_source,
        z_load=z_load,
    )

    h12 = np.column_stack((p2_a / p1_a, p2_b / p1_b))
    h13 = np.column_stack((p3_a / p1_a, p3_b / p1_b))

    z_tube = np.full(freqs_hz.shape, params.z0, dtype=np.complex128)
    l2_by_load = np.full(2, geometry.l2, dtype=float)
    l3_by_load = np.array([geometry.l_load_a, geometry.l_load_b], dtype=float)

    p0_a, v0_a, pl_a, vl_a, p0_b, v0_b, pl_b, vl_b = post_processor.reconstruct_boundary_states_from_h(
        h12,
        h13,
        k_tube=params.wavenumbers,
        z_tube=z_tube,
        l1=geometry.l1,
        l2_by_load=l2_by_load,
        l3_by_load=l3_by_load,
        s_tube=geometry.s_tube,
        s_eff=geometry.s_tube,
    )

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

    true_slab = ElasticSlab(
        radius=geometry.r_slab,
        length=geometry.l_slab,
        rho=SLAB_RHO,
        young=SLAB_YOUNG,
        poisson=SLAB_POISSON,
        loss_factor=SLAB_LOSS_FACTOR,
    )
    true_slab_matrix_pu = true_slab.matrix(omega)
    true_slab_matrix_ident = slab_matrix_pu_to_identification_basis(true_slab_matrix_pu, geometry.s_slab)

    reconstructed_states = {
        "p0_a": p0_a,
        "v0_a": v0_a,
        "pl_a": pl_a,
        "vl_a": vl_a,
        "p0_b": p0_b,
        "v0_b": v0_b,
        "pl_b": pl_b,
        "vl_b": vl_b,
    }
    direct_states = {
        "p0_a": p0_a_direct,
        "v0_a": v0_a_direct,
        "pl_a": pl_a_direct,
        "vl_a": vl_a_direct,
        "p0_b": p0_b_direct,
        "v0_b": v0_b_direct,
        "pl_b": pl_b_direct,
        "vl_b": vl_b_direct,
    }

    rel_error = np.linalg.norm(identified_matrix_sh - true_slab_matrix_ident, axis=(1, 2)) / np.maximum(
        np.linalg.norm(true_slab_matrix_ident, axis=(1, 2)),
        np.finfo(float).tiny,
    )

    print("=== PURE TMM THREE-MIC / TWO-LOAD IDENTIFICATION ===")
    print(f"n_freqs                : {freqs_hz.size}")
    print(f"f_min                 : {freqs_hz.min():.3f} Hz")
    print(f"f_max                 : {freqs_hz.max():.3f} Hz")
    print(f"load A                : {geometry.l_load_a * 1e3:.3f} mm")
    print(f"load B                : {geometry.l_load_b * 1e3:.3f} mm")
    print(f"median relative error : {np.median(rel_error):.6e}")
    print(f"max relative error    : {np.max(rel_error):.6e}")
    for key in ("p0_a", "v0_a", "pl_a", "vl_a", "p0_b", "v0_b", "pl_b", "vl_b"):
        gap = level_db(reconstructed_states[key]) - level_db(direct_states[key])
        print(f"max |{key} gap| [dB]   : {np.max(np.abs(gap)):.6e}")

    plot_mic_pressures(freqs_hz, p1_a, p2_a, p3_a, p1_b, p2_b, p3_b)
    plot_boundary_state_comparison(freqs_hz, reconstructed_states, direct_states)
    plot_matrix_comparison(freqs_hz, identified_matrix_sh, true_slab_matrix_ident)
