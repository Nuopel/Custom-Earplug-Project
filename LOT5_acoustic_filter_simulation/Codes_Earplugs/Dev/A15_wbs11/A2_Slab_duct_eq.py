"""Equivalent homogeneous-duct retrieval for a silicone slab.

Workflow:
1. Build a duct + slab + duct system in the same transfer-matrix framework.
2. Decascade the inlet/outlet ducts to isolate the slab 2-port.
3. Extract effective section parameters ``k_eq`` and ``Zc_eq`` directly from the
   slab transfer matrix, assuming a homogeneous 1D section of known thickness.
4. Reconstruct the equivalent slab matrix and compare it to the original slab.

The retrieved ``k_eq`` and ``Zc_eq`` are effective section parameters for the
confined slab, not intrinsic bulk silicone constants.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ACOUSTMM_ROOT = HERE.parents[2]
REFACTOR_ROOT = HERE.parents[3]
candidate_paths = [ACOUSTMM_ROOT / "src"]
candidate_paths.extend(sorted(REFACTOR_ROOT.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, ElasticSlab, FrozenMatrixElement


def continuity_tracked_wavenumber(raw_k: np.ndarray, length: float) -> np.ndarray:
    """Track a physical branch of arccos(trace/2)/L through frequency."""
    tracked = np.empty_like(raw_k)
    tracked[0] = raw_k[0]
    two_pi_over_l = 2.0 * np.pi / length

    for idx in range(1, raw_k.size):
        candidates: list[complex] = []
        for sign in (1.0, -1.0):
            base = sign * raw_k[idx]
            for shift in (-1, 0, 1):
                candidates.append(base + shift * two_pi_over_l)
        tracked[idx] = min(candidates, key=lambda cand: abs(cand - tracked[idx - 1]))

    return tracked


def align_wavenumber_branch(raw_k: np.ndarray, reference_k: np.ndarray, length: float) -> np.ndarray:
    """Align a retrieved k branch to a reference branch using sign and 2*pi/L shifts."""
    aligned = np.empty_like(raw_k)
    two_pi_over_l = 2.0 * np.pi / length

    for idx in range(raw_k.size):
        ref = reference_k[idx]
        candidates: list[complex] = []
        for sign in (1.0, -1.0):
            base = sign * raw_k[idx]
            for shift in (-1, 0, 1):
                candidates.append(base + shift * two_pi_over_l)
        aligned[idx] = min(candidates, key=lambda cand: abs(cand - ref))
    return aligned


def continuity_tracked_impedance(b: np.ndarray, c: np.ndarray, sin_k_l: np.ndarray) -> np.ndarray:
    """Recover a continuous Zc branch from B/C, selecting the sign by reconstruction error."""
    z_raw = np.lib.scimath.sqrt(b / c)
    z_tracked = np.empty_like(z_raw)

    for idx, z_base in enumerate(z_raw):
        candidates = [z_base, -z_base]
        best = None
        best_score = None
        for candidate in candidates:
            b_rec = 1j * candidate * sin_k_l[idx]
            c_rec = 1j * sin_k_l[idx] / candidate
            score = abs(b[idx] - b_rec) + abs(c[idx] - c_rec)
            if idx > 0:
                score += 0.1 * abs(candidate - z_tracked[idx - 1])
            if best_score is None or score < best_score:
                best = candidate
                best_score = score
        z_tracked[idx] = best

    head = z_tracked[: min(10, z_tracked.size)]
    if np.mean(np.real(head)) < 0.0:
        z_tracked = -z_tracked
    return z_tracked


def extract_equivalent_section_from_matrix(matrix_slab: np.ndarray, omega: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract k_eq and Zc_eq from a slab transfer matrix in [p, U] basis."""
    a = matrix_slab[:, 0, 0]
    b = matrix_slab[:, 0, 1]
    c = matrix_slab[:, 1, 0]
    d = matrix_slab[:, 1, 1]

    trace_half = 0.5 * (a + d)
    k_raw = np.arccos(trace_half + 0j) / length
    k_eq = continuity_tracked_wavenumber(k_raw, length)

    sin_k_l = np.sin(k_eq * length)
    z_eq = continuity_tracked_impedance(b, c, sin_k_l)
    return k_eq, z_eq


def homogeneous_matrix_from_k_zc(k_eq: np.ndarray, zc_eq: np.ndarray, length: float) -> np.ndarray:
    """Rebuild the transfer matrix of the equivalent homogeneous section."""
    k_l = k_eq * length
    ck_l = np.cos(k_l)
    sk_l = np.sin(k_l)

    matrix = np.zeros((k_eq.size, 2, 2), dtype=np.complex128)
    matrix[:, 0, 0] = ck_l
    matrix[:, 0, 1] = 1j * zc_eq * sk_l
    matrix[:, 1, 0] = 1j * sk_l / zc_eq
    matrix[:, 1, 1] = ck_l
    return matrix


def relative_matrix_error(matrix_ref: np.ndarray, matrix_test: np.ndarray) -> np.ndarray:
    return np.linalg.norm(matrix_ref - matrix_test, axis=(1, 2)) / np.maximum(
        np.linalg.norm(matrix_ref, axis=(1, 2)),
        np.finfo(float).tiny,
    )


def plot_effective_params(
    freqs: np.ndarray,
    k_ref: np.ndarray,
    z_ref: np.ndarray,
    k_eq: np.ndarray,
    z_eq: np.ndarray,
    k_eff_rt: np.ndarray | None = None,
    z_eff_rt: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, constrained_layout=True)

    axes[0, 0].semilogx(freqs, np.abs(k_ref), linewidth=2.0, label=r"Original slab")
    axes[0, 0].semilogx(freqs, np.abs(k_eq), "--", linewidth=2.0, label=r"Direct matrix retrieval")
    if k_eff_rt is not None:
        axes[0, 0].semilogx(freqs, np.abs(k_eff_rt), ":", linewidth=2.0, label=r"R/T retrieval aligned")
    axes[0, 0].set_ylabel(r"$|k|$ [rad/m]")
    axes[0, 0].set_title("Original slab vs equivalent section parameters")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend(loc="best")

    axes[0, 1].semilogx(freqs, np.angle(k_ref), linewidth=2.0, label=r"Original slab")
    axes[0, 1].semilogx(freqs, np.angle(k_eq), "--", linewidth=2.0, label=r"Direct matrix retrieval")
    if k_eff_rt is not None:
        axes[0, 1].semilogx(freqs, np.angle(k_eff_rt), ":", linewidth=2.0, label=r"R/T retrieval aligned")
    axes[0, 1].set_ylabel(r"Phase($k$) [rad]")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].set_ylim([-np.pi, np.pi])

    axes[1, 0].semilogx(freqs, np.abs(z_ref), linewidth=2.0, label=r"Original slab")
    axes[1, 0].semilogx(freqs, np.abs(z_eq), "--", linewidth=2.0, label=r"Direct matrix retrieval")
    if z_eff_rt is not None:
        axes[1, 0].semilogx(freqs, np.abs(z_eff_rt), ":", linewidth=2.0, label=r"R/T retrieval")
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel(r"$|Z_c|$ [Pa·s/m$^3$]")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    axes[1, 1].semilogx(freqs, np.angle(z_ref), linewidth=2.0, label=r"Original slab")
    axes[1, 1].semilogx(freqs, np.angle(z_eq), "--", linewidth=2.0, label=r"Direct matrix retrieval")
    if z_eff_rt is not None:
        axes[1, 1].semilogx(freqs, np.angle(z_eff_rt), ":", linewidth=2.0, label=r"R/T retrieval")
    axes[1, 1].set_xlabel("Frequency [Hz]")
    axes[1, 1].set_ylabel(r"Phase($Z_c$) [rad]")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].set_ylim([-np.pi, np.pi])


def plot_matrix_comparison(freqs: np.ndarray, matrix_ref: np.ndarray, matrix_rec: np.ndarray) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True, constrained_layout=True)
    labels = (("A", "B"), ("C", "D"))

    for row in range(2):
        for col in range(2):
            idx_top = 2 * row
            idx_bottom = 2 * row + 1
            label = labels[row][col]
            ref = matrix_ref[:, row, col]
            rec = matrix_rec[:, row, col]

            axes[idx_top, col].semilogx(freqs, np.abs(ref), linewidth=2.0, label="Original slab")
            axes[idx_top, col].semilogx(freqs, np.abs(rec), "--", linewidth=2.0, label="Reconstructed eq slab")
            axes[idx_bottom, col].semilogx(freqs, np.angle(ref), linewidth=2.0, label="Original slab")
            axes[idx_bottom, col].semilogx(freqs, np.angle(rec), "--", linewidth=2.0, label="Reconstructed eq slab")
            axes[idx_top, col].set_ylabel(f"|{label}|")
            axes[idx_bottom, col].set_ylabel(f"Phase({label}) [rad]")
            axes[idx_top, col].grid(True, which="both", alpha=0.3)
            axes[idx_bottom, col].grid(True, which="both", alpha=0.3)
            axes[idx_top, col].legend(loc="best")
            axes[idx_bottom, col].set_ylim([-np.pi, np.pi])

    axes[3, 0].set_xlabel("Frequency [Hz]")
    axes[3, 1].set_xlabel("Frequency [Hz]")
    fig.suptitle("Original slab matrix vs reconstructed equivalent slab matrix")


def plot_reconstruction_error(freqs: np.ndarray, matrix_ref: np.ndarray, matrix_rec: np.ndarray) -> None:
    rel_error = relative_matrix_error(matrix_ref, matrix_rec)

    plt.figure(figsize=(10, 5))
    plt.semilogx(freqs, rel_error, linewidth=2.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Relative matrix error [-]")
    plt.title("Equivalent slab self-consistency check")
    plt.grid(True, which="both", alpha=0.3)


if __name__ == "__main__":
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    r_slab = 3.5e-3
    r_cav = r_slab
    l_slab = 6.6e-3
    l_cav =  6.4e-3
    l_inlet = 5.0e-3
    l_outlet = 5.0e-3

    slab = ElasticSlab(
        radius=r_slab,
        length=l_slab,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )

    inlet = CylindricalDuct(radius=r_slab, length=l_inlet, c0=params.c0, rho0=params.rho0)
    outlet = CylindricalDuct(radius=r_slab, length=l_outlet, c0=params.c0, rho0=params.rho0)
    cav = CylindricalDuct(radius=r_cav, length=l_cav, c0=params.c0, rho0=params.rho0)

    system = inlet + slab + cav + outlet
    slab_element = system.decascade_right(outlet).decascade_right(cav).decascade_left(inlet)
    matrix_slab = slab_element.matrix(omega)

    k_eq, zc_eq = extract_equivalent_section_from_matrix(matrix_slab, omega, l_slab)
    matrix_eq = homogeneous_matrix_from_k_zc(k_eq, zc_eq, l_slab)
    eq_slab = FrozenMatrixElement.from_pu(matrix_eq)

    z_ref_medium = np.full(freqs.shape, params.rho0 * params.c0 / slab.area, dtype=np.complex128)
    k_ref_medium = omega / params.c0

    reflection, transmission, _ = slab_element.reflection_transmission_absorption(
        Z_c=z_ref_medium,
        omega=omega,
        k_ref=k_ref_medium,
        length=l_slab,
    )
    retrieval = slab_element.retrieve_equivalent_duct(
        Z_c=z_ref_medium,
        omega=omega,
        k_ref=k_ref_medium,
        length=l_slab,
        area=slab.area,
        reflection=reflection,
        transmission=transmission,
    )
    k_eff_rt = align_wavenumber_branch(retrieval.k_eff, k_eq, l_slab)
    z_eff_rt = retrieval.Z_eff

    k_original = omega / slab.longitudinal_speed
    zc_original = np.full(freqs.shape, slab.longitudinal_acoustic_impedance, dtype=np.complex128)

    rel_error = relative_matrix_error(matrix_slab, matrix_eq)
    print(f"Mean relative reconstruction error = {np.mean(rel_error):.3e}")
    print(f"Max  relative reconstruction error = {np.max(rel_error):.3e}")
    print(f"Mean |k_eq - k_original| = {np.mean(np.abs(k_eq - k_original)):.3e} rad/m")
    print(f"Mean |Zc_eq - Zc_original| = {np.mean(np.abs(zc_eq - zc_original)):.3e} Pa·s/m^3")
    print(f"Mean |k_eq - k_eff_rt| = {np.mean(np.abs(k_eq - k_eff_rt)):.3e} rad/m")
    print(f"Mean |Zc_eq - Z_eff_rt| = {np.mean(np.abs(zc_eq - z_eff_rt)):.3e} Pa·s/m^3")

    plot_effective_params(freqs, k_original, zc_original, k_eq, zc_eq, k_eff_rt, z_eff_rt)
    plt.show()
    plot_matrix_comparison(freqs, matrix_slab, eq_slab.matrix(omega))
    plt.show()
    plot_reconstruction_error(freqs, matrix_slab, eq_slab.matrix(omega))
    plt.show()

    abs(zc_original)
    abs(zc_eq)
    abs(z_eff_rt)
