"""Minimal WBS 2 physical decascade example with scaled internal L-curve regularization."""

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

from toolkitsd.acoustmm import CylindricalDuct, ElasticSlab, WaveguideElement


class FrozenMatrixElement(WaveguideElement):
    """Waveguide element backed by a precomputed transfer-matrix stack."""

    def __init__(self, matrices: np.ndarray) -> None:
        self.matrices = np.asarray(matrices, dtype=np.complex128)

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        omega = self._as_omega_array(omega)
        if self.matrices.shape[0] != omega.size:
            raise ValueError("matrix stack and omega must have the same length")
        return self.matrices


def add_relative_complex_noise(values: np.ndarray, rel_noise: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal(values.shape) + 1j * rng.standard_normal(values.shape)
    scale = np.maximum(np.abs(values), np.finfo(float).tiny)
    return values + rel_noise * scale * noise


def scale_transfer_matrix(matrices: np.ndarray, z_ref: float) -> np.ndarray:
    scale = np.array([[1.0, 0.0], [0.0, z_ref]], dtype=np.complex128)
    inv_scale = np.array([[1.0, 0.0], [0.0, 1.0 / z_ref]], dtype=np.complex128)
    return np.einsum("ab,nbc,cd->nad", scale, matrices, inv_scale)


def unscale_transfer_matrix(matrices: np.ndarray, z_ref: float) -> np.ndarray:
    scale = np.array([[1.0, 0.0], [0.0, z_ref]], dtype=np.complex128)
    inv_scale = np.array([[1.0, 0.0], [0.0, 1.0 / z_ref]], dtype=np.complex128)
    return np.einsum("ab,nbc,cd->nad", inv_scale, matrices, scale)


def relative_matrix_error(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    ref_norm = np.maximum(np.linalg.norm(reference, axis=(1, 2)), np.finfo(float).tiny)
    return np.linalg.norm(estimate - reference, axis=(1, 2)) / ref_norm


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def rigid_end_pressure(
    recovered_element: WaveguideElement,
    cavity: WaveguideElement,
    omega: np.ndarray,
    *,
    radius: float,
    rho_air: float,
    c_air: float,
    p0: float = 1.0,
) -> np.ndarray:
    z_rigid_like = np.full(omega.shape, 1.0e18 + 0.0j, dtype=np.complex128)
    z_source = rho_air * c_air / (np.pi * radius**2)
    system = recovered_element + cavity
    p_in = system.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)
    return system.p_tm(p_in, z_rigid_like, omega)


def lcurve_corner(residuals: np.ndarray, solutions: np.ndarray) -> int:
    x = np.log(np.maximum(residuals, np.finfo(float).tiny))
    y = np.log(np.maximum(solutions, np.finfo(float).tiny))
    if residuals.size < 3:
        return int(np.argmin(residuals))

    curvature = np.zeros(residuals.size, dtype=np.float64)
    for idx in range(1, residuals.size - 1):
        dx = (x[idx + 1] - x[idx - 1]) / 2.0
        dy = (y[idx + 1] - y[idx - 1]) / 2.0
        d2x = x[idx + 1] - 2.0 * x[idx] + x[idx - 1]
        d2y = y[idx + 1] - 2.0 * y[idx] + y[idx - 1]
        denom = (dx**2 + dy**2) ** 1.5 + 1e-30
        curvature[idx] = abs(dx * d2y - dy * d2x) / denom
    return int(np.nanargmax(curvature))


def lcurve_diagnostics(matrix: np.ndarray, lambda_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    column_scale = np.linalg.norm(matrix, axis=0)
    column_scale = np.where(column_scale > 1.0e-30, column_scale, 1.0)
    matrix_scaled = matrix / column_scale
    gram = matrix_scaled.conj().T @ matrix_scaled
    matrix_h = matrix_scaled.conj().T

    residual_norms = np.zeros(lambda_grid.size, dtype=np.float64)
    solution_norms = np.zeros(lambda_grid.size, dtype=np.float64)
    for lambda_idx, lambda_value in enumerate(lambda_grid):
        inverse_scaled = np.linalg.solve(gram + lambda_value * identity, matrix_h)
        inverse_matrix = inverse_scaled / column_scale[:, np.newaxis]
        residual_norms[lambda_idx] = np.linalg.norm(matrix @ inverse_matrix - identity)
        solution_norms[lambda_idx] = np.linalg.norm(inverse_matrix)

    best_idx = lcurve_corner(residual_norms, solution_norms)
    return residual_norms, solution_norms, best_idx


def is_degenerate_lcurve(residual_norms: np.ndarray, solution_norms: np.ndarray) -> bool:
    residual_span = np.max(residual_norms) / np.maximum(np.min(residual_norms), np.finfo(float).tiny)
    solution_span = np.max(solution_norms) / np.maximum(np.min(solution_norms), np.finfo(float).tiny)
    return residual_span < 10.0 or solution_span < 10.0


def plot_results(
    freqs_hz: np.ndarray,
    true_matrix: np.ndarray,
    direct_matrix: np.ndarray,
    tikhonov_matrix: np.ndarray,
    regularized_matrix: np.ndarray,
    air_condition_raw: np.ndarray,
    air_condition_scaled: np.ndarray,
    direct_error: np.ndarray,
    tikhonov_error: np.ndarray,
    regularized_error: np.ndarray,
    p_end_true: np.ndarray,
    p_end_direct: np.ndarray,
    p_end_tikhonov: np.ndarray,
    p_end_regularized: np.ndarray,
    residual_norms: np.ndarray,
    solution_norms: np.ndarray,
    lambda_grid: np.ndarray,
    selected_lambda_index: int,
    selected_frequency_hz: float,
    degenerate_lcurve: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].semilogx(freqs_hz, air_condition_raw, lw=2.0, label="cond(air), raw [p, U]")
    axes[0].semilogx(freqs_hz, air_condition_scaled, "--", lw=2.0, label="cond(air), scaled [p, Zref U]")
    axes[0].set_ylabel("Condition number")
    axes[0].set_title("Scaling removes unit-driven ill-conditioning of the air section")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, direct_error, lw=2.0, label="Direct decascade")
    axes[1].semilogx(freqs_hz, tikhonov_error, "--", lw=2.0, label="Scaled Tikhonov decascade")
    axes[1].semilogx(freqs_hz, regularized_error, lw=2.0, label="Scaled internal L-curve decascade")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Relative matrix error")
    axes[1].set_title("Direct vs Tikhonov vs L-curve")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].semilogx(freqs_hz, level_db(p_end_true), lw=2.0, label="True slab + cavity")
    axes[2].semilogx(freqs_hz, level_db(p_end_direct), "--", lw=1.8, label="Noisy direct slab + cavity")
    axes[2].semilogx(freqs_hz, level_db(p_end_tikhonov), "-.", lw=1.8, label="Scaled Tikhonov slab + cavity")
    axes[2].semilogx(freqs_hz, level_db(p_end_regularized), ":", lw=2.2, label="Scaled regularized slab + cavity")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel(r"$20 \log_{10}|p_{end}|$ [dB re 1 Pa]")
    axes[2].set_title("Rigid-end pressure at cavity end")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(loc="best")
    plt.show()

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    labels = [["A", "B"], ["C", "D"]]
    for i in range(2):
        for j in range(2):
            ax = axes2[i, j]
            label = labels[i][j]
            ax.semilogx(freqs_hz, np.abs(true_matrix[:, i, j]), lw=2.0, label=f"True {label}")
            ax.semilogx(freqs_hz, np.abs(direct_matrix[:, i, j]), "--", lw=1.8, label=f"Direct {label}")
            ax.semilogx(freqs_hz, np.abs(tikhonov_matrix[:, i, j]), "-.", lw=1.8, label=f"Tikhonov {label}")
            ax.semilogx(freqs_hz, np.abs(regularized_matrix[:, i, j]), ":", lw=2.2, label=f"L-curve {label}")
            ax.set_ylabel(f"|{label}|")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best")

    for ax in axes2[-1, :]:
        ax.set_xlabel("Frequency [Hz]")

    plt.show()

    fig3, ax3 = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax3.loglog(residual_norms, solution_norms, "-o", ms=4, lw=1.8, label="L-curve")
    ax3.loglog(
        residual_norms[selected_lambda_index],
        solution_norms[selected_lambda_index],
        "ro",
        ms=8,
        label=f"Selected lambda = {lambda_grid[selected_lambda_index]:.3e}",
    )
    ax3.set_xlabel(r"Residual norm $||M X - I||_2$")
    ax3.set_ylabel(r"Solution norm $||X||_2$")
    title = f"L-curve for removed air matrix at {selected_frequency_hz:.1f} Hz"
    if degenerate_lcurve:
        title += " (degenerate: scaled problem already well-conditioned)"
    ax3.set_title(title)
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(12)
    freqs_hz = np.logspace(np.log10(100.0), np.log10(10000.0), 400)
    omega = 2.0 * np.pi * freqs_hz
    lambda_grid = np.logspace(-12.0, 1, 121)
    radius = 3.75e-3
    rho_air = 1.2
    c_air = 343.0
    z_ref = rho_air * c_air / (np.pi * radius**2)

    slab = ElasticSlab(
        radius=radius,
        length=6.6e-3,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )

    air = CylindricalDuct(radius=radius, length=6.4e-3, c0=c_air, rho0=rho_air)
    cavity = CylindricalDuct(radius=radius, length=6.4e-3, c0=c_air, rho0=rho_air)
    true_matrix = slab.matrix(omega)
    air_matrix = air.matrix(omega)

    measured_total_matrix = add_relative_complex_noise((slab + air).matrix(omega), rel_noise=1.0e-3, rng=rng)
    measured_air_matrix = add_relative_complex_noise(air_matrix, rel_noise=1.0e-3, rng=rng)

    measured_total = FrozenMatrixElement(measured_total_matrix)
    measured_air = FrozenMatrixElement(measured_air_matrix)

    direct_matrix = measured_total.decascade_right(measured_air).matrix(omega)
    tikhonov_matrix = unscale_transfer_matrix(
        FrozenMatrixElement(scale_transfer_matrix(measured_total_matrix, z_ref)).decascade_right(
            FrozenMatrixElement(scale_transfer_matrix(measured_air_matrix, z_ref)),
            method="tikhonov",
            regularization=1.0e-8,
        ).matrix(omega),
        z_ref,
    )
    regularized_matrix = unscale_transfer_matrix(
        FrozenMatrixElement(scale_transfer_matrix(measured_total_matrix, z_ref)).decascade_right(
            FrozenMatrixElement(scale_transfer_matrix(measured_air_matrix, z_ref)),
            method="lcurve",
            lambda_grid=lambda_grid,
        ).matrix(omega),
        z_ref,
    )

    air_condition_raw = np.linalg.cond(measured_air_matrix)
    air_condition_scaled = np.linalg.cond(scale_transfer_matrix(measured_air_matrix, z_ref))
    direct_error = relative_matrix_error(true_matrix, direct_matrix)
    tikhonov_error = relative_matrix_error(true_matrix, tikhonov_matrix)
    regularized_error = relative_matrix_error(true_matrix, regularized_matrix)
    p_end_true = rigid_end_pressure(slab, cavity, omega, radius=radius, rho_air=rho_air, c_air=c_air)
    p_end_direct = rigid_end_pressure(
        FrozenMatrixElement(direct_matrix),
        cavity,
        omega,
        radius=radius,
        rho_air=rho_air,
        c_air=c_air,
    )
    p_end_tikhonov = rigid_end_pressure(
        FrozenMatrixElement(tikhonov_matrix),
        cavity,
        omega,
        radius=radius,
        rho_air=rho_air,
        c_air=c_air,
    )
    p_end_regularized = rigid_end_pressure(
        FrozenMatrixElement(regularized_matrix),
        cavity,
        omega,
        radius=radius,
        rho_air=rho_air,
        c_air=c_air,
    )

    improvement = direct_error / np.maximum(regularized_error, np.finfo(float).tiny)
    best_idx = int(np.argmax(improvement))
    selected_freq_idx = int(np.argmax(regularized_error))
    residual_norms, solution_norms, selected_lambda_index = lcurve_diagnostics(
        scale_transfer_matrix(measured_air_matrix, z_ref)[selected_freq_idx],
        lambda_grid,
    )
    degenerate_lcurve = is_degenerate_lcurve(residual_norms, solution_norms)

    print("=== WBS 2 PHYSICAL SLAB + AIR INVERSION WITH SCALED INTERNAL L-CURVE ===")
    print("Recovered element             : ElasticSlab")
    print("Removed element               : CylindricalDuct (6.4 mm)")
    print("Measured total                : ElasticSlab + CylindricalDuct")
    print("Noise level on matrices       : 1.0e-3")
    print(f"Z_ref for scaling             : {z_ref:.3e} Pa.s/m^3")
    print(f"Air cond range raw            : {np.min(air_condition_raw):.3e} to {np.max(air_condition_raw):.3e}")
    print(f"Air cond range scaled         : {np.min(air_condition_scaled):.3e} to {np.max(air_condition_scaled):.3e}")
    print(f"Median relative error direct  : {np.median(direct_error):.3e}")
    print(f"Median relative error tikho   : {np.median(tikhonov_error):.3e}")
    print(f"Median relative error lcurve  : {np.median(regularized_error):.3e}")
    print(f"L-curve diagnostic frequency  : {freqs_hz[selected_freq_idx]:.1f} Hz")
    print(f"L-curve selected lambda       : {lambda_grid[selected_lambda_index]:.3e}")
    if degenerate_lcurve:
        print("L-curve diagnosis             : degenerate after scaling; the removed air matrix is already well-conditioned")
        print("Recommendation                : prefer direct inversion on the scaled problem for this case")
    print(f"Best improvement frequency    : {freqs_hz[best_idx]:.1f} Hz")
    print(f"Improvement at that frequency : {improvement[best_idx]:.3e}x")

    plot_results(
        freqs_hz=freqs_hz,
        true_matrix=true_matrix,
        direct_matrix=direct_matrix,
        tikhonov_matrix=tikhonov_matrix,
        regularized_matrix=regularized_matrix,
        air_condition_raw=air_condition_raw,
        air_condition_scaled=air_condition_scaled,
        direct_error=direct_error,
        tikhonov_error=tikhonov_error,
        regularized_error=regularized_error,
        p_end_true=p_end_true,
        p_end_direct=p_end_direct,
        p_end_tikhonov=p_end_tikhonov,
        p_end_regularized=p_end_regularized,
        residual_norms=residual_norms,
        solution_norms=solution_norms,
        lambda_grid=lambda_grid,
        selected_lambda_index=selected_lambda_index,
        selected_frequency_hz=freqs_hz[selected_freq_idx],
        degenerate_lcurve=degenerate_lcurve,
    )
