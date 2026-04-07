from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv, jv

from toolkitsd.acoustmm import ExactFlexuralPlateSeriesImpedance, LowFrequencyFlexuralPlateSeriesImpedance


def plot_impedance_comparison(
    freqs: np.ndarray,
    z_exact: np.ndarray,
    z_lowfreq_d2: np.ndarray,
    z_alt_1_over_96: np.ndarray,
    z_alt_3_over_320: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(z_exact), np.finfo(float).tiny)), linewidth=2.2, label="Exact D1")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(z_lowfreq_d2), np.finfo(float).tiny)), "--", linewidth=2.0, label="Low-frequency D2")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(z_alt_1_over_96), np.finfo(float).tiny)), ":", linewidth=1.8, label="Alt LF 1/96")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(z_alt_3_over_320), np.finfo(float).tiny)), "-.", linewidth=1.8, label="Alt LF 3/320")
    axes[0].set_ylabel("|Z| [dB]")
    axes[0].set_title("Flexural plate impedance magnitude")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs, np.angle(z_exact), linewidth=2.2, label="Exact D1")
    axes[1].semilogx(freqs, np.angle(z_lowfreq_d2), "--", linewidth=2.0, label="Low-frequency D2")
    axes[1].semilogx(freqs, np.angle(z_alt_1_over_96), ":", linewidth=1.8, label="Alt LF 1/96")
    axes[1].semilogx(freqs, np.angle(z_alt_3_over_320), "-.", linewidth=1.8, label="Alt LF 3/320")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Phase(Z) [rad]")
    axes[1].set_ylim([-np.pi, np.pi])
    axes[1].set_title("Flexural plate impedance phase")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


def plot_impedance_real_imag(
    freqs: np.ndarray,
    z_exact: np.ndarray,
    z_lowfreq_d2: np.ndarray,
    z_alt_1_over_96: np.ndarray,
    z_alt_3_over_320: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.real(z_exact)), np.finfo(float).tiny)), linewidth=2.2, label="Exact D1")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.real(z_lowfreq_d2)), np.finfo(float).tiny)), "--", linewidth=2.0, label="Low-frequency D2")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.real(z_alt_1_over_96)), np.finfo(float).tiny)), ":", linewidth=1.8, label="Alt LF 1/96")
    axes[0].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.real(z_alt_3_over_320)), np.finfo(float).tiny)), "-.", linewidth=1.8, label="Alt LF 3/320")
    axes[0].set_ylabel("|Re(Z)| [dB]")
    axes[0].set_title("Flexural plate impedance real part")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.imag(z_exact)), np.finfo(float).tiny)), linewidth=2.2, label="Exact D1")
    axes[1].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.imag(z_lowfreq_d2)), np.finfo(float).tiny)), "--", linewidth=2.0, label="Low-frequency D2")
    axes[1].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.imag(z_alt_1_over_96)), np.finfo(float).tiny)), ":", linewidth=1.8, label="Alt LF 1/96")
    axes[1].semilogx(freqs, 20.0 * np.log10(np.maximum(np.abs(np.imag(z_alt_3_over_320)), np.finfo(float).tiny)), "-.", linewidth=1.8, label="Alt LF 3/320")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|Im(Z)| [dB]")
    axes[1].set_title("Flexural plate impedance imaginary part")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


if __name__ == "__main__":
    rho = 2330.0
    h = 1e-4
    a = 5e-3
    E = 170e9
    nu = 0.28

    bending_stiffness = E * h**3 / (12.0 * (1.0 - nu**2))
    rho_s = rho * h
    area = np.pi * a**2
    plate_mass = rho_s * area

    freqs = np.logspace(1, 5, 500)
    omega = 2.0 * np.pi * freqs

    exact_element = ExactFlexuralPlateSeriesImpedance(
        radius=a,
        rho_plate=rho,
        h=h,
        E=E,
        nu=nu,
    )
    lowfreq_element = LowFrequencyFlexuralPlateSeriesImpedance(
        radius=a,
        rho_plate=rho,
        h=h,
        E=E,
        nu=nu,
    )

    def z_full(omega_array: np.ndarray) -> np.ndarray:
        kb = (rho_s * omega_array**2 / bending_stiffness) ** 0.25
        x = kb * a
        num = iv(1, x) * jv(0, x) + jv(1, x) * iv(0, x)
        den = iv(1, x) * jv(2, x) - jv(1, x) * iv(2, x)
        return -1j * omega_array * plate_mass / area**2 * (num / den)

    def z_alt_1_over_96(omega_array: np.ndarray) -> np.ndarray:
        kb = (rho_s * omega_array**2 / bending_stiffness) ** 0.25
        x = kb * a
        return -1j * omega_array * rho_s * (192.0 / area) * (1.0 / x**4 - 1.0 / 96.0)

    def z_alt_3_over_320(omega_array: np.ndarray) -> np.ndarray:
        kb = (rho_s * omega_array**2 / bending_stiffness) ** 0.25
        x = kb * a
        return -1j * omega_array * rho_s * (192.0 / area) * (1.0 / x**4 - 3.0 / 320.0)

    z_exact = exact_element.acoustic_series_impedance(omega)
    z_lowfreq_d2 = lowfreq_element.acoustic_series_impedance(omega)
    z_alt_96 = z_alt_1_over_96(omega)
    z_alt_320 = z_alt_3_over_320(omega)

    kb_arr = (rho_s * omega**2 / bending_stiffness) ** 0.25
    x_arr = kb_arr * a
    err_d2 = np.abs((np.imag(z_lowfreq_d2) - np.imag(z_exact)) / np.maximum(np.abs(np.imag(z_exact)), 1e-30)) * 100.0
    err_96 = np.abs((np.imag(z_alt_96) - np.imag(z_exact)) / np.maximum(np.abs(np.imag(z_exact)), 1e-30)) * 100.0
    err_320 = np.abs((np.imag(z_alt_320) - np.imag(z_exact)) / np.maximum(np.abs(np.imag(z_exact)), 1e-30)) * 100.0

    print(f"{'f (Hz)':>10}  {'x = kb*a':>10}  {'Z_exact (Im)':>14}  {'err_D2 %':>10}  {'err_1/96 %':>12}  {'err_3/320 %':>12}")
    for idx in np.searchsorted(freqs, [50, 100, 500, 1000, 5000, 10000]):
        if idx < len(freqs):
            print(
                f"{freqs[idx]:>10.1f}  {x_arr[idx]:>10.4f}  {np.imag(z_exact[idx]):>14.4e}  "
                f"{err_d2[idx]:>10.6f}  {err_96[idx]:>12.6f}  {err_320[idx]:>12.6f}"
            )

    f1_approx = (3.196**2 / (2.0 * np.pi * a**2)) * np.sqrt(bending_stiffness / rho_s)
    print(f"\nFirst resonance approximation: f1 ≈ {f1_approx:.1f} Hz")

    plot_impedance_comparison(freqs, z_exact, z_lowfreq_d2, z_alt_96, z_alt_320)
    plot_impedance_real_imag(freqs, z_exact, z_lowfreq_d2, z_alt_96, z_alt_320)
