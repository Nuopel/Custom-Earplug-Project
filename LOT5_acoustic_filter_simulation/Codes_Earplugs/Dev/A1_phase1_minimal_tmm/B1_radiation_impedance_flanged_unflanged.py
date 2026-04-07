"""Radiation impedance comparison: flanged reference vs unflanged approximation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, RadiationImpedance


if __name__ == "__main__":
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 600)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    radius = 0.02
    area = np.pi * radius**2
    zc = params.rho0 * params.c0 / area

    z_flanged = RadiationImpedance(radius=radius, mode="flanged", c0=params.c0, rho0=params.rho0).Z(omega)
    z_unflanged = RadiationImpedance(radius=radius, mode="unflanged_v2", c0=params.c0, rho0=params.rho0).Z(omega)

    zf_n = z_flanged / zc
    zu_n = z_unflanged / zc

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.semilogx(freqs, np.abs(np.real(zf_n)), color="black", linewidth=2.0, label="|Re{Z_rad/Zc}| flanged (reference)")
    ax.semilogx(freqs, np.abs(np.real(zu_n)), "--", color="tab:blue", linewidth=1.8, label="|Re{Z_rad/Zc}| unflanged (approx)")
    ax.semilogx(freqs, np.abs(np.imag(zf_n)), color="tab:red", linewidth=2.0, label="|Im{Z_rad/Zc}| flanged (reference)")
    ax.semilogx(freqs, np.abs(np.imag(zu_n)), "--", color="tab:orange", linewidth=1.8, label="|Im{Z_rad/Zc}| unflanged (approx)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 10.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Normalized Impedance Magnitude Components ($|Z_{rad}/Z_c|$)")
    ax.set_title("Radiation Impedance: Exact/Reference vs Approximation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()
