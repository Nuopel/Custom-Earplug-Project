"""Compare lossless, viscothermal, and BLI ducts for r = 0.5 mm."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, BLIDuct, CylindricalDuct, RigidWall, ViscothermalDuct


if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    radius = 1e-3
    length = 30e-3

    lossless = CylindricalDuct(radius=radius, length=length, c0=params.c0, rho0=params.rho0)
    viscothermal = ViscothermalDuct(radius=radius, length=length, c0=params.c0, rho0=params.rho0)
    bli = BLIDuct(radius=radius, length=length, c0=params.c0, rho0=params.rho0)

    zc_ls = params.rho0 * params.c0 / (np.pi * radius**2)

    tl_ls = lossless.TL(Z_c=zc_ls, omega=omega)
    tl_vt = viscothermal.TL(Z_c=zc_ls, omega=omega)
    tl_bli = bli.TL(Z_c=zc_ls, omega=omega)

    z_rigid = RigidWall().Z(omega)
    zin_ls = lossless.Z_in(z_rigid, omega)
    zin_vt = viscothermal.Z_in(z_rigid, omega)
    zin_bli = bli.Z_in(z_rigid, omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.semilogx(freqs, tl_ls, "--", color="gray", linewidth=1.5, label="Lossless (reference)")
    ax1.semilogx(freqs, tl_vt, color="tab:red", linewidth=2.0, label="Viscothermal")
    ax1.semilogx(freqs, tl_bli, color="tab:blue", linewidth=2.0, label="BLI approximation")
    ax1.set_ylabel("TL (dB)")
    ax1.set_title("Loss Models Comparison for Uniform Duct (r = 0.5 mm)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.semilogx(freqs, np.abs(zin_ls), "--", color="gray", linewidth=1.5, label=r"|Zin| lossless")
    ax2.semilogx(freqs, np.abs(zin_vt), color="tab:red", linewidth=2.0, label=r"|Zin| viscothermal")
    ax2.semilogx(freqs, np.abs(zin_bli), color="tab:blue", linewidth=2.0, label=r"|Zin| BLI")
    ax2.set_yscale("log")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()
