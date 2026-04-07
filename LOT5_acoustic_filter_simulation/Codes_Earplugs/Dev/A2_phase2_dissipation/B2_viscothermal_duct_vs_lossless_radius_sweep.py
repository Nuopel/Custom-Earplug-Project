"""Phase 2 example: viscothermal vs lossless duct with radius sweep."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, RigidWall, ViscothermalDuct


if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    length = 0.3
    radii = [ 1e-2, 1e-3,1e-4]  # 1 mm, 1 cm, 10 cm
    colors = ["tab:red", "tab:blue", "tab:green"]

    fig, (ax_tl, ax_zin) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    z_rigid = RigidWall().Z(omega)

    for r, color in zip(radii, colors):
        lossless = CylindricalDuct(radius=r, length=length, c0=params.c0, rho0=params.rho0)
        viscothermal = ViscothermalDuct(radius=r, length=length, c0=params.c0, rho0=params.rho0)

        zc_ls = params.rho0 * params.c0 / (np.pi * r**2)
        tl_ls = lossless.TL(Z_c=zc_ls, omega=omega)

        _, zc_vt = viscothermal._gamma_zc(omega)
        tl_vt = viscothermal.TL(Z_c=zc_ls, omega=omega)

        zin_ls = lossless.Z_in(Z_load=z_rigid, omega=omega)
        zin_vt = viscothermal.Z_in(Z_load=z_rigid, omega=omega)

        label_base = f"r={r*1e3:.0f} mm"
        ax_tl.semilogx(freqs, tl_ls, "--", color=color, linewidth=1.3, alpha=0.85, label=f"TL lossless ({label_base})")
        ax_tl.semilogx(freqs, tl_vt, "-", color=color, linewidth=2.0, label=f"TL viscothermal ({label_base})")

        ax_zin.semilogx(freqs, np.abs(zin_ls), "--", color=color, linewidth=1.3, alpha=0.85, label=f"|Zin| lossless ({label_base})")
        ax_zin.semilogx(freqs, np.abs(zin_vt), "-", color=color, linewidth=2.0, label=f"|Zin| viscothermal ({label_base})")

    ax_tl.set_ylabel("TL (dB)")
    ax_tl.set_title("Duct Losses vs Radius: Lossless vs Viscothermal")
    ax_tl.grid(True, which="both", alpha=0.3)
    ax_tl.legend(loc="best", fontsize=8, ncol=2)

    ax_zin.set_xlabel("Frequency (Hz)")
    ax_zin.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax_zin.set_yscale("log")
    ax_zin.grid(True, which="both", alpha=0.3)
    ax_zin.legend(loc="best", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.show()
