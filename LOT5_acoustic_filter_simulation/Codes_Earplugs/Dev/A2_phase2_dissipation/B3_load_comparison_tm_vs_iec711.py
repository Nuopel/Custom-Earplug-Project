"""Compare eardrum vs IEC711 load effects on Zin and insertion loss."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters
from toolkitsd.acoustmm import (
    CylindricalDuct,
    EardrumImpedance,
    IEC711Coupler,
    ImpedanceJunction,
    ViscothermalDuct,
)


if __name__ == "__main__":
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    r_canal = 4.0e-3
    r_bore = 1.2e-3
    L_canal = 20e-3
    L_bore = 12e-3
    S_canal = np.pi * r_canal**2
    S_bore = np.pi * r_bore**2

    open_system = CylindricalDuct(radius=r_canal, length=L_canal, c0=params.c0, rho0=params.rho0)
    occ_system = (
        CylindricalDuct(radius=r_canal, length=L_canal / 2.0, c0=params.c0, rho0=params.rho0)
        + ImpedanceJunction(S1=S_canal, S2=S_bore, end_correction=True, rho0=params.rho0)
        + ViscothermalDuct(radius=r_bore, length=L_bore, c0=params.c0, rho0=params.rho0)
        + ImpedanceJunction(S1=S_bore, S2=S_canal, end_correction=True, rho0=params.rho0)
        + CylindricalDuct(radius=r_canal, length=L_canal / 2.0, c0=params.c0, rho0=params.rho0)
    )

    z_tm = EardrumImpedance().Z(omega)
    z_711 = IEC711Coupler(c0=params.c0, rho0=params.rho0).Z(omega)

    zin_open_tm = open_system.Z_in(z_tm, omega)
    zin_occ_tm = occ_system.Z_in(z_tm, omega)
    zin_open_711 = open_system.Z_in(z_711, omega)
    zin_occ_711 = occ_system.Z_in(z_711, omega)

    il_tm = 20.0 * np.log10(np.abs(zin_open_tm / zin_occ_tm))
    il_711 = 20.0 * np.log10(np.abs(zin_open_711 / zin_occ_711))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.semilogx(freqs, np.abs(zin_occ_tm), color="tab:red", linewidth=2.0, label=r"|Zin| occ, Eardrum load")
    ax1.semilogx(freqs, np.abs(zin_occ_711), color="tab:blue", linewidth=2.0, label=r"|Zin| occ, IEC711 load")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax1.set_title("Termination Effect: Eardrum vs IEC711")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.semilogx(freqs, il_tm, color="tab:red", linewidth=2.0, label="IL with Eardrum load")
    ax2.semilogx(freqs, il_711, color="tab:blue", linewidth=2.0, label="IL with IEC711 load")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("IL (dB)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    print(f"Load review: mean IL (TM)    = {np.mean(il_tm):.3f} dB")
    print(f"Load review: mean IL (IEC711)= {np.mean(il_711):.3f} dB")
    print(f"Load review: mean ΔIL        = {np.mean(il_711 - il_tm):.3f} dB")

    plt.tight_layout()
    plt.show()
