"""PlateSeriesImpedance example: series impedance and impact on Zin/TL."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, MatchedLoad, RigidWall, PlateSeriesImpedance


if __name__ == "__main__":
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    radius = 40.0e-3
    area = np.pi * radius**2
    zc = params.rho0 * params.c0 / area

    # Example lossless silicone-like plate parameters.
    plate = PlateSeriesImpedance(
        area=area,
        rho_plate=1500.0,
        E=1.7e6,
        nu=0.48,
        h=0.03,
        c0=params.c0,
    )
    duct = CylindricalDuct(radius=radius, length=20e-3, c0=params.c0, rho0=params.rho0)
    system_with_plate = duct + plate + duct
    system_ref = duct + duct

    z_plate = plate.acoustic_series_impedance(omega)

    zin_ref = system_ref.Z_in(RigidWall().Z(omega), omega)
    zin_plate = system_with_plate.Z_in(RigidWall().Z(omega), omega)

    tl_ref = system_ref.TL(Z_c=zc, omega=omega)
    tl_plate = system_with_plate.TL(Z_c=zc, omega=omega)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax1.semilogx(freqs, np.real(z_plate), color="tab:blue", linewidth=2.0, label="Re{Z_plate}")
    ax1.semilogx(freqs, np.imag(z_plate), color="tab:orange", linewidth=2.0, label="Im{Z_plate}")
    ax1.set_ylabel(r"$Z_{plate}$ (Pa·s/m$^3$)")
    ax1.set_title("PlateSeriesImpedance")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.semilogx(freqs, np.abs(zin_ref), "--", color="gray", linewidth=1.6, label=r"|Zin| reference")
    ax2.semilogx(freqs, np.abs(zin_plate), color="tab:red", linewidth=2.0, label=r"|Zin| with plate")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    ax3.semilogx(freqs, tl_ref, "--", color="gray", linewidth=1.6, label="TL reference")
    ax3.semilogx(freqs, tl_plate, color="tab:green", linewidth=2.0, label="TL with plate")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("TL (dB)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best")

    print(f"Plate review: mean ΔTL = {np.mean(tl_plate - tl_ref):.3f} dB")
    print(f"Plate review: max  ΔTL = {np.max(np.abs(tl_plate - tl_ref)):.3f} dB")

    plt.tight_layout()
    plt.show()
