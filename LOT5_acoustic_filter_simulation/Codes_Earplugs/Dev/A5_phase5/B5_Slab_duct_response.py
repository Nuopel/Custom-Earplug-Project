"""Silicone slab example: impact on Zin/TL in a simple duct chain."""

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

from toolkitsd.acoustmm import AcousticParameters, ViscothermalDuct, CylindricalDuct, ElasticSlab, RigidWall


if __name__ == "__main__":
    freqs = np.logspace(np.log10(80.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    radius = 0.0035
    area = np.pi * radius**2
    zc = params.rho0 * params.c0 / area

    # Example silicone slab parameters.
    slab = ElasticSlab(
        radius=radius,
        length=6.6e-3,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )
    duct = ViscothermalDuct(radius=radius, length=20e-3, c0=params.c0, rho0=params.rho0)
    system_with_slab = duct + slab
    system_ref = duct

    z_slab = np.full_like(freqs, slab.longitudinal_acoustic_impedance, dtype=np.complex128)
    k_slab = omega / slab.longitudinal_speed

    zin_ref = system_ref.Z_in(RigidWall().Z(omega), omega)
    zin_slab = system_with_slab.Z_in(RigidWall().Z(omega), omega)

    tl_ref = system_ref.TL(Z_c=zc, omega=omega)
    tl_slab = system_with_slab.TL(Z_c=zc, omega=omega)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    ax1.semilogx(freqs, np.real(z_slab), color="tab:blue", linewidth=2.0, label=r"Re{$Z_{c,slab}$}")
    ax1.semilogx(freqs, np.imag(z_slab), color="tab:orange", linewidth=2.0, label=r"Im{$Z_{c,slab}$}")
    ax1.semilogx(freqs, np.real(k_slab), color="tab:green", linewidth=1.8, linestyle="--", label=r"Re{$k_{slab}$}")
    ax1.semilogx(freqs, np.imag(k_slab), color="tab:red", linewidth=1.8, linestyle="--", label=r"Im{$k_{slab}$}")
    ax1.set_ylabel("Slab params")
    ax1.set_title("Silicone Slab Longitudinal Properties")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.semilogx(freqs, np.abs(zin_ref), "--", color="gray", linewidth=1.6, label=r"|Zin| reference")
    ax2.semilogx(freqs, np.abs(zin_slab), color="tab:red", linewidth=2.0, label=r"|Zin| with silicone slab")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    ax3.semilogx(freqs, tl_ref, "--", color="gray", linewidth=1.6, label="TL reference")
    ax3.semilogx(freqs, tl_slab, color="tab:green", linewidth=2.0, label="TL with silicone slab")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("TL (dB)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="best")

    print(f"Silicone slab review: mean ΔTL = {np.mean(tl_slab - tl_ref):.3f} dB")
    print(f"Silicone slab review: max  ΔTL = {np.max(np.abs(tl_slab - tl_ref)):.3f} dB")

    plt.tight_layout()
    plt.show()
