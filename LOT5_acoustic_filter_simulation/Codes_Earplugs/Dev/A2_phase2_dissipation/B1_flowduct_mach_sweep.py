"""FlowDuct example: Mach sweep effects on transmission phase and input impedance."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, FlowDuct, RigidWall


if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(8000.0), 500)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    radius = 3.0e-3
    length = 0.3
    area = np.pi * radius**2
    zc = params.rho0 * params.c0 / area
    z_load = RigidWall().Z(omega)

    mach_values = [0.0, 0.05, 0.10]
    colors = ["black", "tab:blue", "tab:red"]

    fig, (ax_phase, ax_zin) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    tau_by_m = {}
    zin_by_m = {}
    for m, c in zip(mach_values, colors):
        duct = FlowDuct(radius=radius, length=length, mach=m, c0=params.c0, rho0=params.rho0)
        T = duct.matrix(omega)
        tau = 2.0 / (T[:, 0, 0] + T[:, 0, 1] / zc + T[:, 1, 0] * zc + T[:, 1, 1])
        zin = duct.Z_in(z_load, omega)
        tau_by_m[m] = tau
        zin_by_m[m] = zin
        phase = np.unwrap(np.angle(tau))

        ax_phase.semilogx(freqs, phase, color=c, linewidth=2.0, label=f"Phase, M={m:.2f}")
        ax_zin.semilogx(freqs, np.abs(zin), color=c, linewidth=2.0, label=f"|Zin|, M={m:.2f}")

    # Quick numeric review in console for engineering sanity checks.
    phase0 = np.unwrap(np.angle(tau_by_m[0.0]))
    phase1 = np.unwrap(np.angle(tau_by_m[0.10]))
    dphase = phase1 - phase0
    dmag_zin = np.abs(zin_by_m[0.10]) - np.abs(zin_by_m[0.0])
    print(f"FlowDuct review: RMS Δphase(M=0.10 vs 0.00) = {np.sqrt(np.mean(dphase**2)):.4f} rad")
    print(f"FlowDuct review: max Δphase(M=0.10 vs 0.00) = {np.max(np.abs(dphase)):.4f} rad")
    print(f"FlowDuct review: RMS Δ|Zin|(M=0.10 vs 0.00) = {np.sqrt(np.mean(dmag_zin**2)):.3e} Pa·s/m^3")
    print(f"FlowDuct review: max Δ|Zin|(M=0.10 vs 0.00) = {np.max(np.abs(dmag_zin)):.3e} Pa·s/m^3")

    ax_phase.set_ylabel("Transmission Phase (rad)")
    ax_phase.set_title("FlowDuct Mach Sweep")
    ax_phase.grid(True, which="both", alpha=0.3)
    ax_phase.legend(loc="best")

    ax_zin.set_xlabel("Frequency (Hz)")
    ax_zin.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax_zin.set_yscale("log")
    ax_zin.grid(True, which="both", alpha=0.3)
    ax_zin.legend(loc="best")

    plt.tight_layout()
    plt.show()
