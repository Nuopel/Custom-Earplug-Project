"""Bare-canal example: quarter-wave resonance check with eardrum load."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, EarCanalBuilder, EardrumImpedance, IEC711Coupler


if __name__ == "__main__":
    freqs = np.logspace(np.log10(100.0), np.log10(10000.0), 800)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    # Uniform 25 mm canal profile so f_qw ≈ c0/(4L) ≈ 3.4 kHz.
    L_canal = 25e-3
    r_canal = 4.0e-3
    x = np.linspace(0.0, L_canal, 9)
    r = np.full_like(x, r_canal)

    canal = EarCanalBuilder(n_segments=40, radius_scale=1.0, c0=params.c0, rho0=params.rho0).build(
        x=x, radius=r
    )
    z_tm = EardrumImpedance().Z(omega)
    z_tm = IEC711Coupler(model="tmm").Z(omega)
    # z_tm = IEC711Coupler(model="lumped").Z(omega)
    zin = canal.Z_in(z_tm, omega)
    p_tm = canal.p_tm(1.0, z_tm, omega)
    p_tm_db = 20.0 * np.log10(np.abs(p_tm))

    f_qw = params.c0 / (4.0 * L_canal)
    search = (freqs > 2000.0) & (freqs < 5000.0)
    idx_peak = np.argmax(p_tm_db[search])
    f_peak = float(freqs[search][idx_peak])
    peak_db = float(p_tm_db[search][idx_peak])

    print(f"Expected quarter-wave frequency: {f_qw:.1f} Hz")
    print(f"Detected p_TM peak frequency:    {f_peak:.1f} Hz")
    print(f"Detected p_TM peak level:        {peak_db:.2f} dB")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.semilogx(freqs, np.abs(zin), color="tab:blue", linewidth=2.0, label=r"$|Z_{in}|$")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    ax1.set_title("Bare Canal + Eardrum Load")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.semilogx(freqs, p_tm_db, color="tab:red", linewidth=2.0, label=r"$20\log_{10}|p_{TM}/p_{in}|$")
    ax2.axvline(f_qw, color="black", linestyle="--", linewidth=1.4, label=f"Expected f_qw={f_qw:.0f} Hz")
    ax2.axvline(f_peak, color="tab:green", linestyle=":", linewidth=1.8, label=f"Detected peak={f_peak:.0f} Hz")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("p_TM/p_in (dB)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # --- Phase 4 sensitivity sweep: radius variation ±1.5 mm around 4.0 mm ---
    radius_scales = [0.625, 1.0, 1.375]  # 2.5 mm, 4.0 mm, 5.5 mm equivalent mean radius
    colors = ["tab:purple", "tab:blue", "tab:green"]

    fig2, (bx1, bx2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    print("\nRadius sensitivity sweep (±1.5 mm around 4.0 mm):")
    for s, c in zip(radius_scales, colors):
        canal_s = EarCanalBuilder(
            n_segments=40, radius_scale=s, c0=params.c0, rho0=params.rho0
        ).build(x=x, radius=r)
        zin_s = canal_s.Z_in(z_tm, omega)
        p_tm_s_db = 20.0 * np.log10(np.abs(canal_s.p_tm(1.0, z_tm, omega)))

        idx_peak_s = np.argmax(p_tm_s_db[search])
        f_peak_s = float(freqs[search][idx_peak_s])
        r_mm = r_canal * s * 1e3
        print(f"  radius_scale={s:.3f} (mean r≈{r_mm:.2f} mm) -> peak {f_peak_s:.1f} Hz")

        bx1.semilogx(freqs, np.abs(zin_s), color=c, linewidth=2.0, label=f"|Zin|, scale={s:.3f}")
        bx2.semilogx(freqs, p_tm_s_db, color=c, linewidth=2.0, label=f"pTM/pin, scale={s:.3f}")

    bx1.set_yscale("log")
    bx1.set_ylabel(r"$|Z_{in}|$ (Pa·s/m$^3$)")
    bx1.set_title("Bare Canal Radius Sensitivity Sweep")
    bx1.grid(True, which="both", alpha=0.3)
    bx1.legend(loc="best")

    bx2.set_xlabel("Frequency (Hz)")
    bx2.set_ylabel("p_TM/p_in (dB)")
    bx2.grid(True, which="both", alpha=0.3)
    bx2.legend(loc="best")

    plt.tight_layout()
    plt.show()
