"""Munjal simple expansion muffler TL: analytical reference vs acoustmm usage.
2.19 TL of a Simple Expansion Chamber Mufﬂer"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

from toolkitsd.acoustmm import AcousticParameters, CylindricalDuct, first_mode_round_duct, tl_simple_expansion_analytic


def pi_formatter(val, _pos):
    n = int(np.round(val / np.pi))
    if n == 0:
        return "0"
    if n == 1:
        return r"$\pi$"
    return rf"${n}\pi$"


def tl_simple_expansion_tmm_from_elements(
    frequencies: np.ndarray,
    chamber_length: float,
    area_ratio: float,
    *,
    c0: float = 340.0,
    rho0: float = 1.2,
    inlet_radius: float = 4e-3,
) -> np.ndarray:
    freqs = np.asarray(frequencies, dtype=np.float64).ravel()
    S1 = np.pi * inlet_radius**2
    S2 = float(area_ratio) * S1
    r2 = np.sqrt(S2 / np.pi)
    omega = 2.0 * np.pi * freqs
    Zc1 = rho0 * c0 / S1
    chamber = CylindricalDuct(radius=r2, length=chamber_length, c0=c0, rho0=rho0)
    return chamber.TL(Z_c=Zc1, omega=omega)


if __name__ == "__main__":
    freqs = np.linspace(1.0, 2000.0, 200)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    chamber_length = 1.0
    inlet_radius = 0.025
    area_ratios = [4.0, 9.0, 16.0]

    k0l = params.wavenumbers * chamber_length

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(area_ratios)))
    for color, m in zip(colors, area_ratios):
        tl_ref = tl_simple_expansion_analytic(freqs, chamber_length, m, c0=params.c0)
        tl_tmm = tl_simple_expansion_tmm_from_elements(
            freqs,
            chamber_length,
            m,
            c0=params.c0,
            rho0=params.rho0,
            inlet_radius = inlet_radius

        )
        chamber_radius = inlet_radius * np.sqrt(m)
        _, f_cut = first_mode_round_duct(a=chamber_radius, c0=params.c0, bc="rigid")
        print(f"Cutoff frequency radius {round(chamber_radius,2)} m : {f_cut:.1f} Hz")
        k0l_cut = (2.0 * np.pi * f_cut / params.c0) * chamber_length

        ax.plot(k0l, tl_ref, color=color, linestyle="-", linewidth=2.0)
        ax.plot(k0l, tl_tmm, color=color/2, linestyle="--", linewidth=1.6)
        if k0l_cut <= 3.25 * np.pi:
            ax.axvline(k0l_cut, color=color, linestyle=":", linewidth=1.4, alpha=0.9)

    ax.set_xlabel(r"$k_0 L$")
    ax.set_ylabel("Transmission Loss (dB)")
    ax.set_title("Simple Expansion Chamber TL (Munjal Reference)")
    ax.set_xlim([0.0, 3.25 * np.pi])
    ax.set_ylim([0.0, 25.0])
    ax.set_xticks([np.pi, 2.0 * np.pi, 3.0 * np.pi])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))
    ax.grid(True, which="both", ls="-")

    ratio_handles = [Line2D([0], [0], color=c, lw=2.0, label=f"m={m:g}") for c, m in zip(colors, area_ratios)]
    method_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Analytical (Munjal)"),
        Line2D([0], [0], color="black", lw=1.6, linestyle="--", label="TMM"),
        Line2D([0], [0], color="black", lw=1.4, linestyle=":", label="1st mode cutoff"),
    ]
    legend_ratio = ax.legend(handles=ratio_handles, title="Area Ratio", loc="upper left")
    ax.add_artist(legend_ratio)
    ax.legend(handles=method_handles, title="Method", loc="lower right")

    plt.show()
