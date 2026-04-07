"""Phase 1 example: exact conical duct vs successive-cylinder approximation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, ConicalDuct, CylindricalDuct


def successive_cone_approx(
    r1: float,
    r2: float,
    length: float,
    n_sub: int,
    *,
    c0: float,
    rho0: float,
):
    radii = np.linspace(r1, r2, n_sub + 1)
    r_mid = 0.5 * (radii[:-1] + radii[1:])
    sub_length = length / n_sub
    segments = [CylindricalDuct(radius=r, length=sub_length, c0=c0, rho0=rho0) for r in r_mid]
    return sum(segments)


def main() -> None:
    freqs = np.linspace(50.0, 4000.0, 800)
    params = AcousticParameters(freqs, c0=340.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs

    r1 = 2.0e-3
    r2 = 5.0e-3
    length = 35.0e-3
    n_list = [4, 16, 64]

    exact = ConicalDuct(r1=r1, r2=r2, length=length, c0=params.c0, rho0=params.rho0)
    S1 = np.pi * r1**2
    zc_ref = params.rho0 * params.c0 / S1
    tl_exact = exact.TL(Z_c=zc_ref, omega=omega)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.plot(freqs, tl_exact, color="black", linewidth=2.3, label="Conical exact")

    for n_sub in n_list:
        approx = successive_cone_approx(r1, r2, length, n_sub, c0=params.c0, rho0=params.rho0)
        tl_approx = approx.TL(Z_c=zc_ref, omega=omega)
        err = np.abs(tl_approx - tl_exact)
        ax1.plot(freqs, tl_approx, "--", linewidth=1.6, label=f"Successive cylinders N={n_sub}")
        ax2.plot(freqs, err, linewidth=1.8, label=f"N={n_sub} (max={np.max(err):.3f} dB)")

    ax1.set_ylabel("TL (dB)")
    ax1.set_title("Conical Duct: Exact vs Successive-Cylinder Approximation")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="best")

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"|ΔTL| vs exact (dB)")
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
