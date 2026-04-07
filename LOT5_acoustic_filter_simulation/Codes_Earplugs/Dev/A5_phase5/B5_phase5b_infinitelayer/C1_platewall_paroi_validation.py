"""Validate the infinite-plate impedance/TL model against analytical paroi formulas."""

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

import toolkitsd.acoustmm as acoustmm_pkg
from toolkitsd.acoustmm import AcousticParameters, InfinitePlate, calculate_zp_parois_simple, integrate_3d_diffuse, tl_paroi_analytic


def param_plate_glass_standard() -> dict[str, float]:
    # Typical soda-lime glass plate
    return {"name": "Glass (5 mm)", "rho": 2500.0, "h": 5.0e-3, "E": 70.0e9, "nu": 0.22}


def param_plate_aluminum_standard() -> dict[str, float]:
    # Typical aluminum sheet
    return {"name": "Aluminum (2 mm)", "rho": 2700.0, "h": 2.0e-3, "E": 69.0e9, "nu": 0.33}


def param_plate_steel_standard() -> dict[str, float]:
    # Typical low-carbon steel sheet
    return {"name": "Steel (1 mm)", "rho": 7850.0, "h": 1.0e-3, "E": 210.0e9, "nu": 0.30}

def param_silicone_standard() -> dict[str, float]:
    """
    Typical silicone elastomer parameters.
    Values representative of soft RTV silicones used in acoustics / ear molds.
    """
    return {
        "name": "Silicone",
        "rho": 1500.0,
        "E": 1.7e6,
        "nu": 0.48,
        "loss_factor": 0.18,
        "blocked_pressure": 2.0,
        "h":0.03
    }

if __name__ == "__main__":
    print(f"Using acoustmm from: {acoustmm_pkg.__file__}")
    freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 800)
    params = AcousticParameters(freqs, c0=343.0, rho0=1.2)
    omega = 2.0 * np.pi * freqs
    k0 = omega / params.c0
    z0 = params.rho0 * params.c0

    # Oblique-incidence legacy case (40 degrees) used in the old paroi script.
    theta = np.deg2rad(40.0)

    materials = [
        param_plate_glass_standard(),
        param_plate_aluminum_standard(),
        param_plate_steel_standard(),
        param_silicone_standard(),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, (ax_z, ax_tl, ax_tl_diff) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for mat, color in zip(materials, colors):
        mu = mat["rho"] * mat["h"]

        z_analytic = calculate_zp_parois_simple(
            omega=omega,
            k0=k0,
            theta=theta,
            E=mat["E"],
            h=mat["h"],
            nu=mat["nu"],
            mu=mu,
        )

        plate = InfinitePlate(
            rho_plate=mat["rho"],
            h=mat["h"],
            E=mat["E"],
            nu=mat["nu"],
            theta=theta,
            c0=params.c0,
        )

        z_elem = plate.specific_impedance(omega)
        tl_elem = plate.TL(Z_c=z0, omega=omega)
        tl_diff_elem = plate.TL_diffuse(Z_c=z0, omega=omega, theta_lim=np.pi / 2.0 - 1e-9, n_eval=50)
        tl_from_z = 20.0 * np.log10(np.abs((2.0 + z_analytic * np.cos(theta) / z0) / 2.0))
        tl_analytic = tl_paroi_analytic(
            omega=omega,
            theta=theta,
            rho0=params.rho0,
            c0=params.c0,
            mu=mu,
            E=mat["E"],
            h=mat["h"],
            nu=mat["nu"],
        )
        tl_diff_analytic = integrate_3d_diffuse(
            lambda f, theta_arr: tl_paroi_analytic(
                omega=2.0 * np.pi * f,
                theta=theta_arr,
                rho0=params.rho0,
                c0=params.c0,
                mu=mu,
                E=mat["E"],
                h=mat["h"],
                nu=mat["nu"],
            ),
            frequencies=freqs,
            theta_lim=np.pi / 2.0 - 1e-9,
            n_eval=50,
        )

        ax_z.semilogx(freqs, np.real(z_analytic), color=color, linestyle="-", linewidth=2.0, label=f"Re(Z) {mat['name']}")
        ax_z.semilogx(freqs, np.imag(z_analytic), color=color, linestyle="--", linewidth=1.5, label=f"Im(Z) {mat['name']}")

        ax_tl.semilogx(freqs, tl_analytic, color=color, linestyle="-", linewidth=2.0, label=f"TL analytic {mat['name']}")
        ax_tl.semilogx(freqs, tl_elem, color=color, linestyle=":", linewidth=2.0, label=f"TL element .TL() {mat['name']}")
        ax_tl_diff.semilogx(
            freqs,
            tl_diff_analytic,
            color=color,
            linestyle="-",
            linewidth=2.0,
            label=f"TL diffuse analytic {mat['name']}",
        )
        ax_tl_diff.semilogx(
            freqs,
            tl_diff_elem,
            color=color,
            linestyle=":",
            linewidth=2.0,
            label=f"TL diffuse element {mat['name']}",
        )

        max_abs_diff = float(np.max(np.abs(tl_analytic - tl_elem)))
        max_abs_diff_from_z = float(np.max(np.abs(tl_from_z - tl_elem)))
        max_z_diff = float(np.max(np.abs(z_analytic - z_elem)))
        print(f"{mat['name']}: max |TL_analytic - TL_element| = {max_abs_diff:.3e} dB")
        print(f"{mat['name']}: max |TL_from_Z  - TL_element| = {max_abs_diff_from_z:.3e} dB")
        print(f"{mat['name']}: max |Z_analytic - Z_element|   = {max_z_diff:.3e} Pa.s/m")

    ax_z.set_title("Plate Impedance (simple paroi model)")
    ax_z.set_ylabel(r"$Z_{plate}$ (Pa.s/m)")
    ax_z.grid(True, which="both", alpha=0.3)
    ax_z.legend(loc="best", ncol=2)

    ax_tl.set_title("Transmission Loss: analytic vs InfinitePlate.TL()")
    ax_tl.set_xlabel("Frequency (Hz)")
    ax_tl.set_ylabel("TL (dB)")
    ax_tl.grid(True, which="both", alpha=0.3)
    ax_tl.legend(loc="best", ncol=2)

    ax_tl_diff.set_title("Diffuse Transmission Loss: analytic vs InfinitePlate.TL_diffuse()")
    ax_tl_diff.set_xlabel("Frequency (Hz)")
    ax_tl_diff.set_ylabel("TL diffuse (dB)")
    ax_tl_diff.grid(True, which="both", alpha=0.3)
    ax_tl_diff.legend(loc="best", ncol=2)

    plt.tight_layout()
    plt.show()
