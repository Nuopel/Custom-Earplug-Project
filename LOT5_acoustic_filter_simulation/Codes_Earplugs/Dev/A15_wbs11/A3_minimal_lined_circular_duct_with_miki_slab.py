from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ACOUSTMM_ROOT = HERE.parents[2]
REFACTOR_ROOT = HERE.parents[3]
candidate_paths = [ACOUSTMM_ROOT / "src"]
candidate_paths.extend(sorted(REFACTOR_ROOT.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from A0_minimal_lined_circular_duct import LinedCylindricalDuct
from A2_Slab_duct_eq import extract_equivalent_section_from_matrix
from toolkitsd.acoustmm import CylindricalDuct, MikiLayer


def rigid_backed_surface_impedance_from_equivalent_section(
    k_eq: np.ndarray,
    zc_eq: np.ndarray,
    *,
    thickness: float,
    area: float,
    omega: np.ndarray,
) -> np.ndarray:
    """Convert retrieved section parameters into rigid-backed wall impedance."""
    zc_specific_eq = zc_eq * area
    rho_eq = zc_specific_eq * k_eq / omega
    bulk_eq = omega * zc_specific_eq / k_eq

    k_medium = omega * np.lib.scimath.sqrt(rho_eq / bulk_eq)
    zc_specific_medium = np.lib.scimath.sqrt(rho_eq * bulk_eq)

    k_medium = np.where(np.abs(k_medium - k_eq) <= np.abs(-k_medium - k_eq), k_medium, -k_medium)
    zc_specific_medium = np.where(
        np.abs(zc_specific_medium - zc_specific_eq) <= np.abs(-zc_specific_medium - zc_specific_eq),
        zc_specific_medium,
        -zc_specific_medium,
    )
    return -1j * zc_specific_medium / np.tan(k_medium * thickness)


def plot_comparison(
    freqs_hz: np.ndarray,
    tl_rigid_db: np.ndarray,
    tl_miki_direct_db: np.ndarray,
    tl_miki_recovered_db: np.ndarray,
    zs_direct: np.ndarray,
    zs_recovered: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, tl_rigid_db, linewidth=2.0, label="Rigid cylindrical duct")
    axes[0].semilogx(freqs_hz, tl_miki_direct_db, linewidth=2.0, label="Lined duct from Miki directly")
    axes[0].semilogx(freqs_hz, tl_miki_recovered_db, "--", linewidth=2.0, label="Lined duct from recovered slab")
    axes[0].set_ylabel("Transmission Loss [dB]")
    axes[0].set_title("Direct Miki lining vs recovered-slab lining")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, np.abs(zs_direct), linewidth=2.0, label=r"$|Z_s|$ direct Miki")
    axes[1].semilogx(freqs_hz, np.abs(zs_recovered), "--", linewidth=2.0, label=r"$|Z_s|$ recovered slab")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$|Z_s|$ [Pa.s/m]")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")


if __name__ == "__main__":
    C0 = 343.0
    RHO0 = 1.2

    freqs_hz = np.logspace(np.log10(50.0), np.log10(5000.0), 300)
    omega = 2.0 * np.pi * freqs_hz

    sigma = 15000.0
    slab_length = 6.6e-3
    lining_thickness = 3e-3
    slab_radius = 3.5e-3
    radius_cyl = 0.01
    length_duct = 0.1

    rigid_cyl_duct = CylindricalDuct(radius=radius_cyl, length=length_duct, c0=C0, rho0=RHO0)

    # Direct Miki-lined duct.
    direct_lining_material = MikiLayer(
        sigma=sigma,
        length=lining_thickness,
        area=np.pi * slab_radius**2,
        rho0=RHO0,
        c0=C0,
        name="Direct Miki slab",
    ).material

    direct_lined_duct = LinedCylindricalDuct(
        radius=radius_cyl,
        length=length_duct,
        frequencies_hz=freqs_hz,
        lining_material=direct_lining_material,
        c0=C0,
        rho0=RHO0,
    )
    zs_direct = direct_lined_duct.zs_wall

    # Recover k_eq / Zc_eq from a homogeneous Miki slab section, then build Zs.
    miki_slab = MikiLayer(
        sigma=sigma,
        length=slab_length,
        area=np.pi * slab_radius**2,
        rho0=RHO0,
        c0=C0,
        name="Recovered Miki slab",
    )
    matrix_slab = miki_slab.matrix(omega)
    k_eq, zc_eq = extract_equivalent_section_from_matrix(matrix_slab, omega, slab_length)
    zs_recovered = rigid_backed_surface_impedance_from_equivalent_section(
        k_eq,
        zc_eq,
        thickness=lining_thickness,
        area=np.pi * slab_radius**2,
        omega=omega,
    )
    recovered_lined_duct = LinedCylindricalDuct(
        radius=radius_cyl,
        length=length_duct,
        frequencies_hz=freqs_hz,
        zs_wall=zs_recovered,
        c0=C0,
        rho0=RHO0,
    )

    tl_rigid_db = rigid_cyl_duct.TL(Z_c=rigid_cyl_duct.Zc, omega=omega)
    tl_miki_direct_db = direct_lined_duct.TL(Z_c=rigid_cyl_duct.Zc, omega=omega)
    tl_miki_recovered_db = recovered_lined_duct.TL(Z_c=rigid_cyl_duct.Zc, omega=omega)

    plot_comparison(freqs_hz, tl_rigid_db, tl_miki_direct_db, tl_miki_recovered_db, zs_direct, zs_recovered)
    plt.show()
