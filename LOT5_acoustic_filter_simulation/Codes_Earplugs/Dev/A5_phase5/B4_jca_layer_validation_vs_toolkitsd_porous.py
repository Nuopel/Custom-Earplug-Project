"""Validate acoustmm JCALayer against toolkitsd.porous rigid-backing response."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import AcousticParameters, JCALayer, RigidWall
from toolkitsd.porous import JCAModel, PorousMaterial, PorousMediumProps, surface_response_on_rigid_backing


if __name__ == "__main__":
    freqs = np.logspace(np.log10(50.0), np.log10(20000.0), 400)
    params = AcousticParameters(freqs, c0=342.2, rho0=1.213)
    omega = 2.0 * np.pi * freqs

    radius = 10e-3
    area = np.pi * radius**2
    thickness = 0.04

    # Shared material definition (same as porous validation example preset).
    mat = PorousMaterial.get_material_preset("melamine_cttm", rho0=params.rho0, c0=params.c0)

    jca_layer = JCALayer(
        phi=mat.phi,
        sigma=mat.sigma,
        alpha_inf=mat.tortu,
        lambda_v=mat.lambda1,
        lambda_t=mat.lambdap,
        length=thickness,
        area=area,
        rho0=mat.rho0,
        c0=mat.c0,
        name="JCA layer (acoustmm)",
    )

    z_tmm_vol = jca_layer.Z_in(RigidWall().Z(omega), omega)
    z_tmm = z_tmm_vol * area  # convert volume-velocity impedance -> surface impedance
    r_tmm = (z_tmm - mat.z0) / (z_tmm + mat.z0)
    abs_tmm = 1.0 - np.abs(r_tmm) ** 2

    pprops = PorousMediumProps.from_material(mat, freqs, model=JCAModel())
    surf = surface_response_on_rigid_backing(pprops, incidence_angle_deg=90.0)
    z_ref = surf.surface_impedance[:, 0]
    abs_ref = surf.absorption[:, 0]

    err_z = np.abs(z_tmm - z_ref) / np.maximum(np.abs(z_ref), 1e-12)
    err_a = np.abs(abs_tmm - abs_ref)
    print(f"JCA validation: mean rel error Zs = {np.mean(err_z):.4e}")
    print(f"JCA validation: max  rel error Zs = {np.max(err_z):.4e}")
    print(f"JCA validation: mean abs error alpha = {np.mean(err_a):.4e}")
    print(f"JCA validation: max  abs error alpha = {np.max(err_a):.4e}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax1.semilogx(freqs, np.real(z_ref) / mat.z0, color="tab:blue", linewidth=2.0, label="Re(Zs)/Z0 porous")
    ax1.semilogx(freqs, np.real(z_tmm) / mat.z0, "--", color="tab:blue", linewidth=1.6, label="Re(Zs)/Z0 TMM")
    ax1.semilogx(freqs, np.imag(z_ref) / mat.z0, color="tab:orange", linewidth=2.0, label="Im(Zs)/Z0 porous")
    ax1.semilogx(freqs, np.imag(z_tmm) / mat.z0, "--", color="tab:orange", linewidth=1.6, label="Im(Zs)/Z0 TMM")
    ax1.set_ylabel(r"$Z_s / Z_0$")
    ax1.set_title("JCALayer Validation: Surface Impedance (Rigid Backing, Normal Incidence)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", ncol=2)

    ax2.semilogx(freqs, abs_ref, color="tab:green", linewidth=2.0, label="Absorption porous")
    ax2.semilogx(freqs, abs_tmm, "--", color="tab:red", linewidth=2.0, label="Absorption TMM")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel(r"$\alpha$")
    ax2.set_ylim(0.0, 1.1)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()
