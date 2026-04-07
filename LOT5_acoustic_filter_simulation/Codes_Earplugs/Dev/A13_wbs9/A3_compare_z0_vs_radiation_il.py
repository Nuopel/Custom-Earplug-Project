from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np

from toolkitsd.acoustmm import CylindricalDuct, IEC711Coupler, JCALayer, RadiationImpedance, ViscothermalDuct
from toolkitsd.porous import PorousMaterial


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    length_outlet = 5e-3
    r_inlet = 5e-3
    r_outlet = r_inlet
    r_duct = r_inlet
    cavity_each_side = 10e-3
    porous_thickness = 10e-3
    p_incident = 1.0

    area = np.pi * r_inlet**2
    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 200)
    omega = 2.0 * np.pi * freqs

    z0_plane = np.full(omega.shape, RHO0 * C0 / area + 0j, dtype=np.complex128)
    z0_radiation = RadiationImpedance(radius=r_inlet, mode="unflanged_v2", c0=C0, rho0=RHO0).Z(omega)
    z0_radiation = RadiationImpedance(radius=r_inlet, mode="flanged", c0=C0, rho0=RHO0).Z(omega)

    mat = PorousMaterial.get_material_preset("melamine_cttm", rho0=RHO0, c0=C0)
    porous_layer = JCALayer(
        phi=mat.phi,
        sigma=mat.sigma,
        alpha_inf=mat.tortu,
        lambda_v=mat.lambda1,
        lambda_t=mat.lambdap,
        length=porous_thickness,
        area=area,
        rho0=RHO0,
        c0=C0,
        name="JCA layer",
    )

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=cavity_each_side, c0=C0, rho0=RHO0)
    filter_system = inlet + halfduct + porous_layer + halfduct + outlet
    air_system = inlet + halfduct + halfduct + outlet

    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    z_rigid = np.full(omega.shape, np.inf + 0.0j, dtype=np.complex128)

    p_end_air_plane_iec = air_system.state_tm_from_incident_wave(p_incident, z_711, z0_plane, omega)[:, 0]
    p_end_filter_plane_iec = filter_system.state_tm_from_incident_wave(p_incident, z_711, z0_plane, omega)[:, 0]
    p_end_air_rad_iec = air_system.state_tm_from_incident_wave(p_incident, z_711, z0_radiation, omega)[:, 0]
    p_end_filter_rad_iec = filter_system.state_tm_from_incident_wave(p_incident, z_711, z0_radiation, omega)[:, 0]

    p_end_air_plane_rigid = air_system.state_tm_from_incident_wave(p_incident, z_rigid, z0_plane, omega)[:, 0]
    p_end_filter_plane_rigid = filter_system.state_tm_from_incident_wave(p_incident, z_rigid, z0_plane, omega)[:, 0]
    p_end_air_rad_rigid = air_system.state_tm_from_incident_wave(p_incident, z_rigid, z0_radiation, omega)[:, 0]
    p_end_filter_rad_rigid = filter_system.state_tm_from_incident_wave(p_incident, z_rigid, z0_radiation, omega)[:, 0]

    il_plane_iec = 20.0 * np.log10(np.maximum(np.abs(p_end_air_plane_iec / p_end_filter_plane_iec), np.finfo(float).tiny))
    il_rad_iec = 20.0 * np.log10(np.maximum(np.abs(p_end_air_rad_iec / p_end_filter_rad_iec), np.finfo(float).tiny))
    il_plane_rigid = 20.0 * np.log10(np.maximum(np.abs(p_end_air_plane_rigid / p_end_filter_plane_rigid), np.finfo(float).tiny))
    il_rad_rigid = 20.0 * np.log10(np.maximum(np.abs(p_end_air_rad_rigid / p_end_filter_rad_rigid), np.finfo(float).tiny))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].semilogx(freqs, il_plane_iec, linewidth=2.2, label=r"Plane-wave $Z_0=\rho_0 c_0/S$")
    axes[0].semilogx(freqs, il_rad_iec, "--", linewidth=2.0, label="Flanged radiation impedance")
    axes[0].set_ylabel("IL [dB]")
    axes[0].set_title("IEC711 IL: plane-wave Z0 vs radiation impedance")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs, il_plane_rigid, linewidth=2.2, label=r"Plane-wave $Z_0=\rho_0 c_0/S$")
    axes[1].semilogx(freqs, il_rad_rigid, "--", linewidth=2.0, label="Flanged radiation impedance")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Rigid-load IL: plane-wave Z0 vs radiation impedance")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()
