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

from function import build_fem_element_from_sparameters, plot_matrix_comparison
from toolkitsd.acoustmm import (
    CylindricalDuct,
    ElasticSlab,
    ElasticSlabSeries,
    ElasticSlabThin,
    IEC711Coupler,
    ViscothermalDuct,
)


def interpolate_complex(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    y_src = np.asarray(y_src, dtype=np.complex128)
    y_real = np.interp(x_dst, x_src, np.real(y_src))
    y_imag = np.interp(x_dst, x_src, np.imag(y_src))
    return y_real + 1j * y_imag


def plot_iec_il_models(
    freqs: np.ndarray,
    il_tmm_models: dict[str, np.ndarray],
    freqs_fem: np.ndarray,
    il_fem_db: np.ndarray,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    for model_name, il_db in il_tmm_models.items():
        ax.semilogx(freqs, il_db, linewidth=2.0, label=f"{model_name} TMM")
    ax.semilogx(freqs_fem, il_fem_db, "--", linewidth=2.2, label="FEM")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("IEC711 insertion loss for bulk slab models")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    FEM_FILE = HERE / "fem_rslt" / "rslt_fem_bulk_silicone_rollerslab.txt"

    C0 = 343.2
    RHO0 = 1.2043

    length_inlet = 5e-3
    r_inlet = 5e-3
    r_outlet = r_inlet
    length_outlet = length_inlet

    r_duct = r_inlet
    slab_length = 3.0e-3
    cavity_each_side = 1.0e-2

    p_incident = 1.0

    slab_params = {
        "radius": r_duct,
        "length": slab_length,
        "rho": 970.0,
        "young": 1.0e6,
        "poisson": 0.45,
        "loss_factor": 0.02,
    }

    s_in = np.pi * r_inlet**2
    s_out = np.pi * r_outlet**2

    freqs = np.logspace(np.log10(50.0), np.log10(4000.0), 400)
    omega = 2.0 * np.pi * freqs

    total_length = length_inlet + 2.0 * cavity_each_side + slab_length + length_outlet
    z0_in = np.full(omega.shape, RHO0 * C0 / s_in + 0j, dtype=np.complex128)
    k0 = omega / C0

    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=C0, rho0=RHO0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=C0, rho0=RHO0)
    halfduct = ViscothermalDuct(radius=r_duct, length=cavity_each_side, c0=C0, rho0=RHO0)

    slab_models = {
        "B1_exact": ElasticSlab(**slab_params),
        "B2_thin": ElasticSlabThin(**slab_params),
        "B3_series": ElasticSlabSeries(**slab_params),
    }

    system_air_only = inlet + halfduct + halfduct + outlet
    z_711 = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega)
    p_end_iec711_air_only = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]

    (
        freqs_fem,
        omega_fem,
        z01_fem,
        _z02_fem,
        _k01_fem,
        _k02_fem,
        _s11,
        _s21,
        matrix_fem,
        fem_element,
    ) = build_fem_element_from_sparameters(
        FEM_FILE,
        s_in,
        s_out,
        RHO0,
        C0,
    )

    z_711_fem = IEC711Coupler(c0=C0, rho0=RHO0).Z(omega_fem)
    p_end_iec711_filter_fem = fem_element.state_tm_from_incident_wave(
        p_incident,
        z_711_fem,
        z01_fem,
        omega_fem,
    )[:, 0]
    p_end_iec711_air_fem = interpolate_complex(freqs, p_end_iec711_air_only, freqs_fem)
    il_iec711_fem_db = 20.0 * np.log10(
        np.maximum(np.abs(p_end_iec711_air_fem / p_end_iec711_filter_fem), np.finfo(float).tiny)
    )

    il_tmm_models = {}
    for model_name, slab_model in slab_models.items():
        system_filter = inlet + halfduct + slab_model + halfduct + outlet

        if model_name == "B1_exact":
            matrix_tmm_exact = system_filter.matrix(omega)
            plot_matrix_comparison(
                freqs,
                matrix_tmm_exact,
                freqs_fem,
                matrix_fem,
                mode="abs_phase",
                title_prefix="Bulk silicone slab: B1 exact vs FEM",
            )
            plt.show()

        p_end_iec711_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in, omega)[:, 0]
        il_tmm_models[model_name] = 20.0 * np.log10(
            np.maximum(np.abs(p_end_iec711_air_only / p_end_iec711_filter), np.finfo(float).tiny)
        )

    plot_iec_il_models(freqs, il_tmm_models, freqs_fem, il_iec711_fem_db)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax.semilogx(freqs, il_tmm_models["B1_exact"], linewidth=2.2, label="B1 exact")
    ax.semilogx(freqs, il_tmm_models["B2_thin"], linewidth=2.0, label="B2 thin")
    ax.semilogx(freqs, il_tmm_models["B3_series"], linewidth=2.0, label="B3 series")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("IL [dB]")
    ax.set_title("Bulk silicone slab: B2/B3 validity against B1")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    plt.show()

    e_complex = slab_params["young"] * (1.0 + 1j * slab_params["loss_factor"])
    m_long = e_complex * (1.0 - slab_params["poisson"]) / (
        (1.0 + slab_params["poisson"]) * (1.0 - 2.0 * slab_params["poisson"])
    )
    c_long = np.sqrt(m_long / slab_params["rho"])
    zc_long = slab_params["rho"] * c_long / s_in
    print(f"M_L = {m_long:.3e} Pa")
    print(f"c_L = {c_long:.3e} m/s")
    print(f"Zc_L = {zc_long:.3e} Pa.s/m^3")
