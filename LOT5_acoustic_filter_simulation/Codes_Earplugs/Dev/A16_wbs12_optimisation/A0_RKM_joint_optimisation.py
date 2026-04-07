from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from toolkitsd.acoustmm import (
    CylindricalDuct,
    GenericFilmSeriesImpedance,
    IEC711Coupler,
    ImpedanceJunction,
    ViscothermalDuct,
    MikiLayer,
    RadiationImpedance
)



def compute_il_iec711_db(
    *,
    freqs_hz: np.ndarray,
    film_resistance: float,
    film_mass: float,
    film_stiffness: float,
    c0: float,
    rho0: float,
    length_inlet: float,
    length_outlet: float,
    r_inlet: float,
    r_outlet: float,
    r_duct: float,
    length_duct: float,
    p_incident: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * freqs_hz
    s_in = np.pi * r_inlet**2
    s_out = np.pi * r_outlet**2
    s_duct = np.pi * r_duct**2
    r0_iec = 3.77e-3
    s_iec = np.pi * r0_iec ** 2

    z0_in = np.full(omega.shape, rho0 * c0 / s_in + 0j, dtype=np.complex128)
    z0_in_filter = RadiationImpedance(radius=r_duct, mode="flanged", c0=c0, rho0=rho0).Z(omega)
    z0_in_inlet = RadiationImpedance(radius=r_inlet, mode="flanged", c0=c0, rho0=rho0).Z(omega)


    inlet = CylindricalDuct(radius=r_inlet, length=length_inlet, c0=c0, rho0=rho0)
    outlet = CylindricalDuct(radius=r_outlet, length=length_outlet, c0=c0, rho0=rho0)
    halfduct = ViscothermalDuct(radius=r_duct, length=length_duct / 2.0, c0=c0, rho0=rho0)
    duct_air = ViscothermalDuct(radius=r_inlet, length=length_duct , c0=c0, rho0=rho0)
    junction_reduction = ImpedanceJunction(s_in, s_duct, rho0=rho0, end_correction=True)
    junction_expansion = ImpedanceJunction(s_duct, s_out, rho0=rho0, end_correction=True)
    junction_iec = ImpedanceJunction(s_out, s_iec, rho0=rho0, end_correction=True)

    film = GenericFilmSeriesImpedance(
        resistance=film_resistance,
        mass=film_mass,
        stiffness=film_stiffness,
    )

    system_filter = inlet + junction_reduction + halfduct + film + halfduct + junction_expansion + outlet + junction_iec
    system_air_only = inlet + duct_air + outlet + junction_iec
    z_711 = IEC711Coupler(c0=c0, rho0=rho0).Z(omega)

    p_end_air = system_air_only.state_tm_from_incident_wave(p_incident, z_711, z0_in_inlet, omega)[:, 0]
    p_end_filter = system_filter.state_tm_from_incident_wave(p_incident, z_711, z0_in_inlet, omega)[:, 0]
    il_iec711_db = 20.0 * np.log10(np.maximum(np.abs(p_end_air / p_end_filter), np.finfo(float).tiny))
    return p_end_air, p_end_filter, il_iec711_db


def evaluate_target_metrics(
    freqs_hz: np.ndarray,
    il_db: np.ndarray,
    *,
    target_il_db: float,
    band_hz: tuple[float, float],
    tolerance_db: float,
) -> dict[str, float]:
    band_mask = (freqs_hz >= band_hz[0]) & (freqs_hz <= band_hz[1])
    if not np.any(band_mask):
        raise ValueError("Target band does not intersect the frequency grid.")

    il_band = il_db[band_mask]
    abs_error = np.abs(il_band - target_il_db)
    median_error = float(np.abs(np.median(il_band) - target_il_db))
    flatness_error = float(np.percentile(il_band, 90.0) - np.percentile(il_band, 10.0))
    bandwidth_error = float(np.mean(np.maximum(0.0, abs_error - tolerance_db) ** 2))
    objective = median_error + 0.25 * flatness_error + 2.0 * bandwidth_error

    return {
        "median_error_db": median_error,
        "flatness_error_db": flatness_error,
        "bandwidth_error_db2": bandwidth_error,
        "objective": objective,
    }


def kmr_objective_from_log10(
    log10_vars: np.ndarray,
    *,
    freqs_hz: np.ndarray,
    target_il_db: float,
    band_hz: tuple[float, float],
    tolerance_db: float,
    c0: float,
    rho0: float,
    length_inlet: float,
    length_outlet: float,
    r_inlet: float,
    r_outlet: float,
    r_duct: float,
    length_duct: float,
    p_incident: float = 1.0,
) -> float:
    log10_film_stiffness, log10_film_mass, log10_film_resistance = np.asarray(log10_vars, dtype=float)
    film_stiffness = 10.0**log10_film_stiffness
    film_mass = 10.0**log10_film_mass
    film_resistance = 10.0**log10_film_resistance

    _, _, il_iec711_db = compute_il_iec711_db(
        freqs_hz=freqs_hz,
        film_resistance=film_resistance,
        film_mass=film_mass,
        film_stiffness=film_stiffness,
        c0=c0,
        rho0=rho0,
        length_inlet=length_inlet,
        length_outlet=length_outlet,
        r_inlet=r_inlet,
        r_outlet=r_outlet,
        r_duct=r_duct,
        length_duct=length_duct,
        p_incident=p_incident,
    )
    metrics = evaluate_target_metrics(
        freqs_hz,
        il_iec711_db,
        target_il_db=target_il_db,
        band_hz=band_hz,
        tolerance_db=tolerance_db,
    )
    return metrics["objective"]


def plot_single_config_results(
    freqs_hz: np.ndarray,
    p_end_air_iec711: np.ndarray,
    p_end_filter_iec711: np.ndarray,
    il_iec711_db: np.ndarray,
    *,
    title_suffix: str,
    target_il_db: float | None = None,
    band_hz: tuple[float, float] | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_air_iec711)), "--", lw=2.0, label="Air cavity only, IEC711")
    axes[0].semilogx(freqs_hz, 20.0 * np.log10(np.abs(p_end_filter_iec711)), "--", lw=2.0, label="Filter + air cavity, IEC711")
    axes[0].set_ylabel(r"$20 \log_{10} |p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title(f"End pressure, {title_suffix}")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(freqs_hz, il_iec711_db, "--", lw=2.2, label="IL, IEC711")
    if target_il_db is not None:
        axes[1].axhline(target_il_db, color="k", linestyle=":", lw=1.8, label=f"Target = {target_il_db:.1f} dB")
    if band_hz is not None:
        axes[1].axvspan(band_hz[0], band_hz[1], color="0.85", alpha=0.35, label="Target band")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("IL [dB]")
    axes[1].set_title("Insertion loss from end-pressure ratio")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_ylim([-10, 50])
    plt.show()


if __name__ == "__main__":
    C0 = 343.2
    RHO0 = 1.2043


    r_inlet = 3.5e-3
    r_outlet = 3.5e-3
    r_duct = 1e-3

    total_length = 14e-3
    length_inlet = 1.0e-3
    length_duct = 10e-3
    length_outlet = total_length - length_duct -length_inlet
    p_incident = 1.0

    target_il_db = 20.0
    target_band_hz = (100.0, 10000.0)
    target_tolerance_db = 1.0

    log10_stiffness_bounds = (1.0, 14.0)
    log10_mass_bounds = (-6.0, 8.0)
    log10_resistance_bounds = (1.0, 15.0)
    initial_guess = np.array([11.0, -1.0, 7.0], dtype=float)

    freqs = np.logspace(np.log10(50.0), np.log10(8000.0), 300)

    objective = partial(
        kmr_objective_from_log10,
        freqs_hz=freqs,
        target_il_db=target_il_db,
        band_hz=target_band_hz,
        tolerance_db=target_tolerance_db,
        c0=C0,
        rho0=RHO0,
        length_inlet=length_inlet,
        length_outlet=length_outlet,
        r_inlet=r_inlet,
        r_outlet=r_outlet,
        r_duct=r_duct,
        length_duct=length_duct,
        p_incident=p_incident,
    )

    optimisation_result = minimize(
        objective,
        x0=initial_guess,
        method="L-BFGS-B",
        bounds=[log10_stiffness_bounds, log10_mass_bounds, log10_resistance_bounds],
    )

    best_stiffness = 10.0**optimisation_result.x[0]
    best_mass = 10.0**optimisation_result.x[1]
    best_resistance = 10.0**optimisation_result.x[2]

    p_end_air, p_end_filter, il_iec711_db = compute_il_iec711_db(
        freqs_hz=freqs,
        film_resistance=best_resistance,
        film_mass=best_mass,
        film_stiffness=best_stiffness,
        c0=C0,
        rho0=RHO0,
        length_inlet=length_inlet,
        length_outlet=length_outlet,
        r_inlet=r_inlet,
        r_outlet=r_outlet,
        r_duct=r_duct,
        length_duct=length_duct,
        p_incident=p_incident,
    )
    best_result = evaluate_target_metrics(
        freqs,
        il_iec711_db,
        target_il_db=target_il_db,
        band_hz=target_band_hz,
        tolerance_db=target_tolerance_db,
    )

    print(f"Target IL = {target_il_db:.2f} dB over {target_band_hz[0]:.0f}-{target_band_hz[1]:.0f} Hz")
    print(f"Optimization success = {optimisation_result.success}")
    print(f"Optimizer message = {optimisation_result.message}")
    print(f"Best film stiffness = {best_stiffness:.3e}")
    print(f"Best film mass = {best_mass:.3e}")
    print(f"Best film resistance = {best_resistance:.3e}")
    print(f"Median error [dB] = {best_result['median_error_db']:.3f}")
    print(f"Flatness error [dB] = {best_result['flatness_error_db']:.3f}")
    print(f"Bandwidth penalty [dB^2] = {best_result['bandwidth_error_db2']:.3f}")
    print(f"Final objective [-] = {best_result['objective']:.3f}")

    title_suffix = f"R={best_resistance:.2e}, M={best_mass:.2e}, K={best_stiffness:.2e}"
    plot_single_config_results(
        freqs,
        p_end_air,
        p_end_filter,
        il_iec711_db,
        title_suffix=title_suffix,
        target_il_db=target_il_db,
        band_hz=target_band_hz,
    )
