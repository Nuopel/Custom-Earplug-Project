"""Compare rigid-end pressure and IL: COMSOL fixed/free slab vs TMM."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

ROOT = HERE.parents[1]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from B1_comsol_validation_cases import COMSOL_RESULTS_DIR, load_comsol_point_cases
from toolkitsd.acoustmm import CylindricalDuct, ElasticSlab


POINT3_INDEX = 3


def level_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), np.finfo(float).tiny))


def rigid_end_tmm_end_states(
    frequencies_hz: np.ndarray,
    *,
    radius: float = 3.75e-3,
    slab_length: float = 6.6e-3,
    cavity_length: float = 6.4e-3,
    air_only_length: float = 13.0e-3,
    rho_air: float = 1.2,
    c_air: float = 343.0,
    p0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi * frequencies_hz
    z_rigid_like = np.full(omega.shape, 1.0e18 + 0.0j, dtype=np.complex128)
    area = np.pi * radius**2
    z_source = rho_air * c_air / area

    air_only = CylindricalDuct(radius=radius, length=air_only_length, c0=c_air, rho0=rho_air)
    slab = ElasticSlab(
        radius=radius,
        length=slab_length,
        rho=1500.0,
        young=2.9e6,
        poisson=0.49,
        loss_factor=0.20,
    )
    slab_plus_cavity = slab + CylindricalDuct(radius=radius, length=cavity_length, c0=c_air, rho0=rho_air)

    p_in_air = air_only.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)
    p_in_tmm = slab_plus_cavity.p_in_from_incident_wave(p0, z_rigid_like, z_source, omega)

    p_end_air = air_only.p_tm(p_in_air, z_rigid_like, omega)
    p_end_tmm = slab_plus_cavity.p_tm(p_in_tmm, z_rigid_like, omega)
    u_end_air = air_only.U_tm(p_in_air, z_rigid_like, omega)
    u_end_tmm = slab_plus_cavity.U_tm(p_in_tmm, z_rigid_like, omega)
    v_end_air = u_end_air / area
    v_end_tmm = u_end_tmm / area
    return p_end_air, p_end_tmm, v_end_air, v_end_tmm


def plot_end_pressure_and_il(
    frequencies_hz: np.ndarray,
    p_end_air_comsol: np.ndarray,
    p_end_free_comsol: np.ndarray,
    p_end_fixed_comsol: np.ndarray,
    p_end_air_tmm: np.ndarray,
    p_end_tmm: np.ndarray,
) -> None:
    il_free_comsol = level_db(p_end_air_comsol) - level_db(p_end_free_comsol)
    il_fixed_comsol = level_db(p_end_air_comsol) - level_db(p_end_fixed_comsol)
    il_tmm = level_db(p_end_air_tmm) - level_db(p_end_tmm)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)

    axes[0].semilogx(frequencies_hz, level_db(p_end_air_comsol), color="#111827", lw=2.0, label="COMSOL air-air")
    axes[0].semilogx(frequencies_hz, level_db(p_end_air_tmm), color="#2563eb", lw=2.0, ls="--", label="TMM air-air")
    axes[0].set_ylabel(r"$20 \log_{10}|p_{end}|$ [dB re 1 Pa]")
    axes[0].set_title("Rigid-end pressure: air-air")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")
    axes[0].set_ylim([-0,10])

    axes[1].semilogx(frequencies_hz, level_db(p_end_free_comsol), color="#16a34a", lw=2.0, label="COMSOL free slab")
    axes[1].semilogx(frequencies_hz, level_db(p_end_fixed_comsol), color="#ea580c", lw=2.0, label="COMSOL fixed slab")
    axes[1].semilogx(frequencies_hz, level_db(p_end_tmm), color="#2563eb", lw=2.0, ls="--", label="TMM slab+cavity")
    axes[1].set_ylabel(r"$20 \log_{10}|p_{end}|$ [dB re 1 Pa]")
    axes[1].set_title("Rigid-end pressure: slab cases")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_ylim([-50,30])

    axes[2].semilogx(frequencies_hz, il_free_comsol, color="#16a34a", lw=2.0, label="IL COMSOL free slab")
    axes[2].semilogx(frequencies_hz, il_fixed_comsol, color="#ea580c", lw=2.0, label="IL COMSOL fixed slab")
    axes[2].semilogx(frequencies_hz, il_tmm, color="#2563eb", lw=2.0, ls="--", label="IL TMM slab+cavity")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("IL [dB]")
    axes[2].set_title("Rigid-end insertion loss relative to air-air")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(loc="best")
    axes[2].set_ylim([-30,60])

    plt.show()


def plot_end_velocity(
    frequencies_hz: np.ndarray,
    v_end_air_comsol: np.ndarray,
    v_end_free_comsol: np.ndarray,
    v_end_fixed_comsol: np.ndarray,
    v_end_air_tmm: np.ndarray,
    v_end_tmm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    axes[0].semilogx(frequencies_hz, level_db(v_end_air_comsol), color="#111827", lw=2.0, label="COMSOL air-air")
    axes[0].semilogx(frequencies_hz, level_db(v_end_air_tmm), color="#2563eb", lw=2.0, ls="--", label="TMM air-air")
    axes[0].set_ylabel(r"$20 \log_{10}|v_{end}|$ [dB re 1 m/s]")
    axes[0].set_title("Rigid-end particle velocity: air-air")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].semilogx(frequencies_hz, level_db(v_end_free_comsol), color="#16a34a", lw=2.0, label="COMSOL free slab")
    axes[1].semilogx(frequencies_hz, level_db(v_end_fixed_comsol), color="#ea580c", lw=2.0, label="COMSOL fixed slab")
    axes[1].semilogx(frequencies_hz, level_db(v_end_tmm), color="#2563eb", lw=2.0, ls="--", label="TMM slab+cavity")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel(r"$20 \log_{10}|v_{end}|$ [dB re 1 m/s]")
    axes[1].set_title("Rigid-end particle velocity: slab cases")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="best")

    plt.show()


if __name__ == "__main__":
    air_freq, _, air_p_rigid, _, air_u_rigid = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_air_air_open.txt",
        COMSOL_RESULTS_DIR / "air_air_air_rigidend.txt",
    )
    free_freq, _, free_p_rigid, _, free_u_rigid = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_freeslab_air_open.txt",
        COMSOL_RESULTS_DIR / "air_freeslab_air_rigidend.txt",
    )
    fixed_freq, _, fixed_p_rigid, _, fixed_u_rigid = load_comsol_point_cases(
        COMSOL_RESULTS_DIR / "air_fixedslab_air_open.txt",
        COMSOL_RESULTS_DIR / "air_fixedslab_air_rigidend.txt",
    )

    if not np.allclose(air_freq, free_freq):
        raise ValueError("Frequency mismatch between air-air and air-freeslab-air cases")
    if not np.allclose(air_freq, fixed_freq):
        raise ValueError("Frequency mismatch between air-air and air-fixedslab-air cases")

    p_end_air_comsol = air_p_rigid[POINT3_INDEX]
    p_end_free_comsol = free_p_rigid[POINT3_INDEX]
    p_end_fixed_comsol = fixed_p_rigid[POINT3_INDEX]
    v_end_air_comsol = air_u_rigid[POINT3_INDEX]
    v_end_free_comsol = free_u_rigid[POINT3_INDEX]
    v_end_fixed_comsol = fixed_u_rigid[POINT3_INDEX]

    p_end_air_tmm, p_end_tmm, v_end_air_tmm, v_end_tmm = rigid_end_tmm_end_states(air_freq)

    il_free_comsol = level_db(p_end_air_comsol) - level_db(p_end_free_comsol)
    il_fixed_comsol = level_db(p_end_air_comsol) - level_db(p_end_fixed_comsol)
    il_tmm = level_db(p_end_air_tmm) - level_db(p_end_tmm)

    print("=== RIGID-END PRESSURE: COMSOL VS TMM ===")
    print("End point: COMSOL point 5 reordered to TMM point 3")
    print("TMM model: ElasticSlab + 6.4 mm cavity, compared against 13 mm air-only reference")
    print()
    print("f [Hz] | |p_air_comsol| [Pa] | |p_free_comsol| [Pa] | |p_fixed_comsol| [Pa] | |p_tmm| [Pa] | IL_free [dB] | IL_fixed [dB] | IL_tmm [dB]")

    for target_hz in (100.0, 300.0, 1000.0, 3000.0, 6000.0, 10000.0):
        idx = int(np.argmin(np.abs(air_freq - target_hz)))
        print(
            f"{air_freq[idx]:6.0f} | "
            f"{np.abs(p_end_air_comsol[idx]):17.6e} | "
            f"{np.abs(p_end_free_comsol[idx]):18.6e} | "
            f"{np.abs(p_end_fixed_comsol[idx]):19.6e} | "
            f"{np.abs(p_end_tmm[idx]):11.6e} | "
            f"{il_free_comsol[idx]:12.3f} | "
            f"{il_fixed_comsol[idx]:13.3f} | "
            f"{il_tmm[idx]:11.3f}"
        )

    print()
    print(f"Max |IL_tmm - IL_free_comsol| : {np.max(np.abs(il_tmm - il_free_comsol)):.3f} dB")
    print(f"Max |IL_tmm - IL_fixed_comsol|: {np.max(np.abs(il_tmm - il_fixed_comsol)):.3f} dB")
    print(f"Velocity possibly wrong because its very tiny quantities")

    plot_end_pressure_and_il(
        frequencies_hz=air_freq,
        p_end_air_comsol=p_end_air_comsol,
        p_end_free_comsol=p_end_free_comsol,
        p_end_fixed_comsol=p_end_fixed_comsol,
        p_end_air_tmm=p_end_air_tmm,
        p_end_tmm=p_end_tmm,
    )
    # plot_end_velocity(
    #     frequencies_hz=air_freq,
    #     v_end_air_comsol=v_end_air_comsol,
    #     v_end_free_comsol=v_end_free_comsol,
    #     v_end_fixed_comsol=v_end_fixed_comsol,
    #     v_end_air_tmm=v_end_air_tmm,
    #     v_end_tmm=v_end_tmm,
    # )
