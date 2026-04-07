import json
from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv, yv


# =============================================================================
# ARTICLE / REFERENCE DATA
# =============================================================================
def get_article_lpm_points() -> Tuple[np.ndarray, np.ndarray]:
    """
    Article-extracted LPM points (example placeholders from your script).
    """
    lpm_data = {
        100: 166.5,
        500: 154.9,
    }
    f_art = np.array(list(lpm_data.keys()), dtype=float)
    mag_art = np.array(list(lpm_data.values()), dtype=float)
    return f_art, mag_art


def get_iec_table() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    IEC 60318-4 Table 1 values, converted to dB re 1 Pa·s·m⁻³ using +120 dB.
    """
    offset = 120.0
    iec_f = np.array(
        [100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
         1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000,
         6300, 8000, 10000],
        dtype=float,
    )
    iec_lv = np.array(
        [44.8, 42.9, 40.8, 39.0, 37.0, 35.0, 33.0, 31.1,
         29.2, 27.2, 26.7, 26.4, 25.5, 24.2, 23.1, 22.0,
         21.1, 20.4, 20.5, 20.8, 23.1],
        dtype=float,
    ) + offset
    iec_tol = np.array(
        [0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.3, 0.6, 0.6,
         0.7, 0.7, 0.7, 0.8, 0.8, 0.9, 1.0, 1.2, 1.2, 1.7, 2.2],
        dtype=float,
    )
    return iec_f, iec_lv, iec_tol


# =============================================================================
# CONSTANTS / INPUTS
# =============================================================================
@dataclass(frozen=True)
class AirProperties:
    rho0: float = 1.20
    c0: float = 343.90
    mu: float = 1.82e-5
    gamma: float = 1.40
    Cp: float = 1.00e3
    lam: float = 24.80e-3


@dataclass(frozen=True)
class Geometry:
    # Main tube
    R0: float = 3.77e-3
    L1: float = 3.12e-3
    L3: float = 4.75e-3
    L5: float = 4.69e-3

    # HR1 (rectangular slit + annular cavity)
    a2: float = 2.53e-3
    b2: float = 2.35e-3
    h2: float = 0.16e-3
    r2: float = 6.30e-3
    R2: float = 9.01e-3
    d1: float = 1.91e-3

    # HR2 (annular slit + annular cavity)
    r4: float = 4.66e-3
    alpha_deg: float = 95.33
    h4: float = 0.05e-3
    R4: float = 9.01e-3
    d2: float = 1.40e-3


@dataclass(frozen=True)
class FrequencyGrid:
    fmin: float = 100.0
    fmax: float = 20000.0
    npts: int = 4000


@dataclass(frozen=True)
class ModelOptions:
    # cavity model used inside TM branches
    hr1_cavity_model: Literal["bessel", "lumped"] = "bessel"
    hr2_cavity_model: Literal["bessel", "lumped"] = "lumped"


# =============================================================================
# DERIVED DIMENSIONS
# =============================================================================
@dataclass(frozen=True)
class DerivedGeometry:
    L0: float
    S0: float
    alpha_r: float

    S_cav2: float
    V_cav2: float
    S_cav4: float
    V_cav4: float

    S_slit2: float
    r_mean4: float
    S_slit4: float

    dl2: float
    dl4_in: float
    dl4_out: float
    R0_in: float
    r4_out: float


def end_corr_slit(h: float, b: float) -> float:
    """
    Munjal Eq. A7 one-end correction for a slit.
    """
    beta = h / b
    eps = 1.0 + beta**2
    dl_h = (
        (1.0 / (3.0 * np.pi)) * (beta + (1.0 - eps**1.5) / beta**2)
        + (1.0 / np.pi)
        * (
            np.log(beta + np.sqrt(eps)) / beta
            + np.log((1.0 + np.sqrt(eps)) / beta)
        )
    )
    return dl_h * h


def derive_geometry(g: Geometry) -> DerivedGeometry:
    L0 = g.L1 + g.L3 + g.L5
    S0 = np.pi * g.R0**2
    alpha_r = np.deg2rad(g.alpha_deg)

    S_cav2 = np.pi * (g.R2**2 - g.r2**2)
    V_cav2 = S_cav2 * g.d1

    S_cav4 = np.pi * (g.R4**2 - g.r4**2)
    V_cav4 = S_cav4 * g.d2

    S_slit2 = g.b2 * g.h2

    r_mean4 = (g.R0 + g.r4) / 2.0
    S_slit4 = 3.0 * alpha_r * r_mean4 * g.h4

    dl2 = end_corr_slit(g.h2, g.b2)

    p_in = 3.0 * alpha_r * g.R0
    p_out = 3.0 * alpha_r * g.r4
    dl4_in = end_corr_slit(g.h4, p_in)
    dl4_out = end_corr_slit(g.h4, p_out)

    R0_in = g.R0 - dl4_in
    r4_out = g.r4 + dl4_out

    return DerivedGeometry(
        L0=L0,
        S0=S0,
        alpha_r=alpha_r,
        S_cav2=S_cav2,
        V_cav2=V_cav2,
        S_cav4=S_cav4,
        V_cav4=V_cav4,
        S_slit2=S_slit2,
        r_mean4=r_mean4,
        S_slit4=S_slit4,
        dl2=dl2,
        dl4_in=dl4_in,
        dl4_out=dl4_out,
        R0_in=R0_in,
        r4_out=r4_out,
    )


# =============================================================================
# FREQUENCY / WAVENUMBERS
# =============================================================================
@dataclass(frozen=True)
class FrequencyState:
    f: np.ndarray
    omega: np.ndarray
    k0: np.ndarray
    Z1: float
    kh: np.ndarray
    kv: np.ndarray


def build_frequency_state(
    air: AirProperties,
    dg: DerivedGeometry,
    freq_cfg: FrequencyGrid,
) -> FrequencyState:
    f = np.logspace(np.log10(freq_cfg.fmin), np.log10(freq_cfg.fmax), freq_cfg.npts)
    omega = 2.0 * np.pi * f
    k0 = omega / air.c0
    Z1 = air.rho0 * air.c0 / dg.S0

    lh = air.lam / (air.rho0 * air.c0 * air.Cp)
    lv = air.mu / (air.rho0 * air.c0)

    kh = (1.0 - 1.0j) / np.sqrt(2.0) * np.sqrt(k0 / lh)
    kv = (1.0 - 1.0j) / np.sqrt(2.0) * np.sqrt(k0 / lv)

    return FrequencyState(f=f, omega=omega, k0=k0, Z1=Z1, kh=kh, kv=kv)


# =============================================================================
# COMMON ACOUSTIC HELPERS
# =============================================================================
def kfield(k_bnd: np.ndarray, h: float) -> np.ndarray:
    arg = k_bnd * h / 2.0
    return 1.0 - np.tan(arg) / arg


def lrf_kZ(
    air: AirProperties,
    fs: FrequencyState,
    h: float,
    S: float,
) -> Tuple[np.ndarray, np.ndarray]:
    Kh = kfield(fs.kh, h)
    Kv = kfield(fs.kv, h)
    Khp = air.gamma - (air.gamma - 1.0) * Kh
    kl = fs.k0 * np.sqrt(Khp / Kv)
    Zl = (air.rho0 * air.c0 / S) / np.sqrt(Khp * Kv)
    return kl, Zl


def lumped_cavity_compliance(V: float, air: AirProperties) -> float:
    return V / (air.rho0 * air.c0**2)


def cylindrical_annular_cavity_impedance(
    k0: np.ndarray,
    r_inner: float,
    r_outer: float,
    S_cav: float,
    air: AirProperties,
) -> np.ndarray:
    """
    Bessel-based annular cavity impedance.
    """
    Bs = yv(1, k0 * r_outer) / jv(1, k0 * r_outer)
    return (
        1j
        * air.rho0
        * air.c0
        * (Bs * jv(0, k0 * r_inner) - yv(0, k0 * r_inner))
        / (S_cav * (Bs * jv(1, k0 * r_inner) - yv(1, k0 * r_inner)))
    )


# =============================================================================
# HR BRANCHES FOR TM
# =============================================================================
def hr1_tm_impedance(
    air: AirProperties,
    g: Geometry,
    dg: DerivedGeometry,
    fs: FrequencyState,
    cavity_model: Literal["bessel", "lumped"] = "bessel",
) -> np.ndarray:
    """
    HR1: rectangular slit + cavity
    """
    kl2, Zl2 = lrf_kZ(air, fs, g.h2, dg.S_slit2)
    Z_slit2 = 1j * Zl2 * np.tan(kl2 * (g.a2 + 2.0 * dg.dl2))

    if cavity_model == "bessel":
        Z_cav2 = cylindrical_annular_cavity_impedance(
            k0=fs.k0,
            r_inner=g.r2,
            r_outer=g.R2,
            S_cav=dg.S_cav2,
            air=air,
        )
    elif cavity_model == "lumped":
        C_cav2 = lumped_cavity_compliance(dg.V_cav2, air)
        Z_cav2 = -1j / (fs.omega * C_cav2)
    else:
        raise ValueError(f"Unknown cavity_model={cavity_model!r}")

    return Z_slit2 + Z_cav2


def hr2_tm_impedance(
    air: AirProperties,
    g: Geometry,
    dg: DerivedGeometry,
    fs: FrequencyState,
    cavity_model: Literal["bessel", "lumped"] = "lumped",
) -> np.ndarray:
    """
    HR2: annular slit + cavity
    """
    kl4, Zl4 = lrf_kZ(air, fs, g.h4, dg.S_slit4)

    As = yv(0, kl4 * dg.r4_out) / jv(0, kl4 * dg.r4_out)
    Z_slit4 = (
        1j
        * Zl4
        * (As * jv(0, kl4 * dg.R0_in) - yv(0, kl4 * dg.R0_in))
        / (As * jv(1, kl4 * dg.R0_in) - yv(1, kl4 * dg.R0_in))
    )

    if cavity_model == "bessel":
        Z_cav4 = cylindrical_annular_cavity_impedance(
            k0=fs.k0,
            r_inner=g.r4,
            r_outer=g.R4,
            S_cav=dg.S_cav4,
            air=air,
        )
    elif cavity_model == "lumped":
        C_cav4 = lumped_cavity_compliance(dg.V_cav4, air)
        Z_cav4 = -1j / (fs.omega * C_cav4)
    else:
        raise ValueError(f"Unknown cavity_model={cavity_model!r}")

    return Z_slit4 + Z_cav4


# =============================================================================
# TRANSFER MATRIX
# =============================================================================
def mm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("ijk,jlk->ilk", A, B)


def tm_tube(k0: np.ndarray, Z1: float, L: float) -> np.ndarray:
    n = len(k0)
    T = np.zeros((2, 2, n), dtype=complex)
    c = np.cos(k0 * L)
    s = np.sin(k0 * L)
    T[0, 0] = c
    T[0, 1] = 1j * Z1 * s
    T[1, 0] = 1j * s / Z1
    T[1, 1] = c
    return T


def tm_shunt(Zhr: np.ndarray) -> np.ndarray:
    n = len(Zhr)
    T = np.zeros((2, 2, n), dtype=complex)
    T[0, 0] = 1.0
    T[1, 1] = 1.0
    T[1, 0] = 1.0 / Zhr
    return T


def compute_tm_response(
    air: AirProperties,
    g: Geometry,
    dg: DerivedGeometry,
    fs: FrequencyState,
    opts: ModelOptions,
) -> Dict[str, np.ndarray]:
    Z_HR1_tm = hr1_tm_impedance(
        air=air,
        g=g,
        dg=dg,
        fs=fs,
        cavity_model=opts.hr1_cavity_model,
    )
    Z_HR2_tm = hr2_tm_impedance(
        air=air,
        g=g,
        dg=dg,
        fs=fs,
        cavity_model=opts.hr2_cavity_model,
    )

    T_ES = mm(
        tm_tube(fs.k0, fs.Z1, g.L1),
        mm(
            tm_shunt(Z_HR1_tm),
            mm(
                tm_tube(fs.k0, fs.Z1, g.L3),
                mm(
                    tm_shunt(Z_HR2_tm),
                    tm_tube(fs.k0, fs.Z1, g.L5),
                ),
            ),
        ),
    )

    Rs = (T_ES[0, 0] - T_ES[1, 0] * fs.Z1) / (T_ES[0, 0] + T_ES[1, 0] * fs.Z1)
    Zs_tm = fs.Z1 * (1.0 + Rs) / (1.0 - Rs)

    mag_tm = 20.0 * np.log10(np.abs(Zs_tm))
    phi_tm = np.angle(Zs_tm)

    fr1_tm = fs.f[np.argmin(np.abs(Z_HR1_tm))]
    fr2_tm = fs.f[np.argmin(np.abs(Z_HR2_tm))]

    return {
        "Z_HR1_tm": Z_HR1_tm,
        "Z_HR2_tm": Z_HR2_tm,
        "Zs_tm": Zs_tm,
        "mag_tm": mag_tm,
        "phi_tm": phi_tm,
        "fr1_tm": fr1_tm,
        "fr2_tm": fr2_tm,
    }


# =============================================================================
# LPM
# =============================================================================
def cav_lpm(L: float, S0: float, air: AirProperties) -> Tuple[float, float]:
    """
    Return (mass, compliance) for main-tube sections in the LPM chain.
    """
    m = air.rho0 * L / S0
    c = S0 * L / (air.rho0 * air.c0**2)
    return m, c


def compute_lpm_hr_parameters(
    air: AirProperties,
    g: Geometry,
    dg: DerivedGeometry,
) -> Dict[str, float]:
    # HR1
    r_a2 = 12.0 * air.mu * g.a2 / (g.b2 * g.h2**3)
    m_a2 = 6.0 * air.rho0 * g.a2 / (5.0 * dg.S_slit2)
    c_a2 = dg.V_cav2 / (air.rho0 * air.c0**2)

    # HR2
    a4_ = g.r4 - g.R0
    b4_ = 3.0 * dg.alpha_r * (g.R0 + g.r4) / 2.0
    S4_ = b4_ * g.h4

    r_a4 = 12.0 * air.mu * a4_ / (b4_ * g.h4**3)
    m_a4 = 6.0 * air.rho0 * a4_ / (5.0 * S4_)
    c_a4 = dg.V_cav4 / (air.rho0 * air.c0**2)

    return {
        "r_a2": r_a2,
        "m_a2": m_a2,
        "c_a2": c_a2,
        "r_a4": r_a4,
        "m_a4": m_a4,
        "c_a4": c_a4,
    }


def compute_lpm_response(
    air: AirProperties,
    g: Geometry,
    dg: DerivedGeometry,
    fs: FrequencyState,
) -> Dict[str, np.ndarray]:
    m1, c1 = cav_lpm(g.L1, dg.S0, air)
    m3, c3 = cav_lpm(g.L3, dg.S0, air)
    m5, c5 = cav_lpm(g.L5, dg.S0, air)

    pars = compute_lpm_hr_parameters(air, g, dg)

    Z_HR1_lpm = (
        pars["r_a2"]
        + 1j * fs.omega * pars["m_a2"]
        + 1.0 / (1j * fs.omega * pars["c_a2"])
    )
    Z_HR2_lpm = (
        pars["r_a4"]
        + 1j * fs.omega * pars["m_a4"]
        + 1.0 / (1j * fs.omega * pars["c_a4"])
    )

    Z = 1.0 / (1j * fs.omega * c5)
    Z = Z + 1j * fs.omega * m5
    Z = 1.0 / (1.0 / Z + 1j * fs.omega * c3 + 1.0 / Z_HR2_lpm)
    Z = Z + 1j * fs.omega * m3
    Z = 1.0 / (1.0 / Z + 1j * fs.omega * c1 + 1.0 / Z_HR1_lpm)
    Zs_lpm = Z + 1j * fs.omega * m1

    mag_lpm = 20.0 * np.log10(np.abs(Zs_lpm))
    phi_lpm = np.angle(Zs_lpm)

    fr1_lpm = 1.0 / (2.0 * np.pi * np.sqrt(pars["m_a2"] * pars["c_a2"]))
    fr2_lpm = 1.0 / (2.0 * np.pi * np.sqrt(pars["m_a4"] * pars["c_a4"]))

    return {
        "Z_HR1_lpm": Z_HR1_lpm,
        "Z_HR2_lpm": Z_HR2_lpm,
        "Zs_lpm": Zs_lpm,
        "mag_lpm": mag_lpm,
        "phi_lpm": phi_lpm,
        "fr1_lpm": fr1_lpm,
        "fr2_lpm": fr2_lpm,
    }


# =============================================================================
# COMPARISON / METRICS
# =============================================================================
def compare_to_iec(
    f: np.ndarray,
    mag_tm: np.ndarray,
    mag_lpm: np.ndarray,
) -> Dict[str, np.ndarray]:
    iec_f, iec_lv, iec_tol = get_iec_table()

    tm_at_iec = np.interp(np.log10(iec_f), np.log10(f), mag_tm)
    lpm_at_iec = np.interp(np.log10(iec_f), np.log10(f), mag_lpm)

    res_tm = tm_at_iec - iec_lv
    res_lpm = lpm_at_iec - iec_lv

    pass_tm = np.abs(res_tm) <= iec_tol
    pass_lpm = np.abs(res_lpm) <= iec_tol

    return {
        "iec_f": iec_f,
        "iec_lv": iec_lv,
        "iec_tol": iec_tol,
        "tm_at_iec": tm_at_iec,
        "lpm_at_iec": lpm_at_iec,
        "res_tm": res_tm,
        "res_lpm": res_lpm,
        "pass_tm": pass_tm,
        "pass_lpm": pass_lpm,
        "rms_tm": np.sqrt(np.mean(res_tm**2)),
        "rms_lpm": np.sqrt(np.mean(res_lpm**2)),
    }


def print_geometry_summary(g: Geometry, dg: DerivedGeometry) -> None:
    print("=== GEOMETRY / END CORRECTIONS ===")
    print(f"Δl2     = {dg.dl2*1e6:.1f} μm  -> eff. length = {(g.a2+2*dg.dl2)*1e3:.3f} mm")
    print(f"Δl4_in  = {dg.dl4_in*1e6:.1f} μm -> R0_in  = {dg.R0_in*1e3:.4f} mm")
    print(f"Δl4_out = {dg.dl4_out*1e6:.1f} μm -> r4_out = {dg.r4_out*1e3:.4f} mm")
    print()


def print_model_summary(tm: Dict[str, np.ndarray], lpm: Dict[str, np.ndarray], cmp: Dict[str, np.ndarray]) -> None:
    n_tm = int(np.sum(cmp["pass_tm"]))
    n_lpm = int(np.sum(cmp["pass_lpm"]))
    n_tot = len(cmp["iec_f"])

    print("=== RESONANCES ===")
    print(f"TM : HR1={tm['fr1_tm']:.0f} Hz  HR2={tm['fr2_tm']:.0f} Hz")
    print(f"LPM: HR1={lpm['fr1_lpm']:.0f} Hz  HR2={lpm['fr2_lpm']:.0f} Hz")
    print()

    print("=== IEC FIT ===")
    print(f"TM :  pass {n_tm}/{n_tot} pts, RMS={cmp['rms_tm']:.2f} dB")
    print(f"LPM:  pass {n_lpm}/{n_tot} pts, RMS={cmp['rms_lpm']:.2f} dB")
    print()


# =============================================================================
# PLOTS
# =============================================================================
def _style_axes(ax, xticks, xlabels) -> None:
    ax.grid(True, which="major", ls="-", alpha=0.3, lw=0.8, color="#888")
    ax.grid(True, which="minor", ls="--", alpha=0.15, lw=0.5, color="#aaa")
    ax.set_xlim(100, 20000)
    ax.tick_params(labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)


def plot_results(
    fs: FrequencyState,
    tm: Dict[str, np.ndarray],
    lpm: Dict[str, np.ndarray],
    cmp: Dict[str, np.ndarray],
    savepath: str = "IEC60318_4_TM_LRF_refactored.png",
) -> None:
    f_art, mag_lpm_art = get_article_lpm_points()

    C_TM = "#2a6eb5"
    C_LPM = "#e74c3c"
    C_IEC = "#e67e22"
    C_HR1 = "#27ae60"
    C_HR2 = "#8e44ad"

    xticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    xlabels = ["100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

    ylo = min(tm["mag_tm"].min(), cmp["iec_lv"].min() - 5) - 2
    yhi = max(tm["mag_tm"].max(), cmp["iec_lv"].max() + 3) + 2

    fig = plt.figure(figsize=(13, 15))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[3, 2.5, 2.5, 1.5], hspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    fig.suptitle(
        "IEC 60318-4 — TM vs LPM vs IEC Table 1\n"
        "Refactored program — dB re 1 Pa·s·m⁻³",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    # -------------------------------------------------------------------------
    # P1: magnitude
    # -------------------------------------------------------------------------
    ax = ax0
    ax.set_title("Impedance modulus", fontsize=11, pad=6)
    ax.fill_between(
        cmp["iec_f"],
        cmp["iec_lv"] - cmp["iec_tol"],
        cmp["iec_lv"] + cmp["iec_tol"],
        color=C_IEC,
        alpha=0.20,
        label="IEC tolerance band",
    )
    ax.semilogx(fs.f, lpm["mag_lpm"], color=C_LPM, lw=1.8, ls="--", label="LPM", alpha=0.8, zorder=2)
    ax.semilogx(fs.f, tm["mag_tm"], color=C_TM, lw=2.2, label="TM", zorder=3)
    ax.errorbar(
        cmp["iec_f"],
        cmp["iec_lv"],
        yerr=cmp["iec_tol"],
        fmt="s",
        color=C_IEC,
        ms=4,
        capsize=3,
        lw=1.1,
        label="IEC Table 1 (+120 dB)",
        zorder=4,
    )
    ax.semilogx(f_art, mag_lpm_art, color="k", lw=1.8, ls="--", label="Article LPM pts", alpha=0.8, zorder=2)

    for fr, col, lbl in [
        (tm["fr1_tm"], C_HR1, f"HR1 {tm['fr1_tm']:.0f}Hz"),
        (tm["fr2_tm"], C_HR2, f"HR2 {tm['fr2_tm']:.0f}Hz"),
    ]:
        ax.axvline(fr, color=col, ls=":", lw=1.4, alpha=0.85)
        ax.text(fr * 1.07, ylo + 4, lbl, color=col, fontsize=8.5, rotation=90, va="bottom")

    ax.set_ylabel("|Zs|  (dB re 1 Pa·s·m⁻³)", fontsize=11)
    ax.set_ylim(100, 200)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    ax.annotate(
        "IEC Table 1: transfer impedance re 1 MPa·s·m⁻³ -> +120 dB applied",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=8,
        color="#666",
        style="italic",
    )
    _style_axes(ax, xticks, xlabels)

    # -------------------------------------------------------------------------
    # P2: phase
    # -------------------------------------------------------------------------
    ax = ax1
    ax.set_title("Impedance phase", fontsize=11, pad=6)
    ax.semilogx(fs.f, lpm["phi_lpm"], color=C_LPM, lw=1.8, ls="--", label="LPM", alpha=0.8)
    ax.semilogx(fs.f, tm["phi_tm"], color=C_TM, lw=2.2, label="TM")
    ax.axhline(0, color="#555", lw=0.9, ls=":")
    for fr, col in [(tm["fr1_tm"], C_HR1), (tm["fr2_tm"], C_HR2)]:
        ax.axvline(fr, color=col, ls=":", lw=1.4, alpha=0.85)
    ax.set_ylabel("Phase (rad)", fontsize=11)
    ax.set_ylim(-np.pi / 2 - 0.2, np.pi / 2 + 0.2)
    ax.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    ax.set_yticklabels(["-π/2", "-π/4", "0", "π/4", "π/2"])
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)
    _style_axes(ax, xticks, xlabels)

    # -------------------------------------------------------------------------
    # P3: scatter vs IEC
    # -------------------------------------------------------------------------
    ax = ax2
    ax.set_title("Model vs IEC Table 1 at specification frequencies", fontsize=11, pad=6)
    ax.fill_between(
        cmp["iec_f"],
        cmp["iec_lv"] - cmp["iec_tol"],
        cmp["iec_lv"] + cmp["iec_tol"],
        color=C_IEC,
        alpha=0.20,
    )
    ax.scatter(cmp["iec_f"], cmp["iec_lv"], marker="s", color=C_IEC, s=55, zorder=4, edgecolors="w", lw=0.5, label="IEC target")
    ax.scatter(cmp["iec_f"], cmp["tm_at_iec"], marker="o", color=C_TM, s=55, zorder=5, edgecolors="w", lw=0.5, label="TM")
    ax.scatter(cmp["iec_f"], cmp["lpm_at_iec"], marker="^", color=C_LPM, s=50, zorder=4, edgecolors="w", lw=0.5, label="LPM", alpha=0.7)

    for fi, li, ti, li2 in zip(cmp["iec_f"], cmp["iec_lv"], cmp["tm_at_iec"], cmp["lpm_at_iec"]):
        ax.plot([fi, fi], [li, ti], color=C_TM, lw=0.8, alpha=0.5)
        ax.plot([fi, fi], [li, li2], color=C_LPM, lw=0.8, alpha=0.4, ls="--")

    ax.set_xscale("log")
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("|Zs|  (dB re 1 Pa·s·m⁻³)", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    _style_axes(ax, xticks, xlabels)

    # -------------------------------------------------------------------------
    # P4: residuals
    # -------------------------------------------------------------------------
    n_tm = int(np.sum(cmp["pass_tm"]))
    n_lpm = int(np.sum(cmp["pass_lpm"]))

    ax = ax3
    ax.set_title(
        f"Residual = model − IEC    "
        f"TM: {n_tm}/{len(cmp['iec_f'])} in tol, RMS={cmp['rms_tm']:.2f} dB    "
        f"LPM: {n_lpm}/{len(cmp['iec_f'])} in tol, RMS={cmp['rms_lpm']:.2f} dB",
        fontsize=10,
        pad=6,
    )
    ax.axhline(0, color="#555", lw=0.9, ls="--")
    ax.axhspan(-1, 1, color=C_IEC, alpha=0.08)

    col_tm = [C_TM if p else "#e74c3c" for p in cmp["pass_tm"]]
    col_lpm = [C_LPM if p else "#e74c3c" for p in cmp["pass_lpm"]]

    ax.scatter(cmp["iec_f"], cmp["res_tm"], c=col_tm, marker="o", s=65, zorder=5, edgecolors="w", lw=0.5, label="TM")
    ax.scatter(cmp["iec_f"], cmp["res_lpm"], c=col_lpm, marker="^", s=55, zorder=4, edgecolors="w", lw=0.5, label="LPM", alpha=0.8)

    for fi, tol in zip(cmp["iec_f"], cmp["iec_tol"]):
        ax.fill_between([fi * 0.82, fi * 1.22], [-tol, -tol], [tol, tol], color=C_IEC, alpha=0.10)

    ax.set_xscale("log")
    ax.set_ylim(-6, 6)
    ax.set_ylabel("Residual (dB)", fontsize=11)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    _style_axes(ax, xticks, xlabels)

    # plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

    with open(savepath + ".meta.json", "w") as fj:
        json.dump(
            {
                "caption": "IEC 60318-4: TM vs LPM vs IEC Table 1",
                "description": "Four-panel comparison of TM model, LPM, and IEC Table 1 reference.",
            },
            fj,
        )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    air = AirProperties()
    g = Geometry()
    freq_cfg = FrequencyGrid()
    opts = ModelOptions(
        hr1_cavity_model="bessel",
        hr2_cavity_model="lumped",   # keep explicit; change to "bessel" if wanted
    )

    dg = derive_geometry(g)
    fs = build_frequency_state(air, dg, freq_cfg)

    print_geometry_summary(g, dg)

    tm = compute_tm_response(air, g, dg, fs, opts)
    lpm = compute_lpm_response(air, g, dg, fs)
    cmp = compare_to_iec(fs.f, tm["mag_tm"], lpm["mag_lpm"])

    print_model_summary(tm, lpm, cmp)

    plot_results(fs, tm, lpm, cmp, savepath="IEC60318_4_TM_LRF_refactored.png")


if __name__ == "__main__":
    main()