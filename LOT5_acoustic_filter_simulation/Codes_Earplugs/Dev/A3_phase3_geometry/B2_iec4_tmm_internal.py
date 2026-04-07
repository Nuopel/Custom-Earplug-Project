"""IEC 60318-4 comparison using the internal acoustmm coupler models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
candidate_paths = [ROOT / "src"]
candidate_paths.extend(sorted(ROOT.parent.glob("Toolkitsd_*/src")))
for candidate in candidate_paths:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from toolkitsd.acoustmm import IEC711Coupler


def get_iec_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IEC 60318-4 Table 1 values in dB re 1 Pa.s.m^-3."""
    offset = 120.0
    iec_f = np.array(
        [
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
        ],
        dtype=float,
    )
    iec_lv = np.array(
        [
            44.8,
            42.9,
            40.8,
            39.0,
            37.0,
            35.0,
            33.0,
            31.1,
            29.2,
            27.2,
            26.7,
            26.4,
            25.5,
            24.2,
            23.1,
            22.0,
            21.1,
            20.4,
            20.5,
            20.8,
            23.1,
        ],
        dtype=float,
    ) + offset
    iec_tol = np.array(
        [
            0.7,
            0.7,
            0.7,
            0.6,
            0.6,
            0.6,
            0.6,
            0.3,
            0.6,
            0.6,
            0.7,
            0.7,
            0.7,
            0.8,
            0.8,
            0.9,
            1.0,
            1.2,
            1.2,
            1.7,
            2.2,
        ],
        dtype=float,
    )
    return iec_f, iec_lv, iec_tol


def compare_to_iec(f: np.ndarray, mag_tm: np.ndarray, mag_lumped: np.ndarray) -> dict[str, np.ndarray]:
    iec_f, iec_lv, iec_tol = get_iec_table()
    tm_at_iec = np.interp(np.log10(iec_f), np.log10(f), mag_tm)
    lumped_at_iec = np.interp(np.log10(iec_f), np.log10(f), mag_lumped)
    res_tm = tm_at_iec - iec_lv
    res_lumped = lumped_at_iec - iec_lv
    pass_tm = np.abs(res_tm) <= iec_tol
    pass_lumped = np.abs(res_lumped) <= iec_tol
    return {
        "iec_f": iec_f,
        "iec_lv": iec_lv,
        "iec_tol": iec_tol,
        "tm_at_iec": tm_at_iec,
        "lumped_at_iec": lumped_at_iec,
        "res_tm": res_tm,
        "res_lumped": res_lumped,
        "pass_tm": pass_tm,
        "pass_lumped": pass_lumped,
        "rms_tm": np.sqrt(np.mean(res_tm**2)),
        "rms_lumped": np.sqrt(np.mean(res_lumped**2)),
    }


def print_model_summary(
    tm_res: dict[str, float],
    lumped_res: dict[str, float],
    cmp: dict[str, np.ndarray],
) -> None:
    n_tm = int(np.sum(cmp["pass_tm"]))
    n_lumped = int(np.sum(cmp["pass_lumped"]))
    n_tot = len(cmp["iec_f"])

    print("=== RESONANCES ===")
    print(f"TM     : HR1={tm_res['hr1']:.0f} Hz  HR2={tm_res['hr2']:.0f} Hz")
    print(f"LUMPED : HR1={lumped_res['hr1']:.0f} Hz  HR2={lumped_res['hr2']:.0f} Hz")
    print()
    print("=== IEC FIT ===")
    print(f"TM     : pass {n_tm}/{n_tot} pts, RMS={cmp['rms_tm']:.2f} dB")
    print(f"LUMPED : pass {n_lumped}/{n_tot} pts, RMS={cmp['rms_lumped']:.2f} dB")
    print()


def _style_axes(ax: plt.Axes, xticks: list[int], xlabels: list[str]) -> None:
    ax.grid(True, which="major", ls="-", alpha=0.3, lw=0.8, color="#888")
    ax.grid(True, which="minor", ls="--", alpha=0.15, lw=0.5, color="#aaa")
    ax.set_xlim(100, 20000)
    ax.tick_params(labelsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)


def plot_results(
    f: np.ndarray,
    z_tm: np.ndarray,
    z_lumped: np.ndarray,
    tm_res: dict[str, float],
    cmp: dict[str, np.ndarray],
    savepath: str = "IEC60318_4_TM_internal.png",
) -> None:
    mag_tm = 20.0 * np.log10(np.abs(z_tm))
    mag_lumped = 20.0 * np.log10(np.abs(z_lumped))
    phi_tm = np.angle(z_tm)
    phi_lumped = np.angle(z_lumped)

    c_tm = "#2a6eb5"
    c_lumped = "#e74c3c"
    c_iec = "#e67e22"
    c_hr1 = "#27ae60"
    c_hr2 = "#8e44ad"

    xticks = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    xlabels = ["100", "200", "500", "1k", "2k", "5k", "10k", "20k"]

    fig = plt.figure(figsize=(13, 15))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[3, 2.5, 2.5, 1.5], hspace=0.45)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    fig.suptitle(
        "IEC 60318-4 from toolbox internals\nTM vs lumped vs IEC Table 1",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    ax0.set_title("Impedance modulus", fontsize=11, pad=6)
    ax0.fill_between(
        cmp["iec_f"],
        cmp["iec_lv"] - cmp["iec_tol"],
        cmp["iec_lv"] + cmp["iec_tol"],
        color=c_iec,
        alpha=0.20,
        label="IEC tolerance band",
    )
    ax0.semilogx(f, mag_lumped, color=c_lumped, lw=1.8, ls="--", label="Internal lumped", alpha=0.85)
    ax0.semilogx(f, mag_tm, color=c_tm, lw=2.2, label="Internal TMM")
    ax0.errorbar(
        cmp["iec_f"],
        cmp["iec_lv"],
        yerr=cmp["iec_tol"],
        fmt="s",
        color=c_iec,
        ms=4,
        capsize=3,
        lw=1.1,
        label="IEC Table 1 (+120 dB)",
    )
    for fr, col, lbl in [
        (tm_res["hr1"], c_hr1, f"HR1 {tm_res['hr1']:.0f}Hz"),
        (tm_res["hr2"], c_hr2, f"HR2 {tm_res['hr2']:.0f}Hz"),
    ]:
        ax0.axvline(fr, color=col, ls=":", lw=1.4, alpha=0.85)
        ax0.text(fr * 1.07, 104, lbl, color=col, fontsize=8.5, rotation=90, va="bottom")
    ax0.set_ylabel("|Zs| (dB re 1 Pa.s.m^-3)", fontsize=11)
    ax0.set_ylim(100, 200)
    ax0.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    _style_axes(ax0, xticks, xlabels)

    ax1.set_title("Impedance phase", fontsize=11, pad=6)
    ax1.semilogx(f, phi_lumped, color=c_lumped, lw=1.8, ls="--", label="Internal lumped", alpha=0.85)
    ax1.semilogx(f, phi_tm, color=c_tm, lw=2.2, label="Internal TMM")
    ax1.axhline(0, color="#555", lw=0.9, ls=":")
    for fr, col in [(tm_res["hr1"], c_hr1), (tm_res["hr2"], c_hr2)]:
        ax1.axvline(fr, color=col, ls=":", lw=1.4, alpha=0.85)
    ax1.set_ylabel("Phase (rad)", fontsize=11)
    ax1.set_ylim(-np.pi / 2 - 0.2, np.pi / 2 + 0.2)
    ax1.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    ax1.set_yticklabels(["-pi/2", "-pi/4", "0", "pi/4", "pi/2"])
    ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
    _style_axes(ax1, xticks, xlabels)

    ax2.set_title("Model vs IEC Table 1 at specification frequencies", fontsize=11, pad=6)
    ax2.fill_between(
        cmp["iec_f"],
        cmp["iec_lv"] - cmp["iec_tol"],
        cmp["iec_lv"] + cmp["iec_tol"],
        color=c_iec,
        alpha=0.20,
    )
    ax2.scatter(cmp["iec_f"], cmp["iec_lv"], marker="s", color=c_iec, s=55, edgecolors="w", lw=0.5, label="IEC target")
    ax2.scatter(cmp["iec_f"], cmp["tm_at_iec"], marker="o", color=c_tm, s=55, edgecolors="w", lw=0.5, label="Internal TMM")
    ax2.scatter(
        cmp["iec_f"],
        cmp["lumped_at_iec"],
        marker="^",
        color=c_lumped,
        s=50,
        edgecolors="w",
        lw=0.5,
        label="Internal lumped",
        alpha=0.85,
    )
    ax2.set_xscale("log")
    ax2.set_ylim(100, 200)
    ax2.set_ylabel("|Zs| (dB re 1 Pa.s.m^-3)", fontsize=11)
    ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
    _style_axes(ax2, xticks, xlabels)

    n_tm = int(np.sum(cmp["pass_tm"]))
    n_lumped = int(np.sum(cmp["pass_lumped"]))
    ax3.set_title(
        f"Residual = model - IEC    TM: {n_tm}/{len(cmp['iec_f'])} in tol, RMS={cmp['rms_tm']:.2f} dB    "
        f"LUMPED: {n_lumped}/{len(cmp['iec_f'])} in tol, RMS={cmp['rms_lumped']:.2f} dB",
        fontsize=10,
        pad=6,
    )
    ax3.axhline(0, color="#555", lw=0.9, ls="--")
    ax3.scatter(cmp["iec_f"], cmp["res_tm"], color=c_tm, marker="o", s=65, edgecolors="w", lw=0.5, label="Internal TMM")
    ax3.scatter(
        cmp["iec_f"],
        cmp["res_lumped"],
        color=c_lumped,
        marker="^",
        s=55,
        edgecolors="w",
        lw=0.5,
        label="Internal lumped",
        alpha=0.85,
    )
    ax3.set_xscale("log")
    ax3.set_ylim(-6, 6)
    ax3.set_ylabel("Residual (dB)", fontsize=11)
    ax3.set_xlabel("Frequency (Hz)", fontsize=11)
    ax3.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    _style_axes(ax3, xticks, xlabels)

    # plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

    # with open(savepath + ".meta.json", "w", encoding="utf-8") as fj:
    #     json.dump(
    #         {
    #             "caption": "IEC 60318-4 with internal acoustmm TMM and lumped models",
    #             "description": "Four-panel comparison of internal TMM and lumped coupler models against IEC Table 1.",
    #         },
    #         fj,
    #     )



if __name__ == "__main__":
    f = np.logspace(np.log10(100.0), np.log10(20000.0), 4000)
    omega = 2.0 * np.pi * f

    tm_model = IEC711Coupler(model="tmm")
    lumped_model = IEC711Coupler(model="lumped")

    z_tm = tm_model.Z(omega)
    z_lumped = lumped_model.Z(omega)
    mag_tm = 20.0 * np.log10(np.abs(z_tm))
    mag_lumped = 20.0 * np.log10(np.abs(z_lumped))

    tm_res = tm_model.branch_resonance_frequencies(omega)
    lumped_res = lumped_model.branch_resonance_frequencies()
    cmp = compare_to_iec(f, mag_tm, mag_lumped)

    print_model_summary(tm_res, lumped_res, cmp)
    plot_results(f, z_tm, z_lumped, tm_res, cmp)
