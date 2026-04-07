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

from toolkitsd.acoustmm import ElasticSlab, ElasticSlabSeries, ElasticSlabThin


rho0, c0 = 1.21, 343.0
f = np.linspace(50, 6500, 2000)
omega = 2 * np.pi * f

# Silicone material (Table I)
params = {
    "radius": 7.5e-3,
    "length": 6.0e-3,
    "rho": 1500.0,
    "young": 2.9e6,
    "poisson": 0.49,
    "loss_factor": 0.20,
}
area = np.pi * params["radius"] ** 2
z_air = rho0 * c0 / area

models = {
    "full": ElasticSlab(**params),
    "thinplate": ElasticSlabThin(**params),
    "series": ElasticSlabSeries(**params),
}
results = {name: model.TL(z_air, omega) for name, model in models.items()}

# Mass law reference in the same physical limit.
m_s = params["rho"] * params["length"]
tl_mass_law = 20 * np.log10(omega * m_s / (2 * rho0 * c0))


fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(f, results["full"], color="#2C3E50", lw=2.5, label="Full propagator  (exact 1D)")
ax.semilogx(
    f,
    results["thinplate"],
    color="#E74C3C",
    lw=2,
    linestyle="--",
    label="Thin plate  (mass + compliance)",
)
ax.semilogx(
    f,
    results["series"],
    color="#2980B9",
    lw=2,
    linestyle="-.",
    label="Series Z  (mass only)",
)
ax.semilogx(
    f,
    tl_mass_law,
    color="#27AE60",
    lw=1.5,
    linestyle=":",
    label="Mass law  (lossless, no stiffness)",
)

ax.set_title(
    "Silicone Earplug — Transmission Loss\n"
    r"(semi-infinite air on both sides, no earcanal, no eardrum)",
    fontsize=12,
)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("TL [dB]")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, which="both", alpha=0.4)
ax.set_xlim([50, 6500])
ax.set_ylim([0, 60])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.show()

i1 = np.argmin(np.abs(f - 1000))
i4 = np.argmin(np.abs(f - 4000))
print(
    f"TL at 1 kHz:  full={results['full'][i1]:.1f} dB  "
    f"| series={results['series'][i1]:.1f} dB  "
    f"| masslaw={tl_mass_law[i1]:.1f} dB"
)
print(
    f"TL at 4 kHz:  full={results['full'][i4]:.1f} dB  "
    f"| series={results['series'][i4]:.1f} dB  "
    f"| masslaw={tl_mass_law[i4]:.1f} dB"
)
