import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, iv, gamma

# ============================================================
# Ambient air
# ============================================================
rho0 = 1.21
c0 = 343.0
Z0 = rho0 * c0

f = np.linspace(50, 6500, 2000)
omega = 2 * np.pi * f
k0 = omega / c0

# ============================================================
# Silicone slab: exact 1D longitudinal model
# ============================================================
E_s = 2.9e6
nu_s = 0.49
rho_s = 1500.0
eta_s = 0.20

E_c = E_s * (1 + 1j * eta_s)
M_c = E_c * (1 - nu_s) / ((1 + nu_s) * (1 - 2 * nu_s))
c_L = np.sqrt(M_c / rho_s)
Z_L = rho_s * c_L
k_L = omega / c_L

l_ID = 0.006
phi = k_L * l_ID

# ============================================================
# MATLAB flexural circular plate model
# ============================================================
a = 0.02
h = l_ID
rhop = rho_s
nu_p = nu_s
E_p = E_s

S_p = np.pi * a**2
M_p = rhop * h * S_p
D = E_p * h**3 / (12 * (1 - nu_p**2))

k_p = np.sqrt(omega * np.sqrt(rhop * h / D))
x = k_p * a

# Exact plate impedance from MATLAB code
num = iv(1, x) * jv(0, x) + jv(1, x) * iv(0, x)
den = iv(1, x) * jv(2, x) - jv(1, x) * iv(2, x)
Z_plate_p_over_U = -1j * omega * M_p / (S_p**2) * (num / den)

# Convert p/U to p/u so it is consistent with the TL formula using Z0 = p/u
Z_plate = Z_plate_p_over_U * S_p

# Low-frequency asymptotic series from MATLAB
g2, g3, g4 = gamma(2), gamma(3), gamma(4)
coef = 16.0 / (1.0 / g3**2 - 1.0 / (g2 * g4))
term = 1.0 / g2 / (x**4) + ((g4 + g2) / (64.0 * g2 * g4) - 1.0 / (16.0 * g3))
Z_app_p_over_U = -1j * omega * M_p / (S_p**2) * coef * term
Z_app = Z_app_p_over_U * S_p

# ============================================================
# Transfer matrices
# ============================================================
T = {
    # Exact 1D propagating slab
    "full_1D_slab": [
        np.cos(phi),
        1j * Z_L * np.sin(phi),
        1j * np.sin(phi) / Z_L,
        np.cos(phi)
    ],

    # 1st-order thin approximation of the 1D slab
    "thin_1D_mass_compliance": [
        np.ones_like(omega),
        1j * omega * rho_s * l_ID * (1 + 1j * eta_s),
        1j * omega * l_ID / (rho_s * c_L**2),
        np.ones_like(omega)
    ],

    # Pure series mass approximation
    "series_mass_only": [
        np.ones_like(omega),
        1j * omega * rho_s * l_ID * (1 + 1j * eta_s),
        np.zeros_like(omega),
        np.ones_like(omega)
    ],

    # MATLAB exact flexural plate as series impedance
    "flexural_plate_exact": [
        np.ones_like(omega),
        Z_plate,
        np.zeros_like(omega),
        np.ones_like(omega)
    ],

    # MATLAB low-frequency asymptotic plate impedance
    "flexural_plate_asympt": [
        np.ones_like(omega),
        Z_app,
        np.zeros_like(omega),
        np.ones_like(omega)
    ]
}

# ============================================================
# TL computation
# ============================================================
def TL_from_matrix(T11, T12, T21, T22, Z=Z0):
    denom = T11 + T12 / Z + T21 * Z + T22
    tau = 2.0 * np.exp(1j * k0 * l_ID) / denom
    TL = 20.0 * np.log10(1.0 / np.abs(tau))
    return TL, tau

results = {name: TL_from_matrix(*vals) for name, vals in T.items()}

# Reference mass law
m_s = rho_s * l_ID
TL_mass_law = 20 * np.log10(omega * m_s / (2 * Z0))

# ============================================================
# Plot TL
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.semilogx(f, results["full_1D_slab"][0], lw=2.5,
            label="Full propagator (exact 1D slab)")
ax.semilogx(f, results["thin_1D_mass_compliance"][0], lw=2,
            linestyle="--", label="Thin 1D (mass + compliance)")
ax.semilogx(f, results["series_mass_only"][0], lw=2,
            linestyle="-.", label="Series Z (mass only)")
ax.semilogx(f, results["flexural_plate_exact"][0], lw=2,
            linestyle="-", label="Flexural plate exact (MATLAB)")
ax.semilogx(f, results["flexural_plate_asympt"][0], lw=2,
            linestyle=":", label="Flexural plate asymptotic")
ax.semilogx(f, TL_mass_law, lw=1.5,
            linestyle=(0, (3, 1, 1, 1)), label="Mass law reference")

ax.set_title("Transmission Loss comparison\n"
             "(1D slab approximations + MATLAB flexural plate)")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("TL [dB]")
ax.grid(True, which="both", alpha=0.4)
ax.set_xlim([50, 6500])
ax.set_ylim([0, 80])
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.show()

# ============================================================
# Plot series impedances for interpretation
# ============================================================
Z_series_mass = 1j * omega * rho_s * l_ID * (1 + 1j * eta_s)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(f, np.imag(Z_series_mass), lw=2, label="Series mass only")
ax.semilogx(f, np.imag(Z_plate), lw=2, label="Flexural plate exact")
ax.semilogx(f, np.imag(Z_app), lw=2, linestyle="--", label="Flexural plate asymptotic")
ax.set_title("Imaginary part of the series impedance")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel(r"Im$(Z_s)$ [Pa.s/m]")
ax.grid(True, which="both", alpha=0.4)
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Numerical checkpoints
# ============================================================
for f0 in [100, 500, 1000, 4000]:
    idx = np.argmin(np.abs(f - f0))
    print(f"\n--- {f[idx]:.0f} Hz ---")
    print(f"Full 1D slab             : {results['full_1D_slab'][0][idx]:7.2f} dB")
    print(f"Thin 1D mass+compliance  : {results['thin_1D_mass_compliance'][0][idx]:7.2f} dB")
    print(f"Series mass only         : {results['series_mass_only'][0][idx]:7.2f} dB")
    print(f"Flexural plate exact     : {results['flexural_plate_exact'][0][idx]:7.2f} dB")
    print(f"Flexural plate asympt    : {results['flexural_plate_asympt'][0][idx]:7.2f} dB")
    print(f"Mass law reference       : {TL_mass_law[idx]:7.2f} dB")