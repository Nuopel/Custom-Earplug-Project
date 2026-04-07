import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
r       = 3.77e-3    # m
L_slab  = 12e-3      # m
rho     = 1500.0     # kg/m³
E       = 1.7e6      # Pa
nu      = 0.48
eta     = 0.18

L_cav   = 10e-3      # m
rho_air = 1.2        # kg/m³
c_air   = 343.0      # m/s
Z0      = rho_air * c_air  # Pa·s/m

S = np.pi * r**2

# --- Frequency axis ---
freqs = np.geomspace(100, 10000, 500)
omega = 2 * np.pi * freqs

# --- Slab wave properties ---
E_star = E * (1 + 1j * eta)
M      = E_star * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
cL     = np.sqrt(M / rho)
k      = omega / cL
Zc     = rho * cL / S

# --- Slab T-matrix ---
T_slab = np.array([
    [np.cos(k * L_slab),           1j * Zc * np.sin(k * L_slab)],
    [1j * np.sin(k * L_slab) / Zc, np.cos(k * L_slab)]
])  # shape (2,2,N)

# --- Cavity T-matrix ---
ka = omega / c_air
Za = Z0 / S
T_cav = np.array([
    [np.cos(ka * L_cav),           1j * Za * np.sin(ka * L_cav)],
    [1j * np.sin(ka * L_cav) / Za, np.cos(ka * L_cav)]
])

# --- Cascade: T_tot = T_slab @ T_cav (per frequency) ---
T11 = T_slab[0,0] * T_cav[0,0] + T_slab[0,1] * T_cav[1,0]
T12 = T_slab[0,0] * T_cav[0,1] + T_slab[0,1] * T_cav[1,1]
T21 = T_slab[1,0] * T_cav[0,0] + T_slab[1,1] * T_cav[1,0]
T22 = T_slab[1,0] * T_cav[0,1] + T_slab[1,1] * T_cav[1,1]

# --- Pressures (p_in = 1 Pa, U_end = 0 rigid termination) ---
p_in = 1.0
p_end_occ = p_in / T11

L_open = L_slab + L_cav
p_end_air = p_in / np.cos(ka * L_open)  # open-air rigid-end reference

# --- SPL conversion ---
p_ref = 20e-6
spl_occ = 20 * np.log10(np.abs(p_end_occ) / p_ref)
spl_air = 20 * np.log10(np.abs(p_end_air) / p_ref)

# --- Insertion Loss ---
IL = 20 * np.log10(np.abs(p_end_air) / np.abs(p_end_occ))

# --- Plot: pressure at rigid end in dB SPL ---
plt.figure(figsize=(8, 5))
plt.semilogx(freqs, spl_air, label='Open air reference')
plt.semilogx(freqs, spl_occ, label='Occluded (slab + cavity)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Pressure at rigid end [dB SPL]')
plt.title('Rigid-end Pressure')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot: insertion loss ---
plt.figure(figsize=(8, 5))
plt.semilogx(freqs, IL, label='Insertion Loss')
plt.xlabel('Frequency [Hz]')
plt.ylabel('IL [dB]')
plt.title('Insertion Loss')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()