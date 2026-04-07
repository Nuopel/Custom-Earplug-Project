# Transfer Matrix Method — Silicone Slab + Air Cavity + Rigid End

## Introduction

This document describes the transfer-matrix method (TMM) applied to a simplified earplug surrogate model: a viscoelastic silicone slab cascaded with an air cavity terminated by a rigid boundary (zero volume velocity), without the IEC 60318-4 ear simulator complexity.

**Configuration:** fixed inlet pressure → slab (12 mm) → air cavity (10 mm) → rigid end.

The TMM framework allows systematic computation of acoustic pressure and insertion loss by chaining 2×2 matrices representing each element.

---

## 1. State Vector Convention

The acoustic state at any plane is described by pressure $p$ (Pa) and volume velocity $U$ (m³/s):

$\begin{bmatrix} p_\text{in} \\ U_\text{in} \end{bmatrix} = \mathbf{T} \begin{bmatrix} p_\text{out} \\ U_\text{out} \end{bmatrix}$

For a 2×2 transfer matrix:

$\mathbf{T} = \begin{bmatrix} T_{11} & T_{12} \\ T_{21} & T_{22} \end{bmatrix}$

---

## 2. Generic Layer Matrix

For a homogeneous layer of length $L$, complex wavenumber $k$, and acoustic characteristic impedance $Z_c$:

$\mathbf{T} = \begin{bmatrix} \cos(kL) & j Z_c \sin(kL) \\ j \sin(kL)/Z_c & \cos(kL) \end{bmatrix}$

where $j = \sqrt{-1}$.

---

## 3. Silicone Slab Properties

### 3.1 Material Parameters

| Parameter       | Symbol          | Value      |
| --------------- | --------------- | ---------- |
| Radius          | $r$             | 3.77 mm    |
| Thickness       | $L_\text{slab}$ | 13 mm      |
| Density         | $\rho$          | 1500 kg/m³ |
| Young's modulus | $E$             | 1.7 MPa    |
| Poisson ratio   | $\nu$           | 0.48       |
| Loss factor     | $\eta$          | 0.18       |

Cross-sectional area:

$S = \pi r^2 = 4.47 \times 10^{-5} \text{ m}^2$

### 3.2 Complex Modulus and Wave Properties

**Complex Young's modulus** (viscoelastic damping via Kelvin–Voigt):

$E^* = E(1 + j\eta) = 1.7 \times 10^6 (1 + 0.18j) \text{ Pa}$

**Longitudinal modulus** (nearly incompressible isotropic solid):

$M = E^* \frac{1 - \nu}{(1 + \nu)(1 - 2\nu)}$

For $\nu = 0.48$:

$M \approx 4.3 \times 10^6 (1 + 0.18j) \text{ Pa}$

**Longitudinal wave speed**:

$c_L = \sqrt{\frac{M}{\rho}} \approx 53 \text{ m/s (complex)}$

**Complex wavenumber** ($\omega = 2\pi f$):

$k = \frac{\omega}{c_L}$

**Specific acoustic impedance**:

$z_{c,\text{spec}} = \rho c_L \approx 79\,500 \text{ Pa.s/m (complex)}$

**Acoustic characteristic impedance** (2-port):

$Z_c = \frac{z_{c,\text{spec}}}{S} \approx 1.78 \times 10^9 \text{ Pa.s/m}^3$

### 3.3 Slab Transfer Matrix

$\mathbf{T}_\text{slab} = \begin{bmatrix} \cos(k L_\text{slab}) & j Z_c \sin(k L_\text{slab}) \\ j \sin(k L_\text{slab})/Z_c & \cos(k L_\text{slab}) \end{bmatrix}$

At low frequencies ($kL \ll 1$), the slab behaves as a lumped mass–spring–damper:

- **Mass**: $m \approx \rho L_\text{slab} / S$
- **Stiffness**: $1/c \approx M L_\text{slab} / S$
- **Damping**: $r \approx \eta \sqrt{\rho M} \, L_\text{slab} / S$

Resonance frequency:

$f_r \approx \frac{1}{2\pi} \sqrt{\frac{M}{\rho L_\text{slab}^2}} \approx 400 \text{ Hz}$

---

## 4. Air Cavity Properties

### 4.1 Material Parameters

| Parameter          | Symbol            | Value      |
| ------------------ | ----------------- | ---------- |
| Length             | $L_\text{cav}$    | 10 mm      |
| Density            | $\rho_\text{air}$ | 1.2 kg/m³  |
| Sound speed        | $c_\text{air}$    | 343 m/s    |
| Specific impedance | $Z_0$             | 412 Pa·s/m |

**Air wavenumber**:

$k_\text{air} = \frac{\omega}{c_\text{air}}$

**Acoustic characteristic impedance**:

- $Z_\text{air,ac} = \frac{Z_0}{S} \approx 9.2 \times 10^6 \text{ Pa.s/m}^3$

### 4.2 Cavity Transfer Matrix

$\mathbf{T}_\text{cav} = \begin{bmatrix} \cos(k_\text{air} L_\text{cav}) & j Z_\text{air,ac} \sin(k_\text{air} L_\text{cav}) \\ j \sin(k_\text{air} L_\text{cav})/Z_\text{air,ac} & \cos(k_\text{air} L_\text{cav}) \end{bmatrix}$

---

## 5. Cascaded System: Slab + Cavity

The total transfer matrix from inlet to cavity end is:

$\mathbf{T}_\text{tot} = \mathbf{T}_\text{slab} \cdot \mathbf{T}_\text{cav}$

Explicitly:

$T_{11,\text{tot}} = T_{11,\text{slab}} T_{11,\text{cav}} - T_{12,\text{slab}} T_{21,\text{cav}}$

$T_{12,\text{tot}} = T_{11,\text{slab}} T_{12,\text{cav}} + T_{12,\text{slab}} T_{22,\text{cav}}$

$T_{21,\text{tot}} = T_{21,\text{slab}} T_{11,\text{cav}} + T_{22,\text{slab}} T_{21,\text{cav}}$

$T_{22,\text{tot}} = T_{21,\text{slab}} T_{12,\text{cav}} + T_{22,\text{slab}} T_{22,\text{cav}}$

---

## 6. Boundary Conditions

### 6.1 Inlet: Fixed Pressure

Plane-wave incident pressure (ideal source, zero source impedance):

$p_\text{in} = p_0 = 1 \text{ Pa}$

### 6.2 Outlet: Rigid Termination

At the rigid end, volume velocity is zero:

$U_\text{end} = 0$

From the transfer matrix row 1:

$p_\text{in} = T_{11,\text{tot}} \, p_\text{end} + T_{12,\text{tot}} \, U_\text{end}$

With $U_\text{end} = 0$:

$\boxed{p_\text{end} = \frac{p_\text{in}}{T_{11,\text{tot}}} = \frac{p_0}{T_{11,\text{tot}}}}$

---

## 7. Pressure Calculation — Step by Step

**Step 1 — Slab matrix:** compute $k = \omega/c_L$, $Z_c = \rho c_L / S$, form $\mathbf{T}_\text{slab}$.

**Step 2 — Cavity matrix:** compute $k_\text{air} = \omega/c_\text{air}$, $Z_\text{air,ac} = Z_0/S$, form $\mathbf{T}_\text{cav}$.

**Step 3 — Cascade:** $\mathbf{T}_\text{tot} = \mathbf{T}_\text{slab} \cdot \mathbf{T}_\text{cav}$.

**Step 4 — Occluded pressure:**

$p_\text{end,occ} = \frac{1}{T_{11,\text{tot}}}$

**Step 5 — Open reference** (air only, $L_\text{open} = L_\text{slab} + L_\text{cav} = 22$ mm):

$p_\text{end,open} = \frac{1}{\cos(k_\text{air} L_\text{open})}$

**Step 6 — Insertion loss:**

$\text{IL}(f) = 20 \log_{10} \left| \frac{p_\text{end,open}}{p_\text{end,occ}} \right| \text{ dB}$

---

## 8. Results

| Frequency     | $p_\text{end,occ}$ | $p_\text{end,open}$ | IL (dB) |
| ------------- | ------------------ | ------------------- | ------- |
| 100 Hz        | 0.67 Pa            | 1.00 Pa             | **3.5** |
| < 1 kHz mean  | ~0.85 Pa           | ~1.00 Pa            | 2–5     |
| Peak (~2 kHz) | —                  | —                   | **~90** |

---

---

## 10. Code Implementation

```python
# Slab properties
E_star = E * (1 + 1j * eta)
M = E_star * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
cL = np.sqrt(M / rho)
k = omega / cL
Zc = (rho * cL) / S

# Slab T-matrix
T11_slab = np.cos(k * L_slab)
T12_slab = 1j * Zc * np.sin(k * L_slab)
T21_slab = 1j * np.sin(k * L_slab) / Zc
T22_slab = np.cos(k * L_slab)

# Cavity T-matrix
ka = omega / c_air
Za = Z0 / S
T11_cav = np.cos(ka * L_cav)
T12_cav = 1j * Za * np.sin(ka * L_cav)
T21_cav = 1j * np.sin(ka * L_cav) / Za
T22_cav = np.cos(ka * L_cav)

# Cascade
T11_tot = T11_slab * T11_cav - T12_slab * T21_cav
T12_tot = T11_slab * T12_cav + T12_slab * T22_cav
T21_tot = T21_slab * T11_cav + T22_slab * T21_cav
T22_tot = T21_slab * T12_cav + T22_slab * T22_cav

# Pressure at rigid end (fixed p_in = 1 Pa)
p_end_occ  = 1.0 / T11_tot
p_end_open = 1.0 / np.cos(ka * L_open)

# Insertion loss
IL = 20 * np.log10(np.abs(p_end_open / p_end_occ))
```
