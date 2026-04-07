## 0. Physical Constants (Table II)

| Symbol      | Parameter                          | Value        | Unit     |
| ----------- | ---------------------------------- | ------------ | -------- |
| $P_0$     | Static pressure                    | 1.01 × 10⁵   | Pa       |
| $T_0$     | Temperature                        | 293.15       | K        |
| $\rho_0$  | Air density                        | 1.20         | kg/m³    |
| $c_0$     | Speed of sound                     | 343.90       | m/s      |
| $\mu$     | Dynamic viscosity                  | 1.82 × 10⁻⁵  | Pa·s     |
| $\lambda$ | Thermal conductivity               | 24.80 × 10⁻³ | W/(m·K)  |
| $\gamma$  | Ratio of specific heats            | 1.40         | —        |
| $C_p$     | Specific heat at constant pressure | 1.00 × 10³   | J/(kg·K) |

***

## 1. Geometric Dimensions (Table I, G.R.A.S. RA0045)

### Main Cavity

| Symbol  | Parameter          | Mean value |
| ------- | ------------------ | ---------- |
| $R_0$ | Radius             | 3.77 mm    |
| $L_0$ | Total length       | 12.56 mm   |
| $L_1$ | 1st section length | 3.12 mm    |
| $L_3$ | 2nd section length | 4.75 mm    |
| $L_5$ | 3rd section length | 4.69 mm    |

### 1st Helmholtz Resonator (rectangular slit + 1st annular cavity)

| Symbol  | Parameter                  | Mean value |
| ------- | -------------------------- | ---------- |
| $a_2$ | Slit length                | 2.53 mm    |
| $b_2$ | Slit width                 | 2.35 mm    |
| $h_2$ | Slit thickness             | 0.16 mm    |
| $r_2$ | Inner radius of 1st cavity | 6.30 mm    |
| $R_2$ | Outer radius of 1st cavity | 9.01 mm    |
| $d_1$ | Thickness of 1st cavity    | 1.91 mm    |

### 2nd Helmholtz Resonator (annular slit + 2nd annular cavity)

| Symbol     | Parameter                                                   | Mean value |
| ---------- | ----------------------------------------------------------- | ---------- |
| $r_4$    | Outer radius of annular slit (= inner radius of 2nd cavity) | 4.66 mm    |
| $\alpha$ | Angle of each of the 3 annular slit parts                   | 95.33°     |
| $h_4$    | Slit thickness                                              | 0.05 mm    |
| $R_4$    | Outer radius of 2nd cavity                                  | 9.01 mm    |
| $d_2$    | Equivalent thickness of 2nd cavity                          | 1.40 mm    |

***

## 2. Global TMM Structure

The simulator is decomposed into 5 two-port elements cascaded as:
$
\mathbf{T}^{ES} = T_1 \cdot T_2 \cdot T_3 \cdot T_4 \cdot T_5
$

where $T_1, T_3, T_5$ are the three main cavity sections and $T_2, T_4$ are the two Helmholtz resonators.

***

## 3. Main Cavity Sections — $T_m$, $m \in \{1,3,5\}$

Standard lossless plane wave TM (pressure / volume velocity convention): 

$
T_m = \begin{bmatrix} \cos(k_0 L_m) & j Z_1 \sin(k_0 L_m) \\ j \sin(k_0 L_m)/Z_1 & \cos(k_0 L_m) \end{bmatrix}
$

with:
$
k_0 = \frac{\omega}{c_0}, \qquad S_0 = \pi R_0^2, \qquad Z_1 = \frac{\rho_0 c_0}{S_0}
$

***

## 4. Helmholtz Resonators — $T_n$, $n \in \{2,4\}$

Each resonator is a pure shunt element (no propagation along main axis):

$
T_n = \begin{bmatrix} 1 & 0 \\ 1/Z_{HR,n} & 1 \end{bmatrix}
$

where $Z_{HR,n} = Z_{\text{slit},n} + Z_{\text{cav},n}$ is computed via the LRF model (see Sections 5 and 6).

***

## 5. 1st Helmholtz Resonator — Rectangular Slit + 1st Annular Cavity

### 5.1 LRF Thermoviscous Wavenumbers

Define the thermal and viscous characteristic lengths:
$
l_h = \frac{\lambda}{\rho_0 c_0 C_p}, \qquad l_\nu' = \frac{\mu}{\rho_0 c_0}
$

The thermoviscous wavenumbers (same for both slits):
$
k_h = \frac{1-j}{\sqrt{2}} \sqrt{\frac{k_0}{l_h}}, \qquad k_\nu = \frac{1-j}{\sqrt{2}} \sqrt{\frac{k_0}{l_\nu'}}
$

### 5.2 LRF Mean Fields for Rectangular Slit

The slit has thickness $h_2$, so the fields are evaluated at $h_2/2$:
$
K_{h,2} = 1 - \frac{\tan(k_h h_2/2)}{k_h h_2/2}, \qquad K_{\nu,2} = 1 - \frac{\tan(k_\nu h_2/2)}{k_\nu h_2/2}
$
$
K'_{h,2} = \gamma - (\gamma-1)\,K_{h,2}
$

### 5.3 LRF Impedance and Wavenumber of Rectangular Slit

Cross-section of slit: $S_{\text{slit},2} = b_2 \cdot h_2$

$
k_{l,2}^2 = k_0^2 \frac{K'_{h,2}}{K_{\nu,2}}, \qquad Z_{l,2}^2 = \frac{(\rho_0 c_0)^2}{S_{\text{slit},2}^2\, K'_{h,2}\, K_{\nu,2}}
$

### 5.4 End Correction for Rectangular Slit

The effective slit length is $a_2 + 2\Delta l_2$. The end correction uses $\beta_2 = h_2/b_2$ and $\varepsilon_2 = 1 + \beta_2^2$: 
$
\frac{\Delta l_2}{h_2} = \frac{1}{3\pi}\left[\beta_2 + \frac{1-\varepsilon_2^{3/2}}{\beta_2^2}\right] + \frac{1}{\pi}\left[\frac{1}{\beta_2}\ln\!\left(\beta_2 + \sqrt{\varepsilon_2}\right) + \ln\!\left(\frac{1}{\beta_2}(1+\sqrt{\varepsilon_2})\right)\right]
$

### 5.5 Rectangular Slit Impedance

$
Z_{\text{slit},2} = j Z_{l,2} \tan\!\left(k_{l,2}(a_2 + 2\Delta l_2)\right)
$

### 5.6 1st Annular Side Cavity Impedance

Cross-section: $S_{\text{cav},2} = \pi(R_2^2 - r_2^2)$

$
B_{s,2} = \frac{Y_1(k_0 R_2)}{J_1(k_0 R_2)}
$
$
Z_{\text{cav},2} = \frac{j\rho_0 c_0 \left[B_{s,2}\,J_0(k_0 r_2) - Y_0(k_0 r_2)\right]}{S_{\text{cav},2}\left[B_{s,2}\,J_1(k_0 r_2) - Y_1(k_0 r_2)\right]}
$

***

## 6. 2nd Helmholtz Resonator — Annular Slit + 2nd Annular Cavity

### 6.1 Annular Slit Cross-Section

The slit consists of 3 identical parts each at angle $\alpha = 95.33°$. The total arc length at inner radius $r_4$ and outer radius $R_0$ (= $R_2$ = 9.01 mm, the main cavity wall). The effective width of the annular slit is approximated by equivalent perimeters:

$
S_{\text{slit},4} = 3 \cdot \frac{\alpha}{360°} \cdot \pi(R_0^2 - r_4^2) \cdot h_4 / d_{\text{gap}}
$

In practice, the paper uses the annular slit as a radial propagation problem, so $S_{\text{slit},4}$ enters implicitly via $Z_{l,4}$ from the annular LRF model.

### 6.2 End Corrections for Annular Slit

Inner perimeter (equivalent width): $p_{\text{in}} = 2\pi r_4$, outer: $p_{\text{out}} = 2\pi R_0$. End corrections use the same baffled piston formula as Eq. 5.4 with $b \to p/3$ (for 3 sectors) and $h \to h_4$:

$
\Delta l_{4,\text{in}}: \quad \beta_{4,\text{in}} = \frac{h_4}{p_{\text{in}}/3}, \qquad \Delta l_{4,\text{out}}: \quad \beta_{4,\text{out}} = \frac{h_4}{p_{\text{out}}/3}
$

Corrected radii used in the Bessel function arguments:
$
R_0^{in} = R_0 - \Delta l_{4,\text{in}}, \qquad r_4^{out} = r_4 + \Delta l_{4,\text{out}}
$

### 6.3 LRF Fields for Annular Slit

Same structure as Section 5.2 but using $h_4$: 
$
K_{h,4} = 1 - \frac{\tan(k_h h_4/2)}{k_h h_4/2}, \qquad K_{\nu,4} = 1 - \frac{\tan(k_\nu h_4/2)}{k_\nu h_4/2}
$
$
K'_{h,4} = \gamma - (\gamma-1)\,K_{h,4}
$
$
k_{l,4}^2 = k_0^2 \frac{K'_{h,4}}{K_{\nu,4}}, \qquad Z_{l,4}^2 = \frac{(\rho_0 c_0)^2}{S_{\text{slit},4}^2\, K'_{h,4}\, K_{\nu,4}}
$

### 6.4 Annular Slit Impedance (radial propagation)

$
A_s = \frac{Y_0(k_{l,4}\, r_4^{out})}{J_0(k_{l,4}\, r_4^{out})}
$
$
Z_{\text{slit},4} = \frac{j Z_{l,4}\left[A_s J_0(k_{l,4}\,R_0^{in}) - Y_0(k_{l,4}\,R_0^{in})\right]}{A_s J_1(k_{l,4}\,R_0^{in}) - Y_1(k_{l,4}\,R_0^{in})}
$

### 6.5 2nd Annular Side Cavity Impedance

Cross-section: $S_{\text{cav},4} = \pi(R_4^2 - r_4^2)$

$
B_{s,4} = \frac{Y_1(k_0 R_4)}{J_1(k_0 R_4)}
$
$
Z_{\text{cav},4} = \frac{j\rho_0 c_0 \left[B_{s,4}\,J_0(k_0 r_4) - Y_0(k_0 r_4)\right]}{S_{\text{cav},4}\left[B_{s,4}\,J_1(k_0 r_4) - Y_1(k_0 r_4)\right]}
$

***

## 7. Input Impedance of the Simulator

From the global TM with rigid terminal ($Z_{term} \to \infty$): 
$
R_s = \frac{T_{11}^{ES} - T_{21}^{ES} Z_1}{T_{11}^{ES} + T_{21}^{ES} Z_1}, \qquad Z_s = Z_1\,\frac{1 + R_s}{1 - R_s}
$

## 8. Validation Targets

Your computed $Z_s$ should exhibit:
- 1st resonance of main cavity at ~**13.5 kHz**
- Helmholtz resonator 1 at ~**1400 Hz**
- Helmholtz resonator 2 at ~**3800 Hz**

***

## 9. Equivalent Tympanic Impedance (for earplug IL simulation)

Once $Z_s$ is known, use the **Reduced Impedance method** to propagate back through the main cavity length $L_0$: 
$
Z_{ep} = \frac{T_{22}\,Z_{rp} - T_{12}}{T_{11} - T_{21}\,Z_{rp}}
$

where $Z_{rp} = Z_s$ and the $T_{ij}$ elements are those of a simple cylindrical duct of length $L_0$:
$
T_{11} = T_{22} = \cos(k_0 L_0), \quad T_{12} = jZ_1\sin(k_0 L_0), \quad T_{21} = j\sin(k_0 L_0)/Z_1
$

This $Z_{ep}$ is then your terminal boundary condition in the ear canal TMM.
