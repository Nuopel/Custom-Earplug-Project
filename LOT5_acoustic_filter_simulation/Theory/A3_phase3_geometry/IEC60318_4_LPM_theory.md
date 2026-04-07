# IEC 60318-4 Ear Simulator — Lumped Parameter Model (LPM)

> **Reference:** Luan et al. (2021) — *A Transfer Matrix Model of the IEC 60318-4 Ear Simulator*
> **Device:** G.R.A.S. RA0045 (compliant with IEC 60318-4)
> **Convention:** pressure / volume-velocity acoustic impedance [Pa·s·m⁻³]

---

## 1. Physical Constants

| Symbol    | Parameter                        | Value        | Unit       |
| --------- | -------------------------------- | ------------ | ---------- |
| $\rho_0$  | Air density                      | 1.20         | kg·m⁻³     |
| $c_0$     | Speed of sound                   | 343.90       | m·s⁻¹      |
| $\mu$     | Dynamic viscosity                | 1.82 × 10⁻⁵  | Pa·s       |
| $\lambda$ | Thermal conductivity             | 24.80 × 10⁻³ | W·m⁻¹·K⁻¹  |
| $\gamma$  | Ratio of specific heats          | 1.40         | —          |
| $C_p$     | Specific heat at const. pressure | 1.00 × 10³   | J·kg⁻¹·K⁻¹ |
| $P_0$     | Static pressure                  | 1.01 × 10⁵   | Pa         |
| $T_0$     | Temperature                      | 293.15       | K          |

---

## 2. Geometry (G.R.A.S. RA0045 — CT scan mean values)

### 2.1 Main Cavity (cylindrical tube)

| Symbol              | Parameter          | Value    |
| ------------------- | ------------------ | -------- |
| $R_0$               | Radius             | 3.77 mm  |
| $L_0 = L_1+L_3+L_5$ | Total length       | 12.56 mm |
| $L_1$               | 1st section length | 3.12 mm  |
| $L_3$               | 2nd section length | 4.75 mm  |
| $L_5$               | 3rd section length | 4.69 mm  |

Derived: $S_0 = \pi R_0^2$

### 2.2 1st Helmholtz Resonator (rectangular slit + 1st annular cavity)

| Symbol | Parameter                          | Value   |
| ------ | ---------------------------------- | ------- |
| $a_2$  | Slit length (radial)               | 2.53 mm |
| $b_2$  | Slit width                         | 2.35 mm |
| $h_2$  | Slit thickness                     | 0.16 mm |
| $r_2$  | Inner radius of 1st side cavity    | 6.30 mm |
| $R_2$  | Outer radius of 1st side cavity    | 9.01 mm |
| $d_1$  | Axial thickness of 1st side cavity | 1.91 mm |

Derived: $S_{\text{slit},2} = b_2 \cdot h_2$, $\quad V_{\text{cav},2} = \pi(R_2^2 - r_2^2)\,d_1$

### 2.3 2nd Helmholtz Resonator (annular slit × 3 + 2nd annular cavity)

| Symbol   | Parameter                           | Value   |
| -------- | ----------------------------------- | ------- |
| $r_4$    | Inner radius of 2nd side cavity     | 4.66 mm |
| $\alpha$ | Angle of each of the 3 slit sectors | 95.33°  |
| $h_4$    | Annular slit thickness              | 0.05 mm |
| $R_4$    | Outer radius of 2nd side cavity     | 9.01 mm |
| $d_2$    | Axial thickness of 2nd side cavity  | 1.40 mm |

Derived (equivalent rectangular approximation):

$a_4 = r_4 - R_0 \qquad b_4 = 3\,\alpha_\text{rad}\,\frac{R_0 + r_4}{2}$

$S_{\text{slit},4} = b_4 \cdot h_4 \qquad V_{\text{cav},4} = \pi(R_4^2 - r_4^2)\,d_2$

---

## 3. Circuit Topology

The simulator is modelled as a **ladder network** of 5 elements in cascade:

```
Input ──[m_a1]── NodeA ──[m_a3]── NodeB ──[m_a5]── NodeC ══╗
                   │                │                │     (rigid)
               c_a1║Z_HR2       c_a3║Z_HR4         c_a5
                  GND              GND              GND
```

- **Series elements** `[m_am]`: acoustic mass of each main cavity section
- **Shunt elements** at each node: cavity compliance $c_{a,m}$ **in parallel** with the Helmholtz resonator impedance $Z_{\text{HR},n}$
- **Rigid terminal** at NodeC: infinite impedance → zero volume velocity

---

## 4. LPM Element Formulas

### 4.1 Main Cavity Sections ($m = 1, 3, 5$)

Each section of the main cylindrical cavity contributes:

$m_{a,m} = \frac{\rho_0\, L_m}{S_0}
\qquad [\text{kg·m}^{-4}]$

$c_{a,m} = \frac{S_0\, L_m}{\rho_0\, c_0^2} = \frac{V_m}{\rho_0\, c_0^2}
\qquad [\text{m}^3\text{·Pa}^{-1}]$

where $V_m = S_0 \cdot L_m$ is the section volume.

### 4.2 Helmholtz Resonator Neck — Rectangular Slit ($n = 2$)

From the Hagen-Poiseuille law for viscous flow between parallel plates:

$r_{a,2} = \frac{12\,\mu\,a_2}{b_2\,h_2^3}
\qquad [\text{Pa·s·m}^{-3}]$

$m_{a,2} = \frac{6\,\rho_0\,a_2}{5\,S_{\text{slit},2}}
\qquad [\text{kg·m}^{-4}]$

> **Note:** The factor 6/5 arises from the kinetic energy correction for Poiseuille flow.
> The factor 12 comes from integrating the viscous shear stress across the gap.

### 4.3 Helmholtz Resonator Neck — Annular Slit ($n = 4$)

The same formulas apply using the equivalent rectangular dimensions:

$r_{a,4} = \frac{12\,\mu\,a_4}{b_4\,h_4^3}
\qquad m_{a,4} = \frac{6\,\rho_0\,a_4}{5\,S_{\text{slit},4}}$

> **Warning:** The annular slit has $h_4 = 0.05$ mm. The viscous penetration depth
> $\delta_\nu = \sqrt{2\mu/(\omega\rho_0)}$ reaches **0.07 mm at 1 kHz** — larger than the gap.
> The Poiseuille approximation is therefore **invalid at all audio frequencies** for this slit.
> This is the primary source of LPM error above 1 kHz.

### 4.4 Helmholtz Resonator Side Cavity ($n = 2, 4$)

$c_{a,n} = \frac{V_{\text{cav},n}}{\rho_0\,c_0^2}
\qquad [\text{m}^3\text{·Pa}^{-1}]$

### 4.5 Helmholtz Resonator Input Impedance

$Z_{\text{HR},n} = r_{a,n} + j\omega\,m_{a,n} + \frac{1}{j\omega\,c_{a,n}}
\qquad [\text{Pa·s·m}^{-3}]$

This is a series RLC circuit. Resonance frequency:

$f_{r,n} = \frac{1}{2\pi\sqrt{m_{a,n}\,c_{a,n}}}$

| Resonator        | LPM $f_r$ | Physical target |
| ---------------- | --------- | --------------- |
| HR1 (rect. slit) | ≈ 1221 Hz | ≈ 1400 Hz       |
| HR2 (ann. slit)  | ≈ 3359 Hz | ≈ 3800 Hz       |

---

## 5. Input Impedance — Backward Recursion

The input impedance $Z_s$ is computed **from the rigid terminal backward** to the input:

**Step 1** — NodeC (rigid end, only shunt $c_{a,5}$):
$Z_C = \frac{1}{j\omega\,c_{a,5}}$

**Step 2** — Add series mass $m_{a,5}$:
$Z_C' = Z_C + j\omega\,m_{a,5}$

**Step 3** — NodeB (shunt $c_{a,3} \parallel Z_{\text{HR},4}$):
$Z_B = \left[\frac{1}{Z_C'} + j\omega\,c_{a,3} + \frac{1}{Z_{\text{HR},4}}\right]^{-1}$

**Step 4** — Add series mass $m_{a,3}$:
$Z_B' = Z_B + j\omega\,m_{a,3}$

**Step 5** — NodeA (shunt $c_{a,1} \parallel Z_{\text{HR},2}$):
$Z_A = \left[\frac{1}{Z_B'} + j\omega\,c_{a,1} + \frac{1}{Z_{\text{HR},2}}\right]^{-1}$

**Step 6** — Add series mass $m_{a,1}$ → input impedance:
$\boxed{Z_s = Z_A + j\omega\,m_{a,1}}$

---

## 6. IEC 60318-4 Reference — Unit Conversion

The IEC 60318-4 Table 1 specifies the **acoustic transfer impedance**:

$Z_T = \frac{p_\text{mic}}{U_\text{in}} \qquad \text{[Pa·s·m}^{-3}\text{]}$

expressed in **dB re 1 MPa·s·m⁻³**.

To compare with the LPM input impedance $Z_s$ (in dB re 1 Pa·s·m⁻³):

$\text{Level}_{\text{Pa}} = \text{Level}_{\text{MPa}} + 20\log_{10}(10^6) = \text{Level}_{\text{MPa}} + 120\;\text{dB}$

The relationship between input and transfer impedance:

$Z_s = T_{11}^{ES} \cdot Z_T, \qquad T_{11}^{ES} = \cos(k_0 L_0)$

At low frequencies $T_{11} \approx 1$ (difference < 0.5 dB below 2 kHz).

---

## 7. Known Limitations of the LPM

| Limitation                             | Effect                       | Frequency range           |
| -------------------------------------- | ---------------------------- | ------------------------- |
| Poiseuille flow assumption in slits    | Wrong $r_a$, $m_a$           | All frequencies for $h_4$ |
| No thermoviscous losses in main cavity | Missing wall damping         | All frequencies           |
| Lumped (not distributed) main cavity   | Phase errors                 | > 3 kHz                   |
| Approx. annular→rectangular slit       | Biased HR2 resonance         | > 1 kHz                   |
| **Net result**                         | **2–5 dB error vs IEC spec** | **> 1 kHz**               |

**Remedy:** Use the Transfer Matrix + Low Reduced Frequency (TM+LRF) model, which replaces
the Poiseuille neck approximations with exact complex thermoviscous wavenumbers and replaces
the lumped LC ladder with proper transmission line matrices for the main cavity.

---

## 8. Numerical Values Summary

```
S0     = 44.65 mm²       (main cavity cross-section)

Main cavity sections:
  m_a1 =   83.85  kg/m⁴    c_a1 = 9.816e-13 m³/Pa
  m_a3 =  127.66  kg/m⁴    c_a3 = 1.494e-12 m³/Pa
  m_a5 =  126.04  kg/m⁴    c_a5 = 1.476e-12 m³/Pa

HR1 — rectangular slit (b2×h2 = 2.35×0.16 mm):
  r_a2 = 5.74e+07  Pa·s/m³
  m_a2 = 9689.4    kg/m⁴
  c_a2 = 1.754e-12 m³/Pa      →  fr1 ≈ 1221 Hz

HR2 — annular slit ×3 (a4=0.89mm, b4=21.04mm, h4=0.05mm):
  r_a4 = 7.39e+07  Pa·s/m³
  m_a4 = 1218.3    kg/m⁴
  c_a4 = 1.843e-12 m³/Pa      →  fr2 ≈ 3359 Hz
```

---

*Next step: TM + LRF model — replaces the above with exact thermoviscous wave propagation.*
