# Earplug TMM Simulation — Reference Document

---

# Part I — Theory

## 1. Acoustic State Vector and TMM Principle

The Transfer Matrix Method (TMM) propagates a state vector $\mathbf{v} = \{p, U\}^T$ — acoustic pressure and volume velocity — through a sequence of elements. Each element is described by a 2×2 complex matrix $\mathbf{T}(\omega)$ relating input to output:

$\begin{pmatrix} p_1 \\ U_1 \end{pmatrix} = \mathbf{T}(\omega) \begin{pmatrix} p_2 \\ U_2 \end{pmatrix}$

For a chain of $N$ elements, the global system matrix is the ordered product:

$\mathbf{T}_{sys}(\omega) = \mathbf{T}_1 \cdot \mathbf{T}_2 \cdots \mathbf{T}_N$

This product is **non-commutative** — order always reflects physical ordering from source to load. The matrix must satisfy $\det(\mathbf{T}) = S_{in}/S_{out}$ (reciprocity), equal to 1 for a uniform cross-section system.

## 2. Input Impedance

Given any load impedance $Z_L$ at the downstream end, the input impedance seen at the upstream end is:

$Z_{in}(\omega) = \frac{T_{11}\, Z_L + T_{12}}{T_{21}\, Z_L + T_{22}}$

This is the core observable from which TL, IL, and pressure distributions are all derived.

## 3. Transmission Loss (TL)

TL is an **intrinsic property of the filter element alone**, evaluated between two semi-infinite ducts of characteristic impedance $Z_c = \rho c / S$ (anechoic termination):

$\text{TL}(f) = 20\log_{10}\left|\frac{T_{11} + T_{12}/Z_c + T_{21}\,Z_c + T_{22}}{2}\right|$

TL characterises the device independently of its acoustic environment. It is the standard metric for muffler design in Munjal.

## 4. Insertion Loss (IL)

IL is the **system-level metric** relevant to REAT measurement. It compares the TM pressure with and without the filter inserted into the actual ear canal:

$\text{IL}(f) = 20\log_{10}\left|\frac{p_{TM}^{open}}{p_{TM}^{occluded}}\right|$

In terms of input impedance this simplifies to:

$\text{IL}(f) = 20\log_{10}\left|\frac{Z_{in}^{open}}{Z_{in}^{occ}}\right|$

---

## 5. Matrix Catalog

### 5.1 Lossless Cylindrical Duct (Munjal Ch. 2)

The exact plane-wave solution for a rigid-walled uniform duct of length $L$ and cross-section $S$:

$\mathbf{T}_{cyl} = \begin{pmatrix} \cos(kL) & jZ_c \sin(kL) \\ j\sin(kL)/Z_c & \cos(kL) \end{pmatrix}, \qquad k = \frac{\omega}{c}, \quad Z_c = \frac{\rho c}{S}$

Valid in the plane-wave regime $f < 1.84\,c / (\pi D)$. For the ear canal ($D \approx 8$ mm) this covers up to ~25 kHz, well above the target range.

---

### 5.2 Conical Duct — Munjal Exact vs. Successive Cylinders

The exact Munjal conical matrix introduces virtual apex distances $x_1, x_2$ measured from the cone tip:

$x_1 = \frac{R_1\,L}{R_2 - R_1}, \qquad x_2 = x_1 + L$

$\mathbf{T}_{cone} = \begin{pmatrix} \frac{x_1}{x_2}\!\left(\cos k\Delta - \frac{\sin k\Delta}{k x_2}\right) & \frac{j\rho c}{\pi R_1 R_2}\sin k\Delta \\ \frac{j\pi}{\rho c}\!\left(R_1^2\sin k\Delta + x_1 x_2\!\left[\cos k\Delta - \frac{\sin k\Delta}{k\Delta}\right]\!\cdot\!\frac{-(k\Delta)}{x_1 x_2}\right) & \frac{x_2}{x_1}\!\left(\cos k\Delta + \frac{\sin k\Delta}{k x_1}\right) \end{pmatrix}$

with $\Delta = L$.

The **successive cylinders** approximation subdivides the cone into $N_{sub}$ cylindrical slices with linearly interpolated radii. For Stinson ear canal geometry, 20–50 slices converges to <1 dB RMS versus the Munjal exact result.

| Approach             | Implementation cost  | Accuracy at 20 slices | BLI composable      |
| -------------------- | -------------------- | --------------------- | ------------------- |
| Munjal exact         | Medium (apex math)   | Reference             | Requires extra care |
| Successive cylinders | Low (reuses `T_cyl`) | <1 dB RMS             | Yes, trivial        |

---

### 5.3 Viscothermal Duct — Stinson / Kirchhoff

For narrow ducts where $R \lesssim \delta_v = \sqrt{2\mu/\rho\omega}$, viscous and thermal boundary layer losses are significant. This is the critical model for the ear canal at 4–7 mm radius. The Kirchhoff solution introduces complex wavenumber and characteristic impedance via the Bessel function ratio $J(z) = 2J_1(z)/(z J_0(z))$:

$\Gamma = \frac{j\omega}{c}\sqrt{\frac{1 + (\gamma-1)\,J(k_t R)}{1 - J(k_v R)}}$

$Z_c^{vt} = \frac{\rho c}{S}\sqrt{\frac{1}{\bigl(1 - J(k_v R)\bigr)\bigl(1 + (\gamma-1)\,J(k_t R)\bigr)}}$

where $k_v = \sqrt{j\omega\rho/\mu}$ (viscous) and $k_t = \sqrt{j\omega\rho C_p/\kappa}$ (thermal). The **matrix form is identical to `T_cyl`**, substituting $jk \to \Gamma$ and $Z_c \to Z_c^{vt}$.

This is the **Stinson tube dispersion model** — it accurately predicts the dispersive slow-wave behaviour in sub-mm bores, the metamaterial-analog regime where phase speed drops well below $c$ and attenuation per wavelength rises sharply.

**Simplified BLI approximation** (valid for canal segments where $R \gg \delta_v$):

$k_{eff} \approx k\left[1 + \frac{(1+j)}{R}\left(\sqrt{\frac{\mu}{\rho\omega}} + (\gamma-1)\sqrt{\frac{\kappa}{\rho c_p \omega}}\right)\right]$

Use this for phases 1–3; reserve the full Kirchhoff for sub-mm bore elements.

---

### 5.4 Impedance Discontinuity (Rupture d'Impédance)

At an abrupt area change $S_1 \to S_2$, pressure continuity and volume velocity conservation give the junction matrix:

$\mathbf{T}_{junc} = \begin{pmatrix} 1 & 0 \\ 0 & S_2/S_1 \end{pmatrix}$

For a more accurate model, an **end correction** $\delta\ell \approx 0.6133\,r_{small}$ adds a reactive acoustic mass in series immediately before the junction:

$Z_{mass} = \frac{j\omega\rho\,\delta\ell}{S_{small}}, \qquad \mathbf{T}_{mass} = \begin{pmatrix} 1 & Z_{mass} \\ 0 & 1 \end{pmatrix}$

---

### 5.5 Thin Plate / Lumped Shell

A flexible thin shell (silicone wall, membrane) acts as a **series impedance** element:

$\mathbf{T}_{plate} = \begin{pmatrix} 1 & Z_w \\ 0 & 1 \end{pmatrix}, \qquad Z_w = j\omega m_s - \frac{j k_s}{\omega} + r_s$

where $m_s = \rho_s\,t$ [kg/m²] is surface mass, $k_s = E\,t\,/\,[r^2(1-\nu^2)]$ is areal stiffness, and $r_s$ is structural damping resistance. For a silicone shell of $t = 2$ mm and $E \approx 1$ MPa, the mass law dominates above ~300 Hz.

---

### 5.6 JCA Porous Layer (Johnson-Champoux-Allard)

The JCA model describes an equivalent fluid with complex effective density and bulk modulus, requiring five parameters:

| Parameter               | Symbol          | Typical melamine foam |
| ----------------------- | --------------- | --------------------- |
| Open porosity           | $\phi$          | 0.99                  |
| Static flow resistivity | $\sigma$        | 8 000–12 000 Pa·s/m²  |
| High-freq. tortuosity   | $\alpha_\infty$ | 1.0–1.02              |
| Viscous char. length    | $\Lambda$       | 100–200 µm            |
| Thermal char. length    | $\Lambda'$      | 200–400 µm            |

**Effective density:**

$\tilde{\rho}(\omega) = \frac{\alpha_\infty \rho_0}{\phi}\left[1 + \frac{\sigma\phi}{j\omega\rho_0\alpha_\infty}\sqrt{1 + \frac{4j\alpha_\infty^2\eta\rho_0\omega}{\sigma^2\Lambda^2\phi^2}}\right]$

**Effective bulk modulus:**

$\tilde{K}(\omega) = \frac{\gamma P_0/\phi}{\displaystyle\gamma - (\gamma-1)\left[1 + \frac{8\kappa}{j\omega\rho_0 C_p \Lambda'^2}\sqrt{1 + \frac{j\omega\rho_0 C_p \Lambda'^2}{16\kappa}}\right]^{-1}}$

From these, $k_p = \omega\sqrt{\tilde{\rho}/\tilde{K}}$ and $Z_p = \sqrt{\tilde{\rho}\tilde{K}}/S$. The layer matrix is again `T_cyl` with these substituted.

---

### 5.7 Radiation Impedance

| Condition                       | Expression                                                                          |
| ------------------------------- | ----------------------------------------------------------------------------------- |
| Unflanged, low $ka$             | $Z_R = \frac{\rho c}{S}\left[\frac{(ka)^2}{4} + j\cdot 0.6133\,ka\right]$           |
| Flanged tube (Levine-Schwinger) | $Z_R = \frac{\rho c}{S}\left[1 - \frac{J_1(2ka)}{ka} + j\frac{H_1(2ka)}{ka}\right]$ |
| Rigid wall (hard termination)   | $Z_R \to \infty,\quad r = 1$                                                        |
| Anechoic (matched load)         | $Z_R = Z_c = \rho c / S$                                                            |

---

### 5.8 Eardrum Termination $Z_{TM}$

The Shaw (1974) model gives a lumped middle ear load as seen from the ear canal:

$Z_{TM}(\omega) = R_1 + j\omega M_1 + \frac{1}{j\omega C_1} + \frac{Z_{coch}}{1 + Z_{coch}/Z_{annular}}$

Typical parameters: $M_1 \approx 1.4 \times 10^{-3}$ kg/m⁴ (TM mass), $C_1 \approx 1.2 \times 10^{-12}$ m³/Pa (TM compliance), $R_{coch} \approx 10^9$ Pa·s/m³ above 1 kHz.

A practical first-pass is the **Keefe (1993)** measured dataset, providing $Z_{TM}(f)$ as tabulated complex values for an average adult ear. For the **IEC 711 coupler**, $Z_{TM}$ is replaced by the standardised 2 cc cavity impedance — purely stiffness-controlled at low frequency and reproducible for lab validation.

---

### 5.9 Mean Flow — Convected Wave (Mach Correction)

Relevant for vented plugs or breathing dynamics. Uniform mean flow at Mach number $M = U_0/c_0$ splits the wavenumber into downstream $k^+$ and upstream $k^-$ components:

$k^\pm = \frac{k}{1 \pm M}$

The convected cylindrical duct matrix becomes:

$\mathbf{T}_{flow} = \begin{pmatrix} \tfrac{1}{2}(e^{jk^+L} + e^{-jk^-L}) & \tfrac{Z_c}{2}(e^{jk^+L} - e^{-jk^-L}) \\ \tfrac{1}{2Z_c}(e^{jk^+L} - e^{-jk^-L}) & \tfrac{1}{2}(e^{jk^+L} + e^{-jk^-L}) \end{pmatrix}$

At $M = 0$ this reduces exactly to `T_cyl`. Implement as `FlowDuct(r, L, mach=0.0)` — activated only when $M > 0$.

---

# Part II — Python Package Architecture

## 6. Design Philosophy

The package is **class-oriented with operator overloading**. Each TMM block is an `AcousticElement` object encapsulating its physics and parameters. Cascade is defined via `__add__`, returning a new `AcousticElement` — so the system is always just one element that exposes `.Z_in()`, `.TL()`, and `.IL()` directly. No separate system class is needed because a composed element is physically equivalent to a single 2×2 matrix.

```python
canal  = sum(ViscothermalDuct(r_i, L_i) for r_i, L_i in stinson_segments)
plug   = ImpedanceJunction(S1, S2) + ViscothermalDuct(r_bore, L_bore) + JCALayer(params)
system = canal + plug + canal_post

IL = system.IL(Z_open=Z_canal_open, Z_occ=Z_TM, omega=omega)
```

## 7. Class Hierarchy

```
AcousticElement              (ABC)
├── ComposedElement          (result of A + B, itself an AcousticElement)
│
├── CylindricalDuct          (lossless, Munjal Ch.2)
├── ConicalDuct              (Munjal exact, apex distances)
├── ViscothermalDuct         (Kirchhoff/Stinson, k → Γ)
├── FlowDuct                 (CylindricalDuct + Mach convection)
│
├── ImpedanceJunction        (area step + optional end correction)
├── PlateSeriesImpedance     (area-normalized series plate obstruction)
├── AcousticMass             (lumped series element)
├── AcousticCompliance       (lumped shunt element)
│
├── JCALayer                 (Johnson-Champoux-Allard equivalent fluid)
│
└── MeasuredElement          (experimental T(f) from ASTM E2611 extraction)

Termination                  (not an AcousticElement — returns Z(omega) only)
├── EardrumImpedance         (Shaw model or tabulated Keefe data)
├── IEC711Coupler            (2 cc standardised cavity)
├── RadiationImpedance       (flanged / unflanged)
└── RigidWall                (Z → ∞)

Geometry
├── StinsonGeometry          (loads A(s) table, returns [(r_i, L_i)])
└── EarCanalBuilder          (Stinson → list of ViscothermalDuct → ComposedElement)
```

## 8. Base Class Design

```python
# pyearplug/elements/base.py
import numpy as np
from abc import ABC, abstractmethod

class AcousticElement(ABC):

    @abstractmethod
    def matrix(self, omega: np.ndarray) -> np.ndarray:
        # Returns complex (N_freq, 2, 2) transfer matrix
        ...

    def __add__(self, other):
        return ComposedElement(self, other)

    def __radd__(self, other):
        if other == 0:          # enables sum([e1, e2, e3])
            return self
        return other.__add__(self)

    def Z_in(self, Z_load: np.ndarray, omega: np.ndarray) -> np.ndarray:
        T = self.matrix(omega)
        return (T[:, 0, 0] * Z_load + T[:, 0, 1]) / \
               (T[:, 1, 0] * Z_load + T[:, 1, 1])

    def TL(self, Z_c: float, omega: np.ndarray) -> np.ndarray:
        T = self.matrix(omega)
        num = T[:, 0, 0] + T[:, 0, 1] / Z_c + T[:, 1, 0] * Z_c + T[:, 1, 1]
        return 20 * np.log10(np.abs(num / 2.0))

    def IL(self, Z_open: np.ndarray, Z_occ: np.ndarray,
           omega: np.ndarray) -> np.ndarray:
        Zin_open = self.Z_in(Z_open, omega)
        Zin_occ  = self.Z_in(Z_occ,  omega)
        return 20 * np.log10(np.abs(Zin_open / Zin_occ))

    def p_TM(self, p_in: float, Z_load: np.ndarray,
             omega: np.ndarray) -> np.ndarray:
        T = self.matrix(omega)
        return p_in * Z_load / (T[:, 0, 0] * Z_load + T[:, 0, 1])


class ComposedElement(AcousticElement):

    def __init__(self, left, right):
        self.left  = left
        self.right = right

    def matrix(self, omega: np.ndarray) -> np.ndarray:
        L = self.left.matrix(omega)     # (N_freq, 2, 2)
        R = self.right.matrix(omega)    # (N_freq, 2, 2)
        return np.einsum('nij,njk->nik', L, R)
```

**Key design decisions:**

- `matrix()` always returns `(N_freq, 2, 2)` — every subclass is **vectorised over frequency by contract**
- `ComposedElement` **is-a** `AcousticElement` → `(a + b + c).IL(...)` chains with zero boilerplate
- `__radd__` with sentinel `if other == 0` allows `sum(segment_list)` natively
- `np.einsum` for the cascade product — clean, readable, no explicit frequency loop
- `p_TM()`, `TL()`, `IL()`, `Z_in()` live in the base so every subclass inherits them for free

## 9. Package Structure

```
pyearplug/
│
├── pyproject.toml
├── constants.py               # ρ₀, c₀, μ, κ, Cₚ, γ at 20 °C
│
├── elements/
│   ├── base.py                # AcousticElement, ComposedElement
│   ├── ducts.py               # CylindricalDuct, ConicalDuct,
│   │                          # ViscothermalDuct, FlowDuct
│   ├── porous.py              # JCALayer
│   ├── lumped.py              # PlateSeriesImpedance, AcousticMass,
│   │                          # AcousticCompliance, ImpedanceJunction
│   ├── boundaries.py          # RadiationImpedance, EardrumImpedance,
│   │                          # IEC711Coupler, RigidWall
│   └── measured.py            # MeasuredElement (ASTM E2611 data)
│
├── geometry/
│   ├── stinson.py             # A(s) loader, segment discretisation
│   └── canal.py               # EarCanalBuilder
│
├── metrics.py                 # extract_T_from_measurements() (ASTM E2611)
│                              # standalone IL(), TL(), absorption_coeff()
├── optimize.py                # SciPy wrapper for flat IL target
│
├── data/
│   ├── stinson_geometry.csv   # R(s), A(s) from Stinson (1985) Table I
│   └── keefe_ZTM.csv          # tabulated ZTM(f) complex — Keefe (1993)
│
├── tests/
│   ├── test_ducts.py
│   ├── test_porous.py
│   ├── test_lumped.py
│   └── test_benchmarks.py     # TMM vs COMSOL cross-checks
│
└── notebooks/
    ├── 1a_bare_canal.ipynb
    ├── 1b_benchmark.ipynb
    ├── 2a_bare_tube.ipynb
    ├── 2b_silicone_shell.ipynb
    ├── 2c_filter_model.ipynb
    └── 3_optimization.ipynb
```

## 10. pytmm Refactor Strategy

`pytmm` (DerPhysikeR) is a minimal single-file implementation with scalar matrix functions but no frequency vectorisation, no class structure, and no boundary conditions. The refactor proceeds in four steps:

1. **Extract and audit** — pull `t_tube`, `t_cone`, `z_radiation` and map each to the corresponding class body in `elements/ducts.py` and `elements/boundaries.py`.
2. **Vectorise** — wrap each function in `np.vectorize` first, then rewrite as true array operations returning `(N_freq, 2, 2)`.
3. **Munjal additions** — implement `ConicalDuct`, `ViscothermalDuct`, `JCALayer`, and `FlowDuct` from scratch following the matrix catalog above. `pytmm` only seeds the lossless cylinder and radiation end correction.
4. **Class wrap** — drop each vectorised function into the corresponding `AcousticElement` subclass. Delete the original functional module; only the class API is exposed externally.

---

# Part III — Development Plan

## Phase 0 — Setup (Day 1)

- [x] Initialise `pyearplug` with `pyproject.toml` (`numpy`, `scipy`, `matplotlib`)
- [x] Configure `pytest` with minimal test runner and coverage
- [x] 
- [x] Define shared constants in `constants.py` ($\rho_0, c_0, \mu, \kappa, C_p, \gamma$ at 20 °C)

## Phase 1 — Core Engine (Days 2–4)

> Foundation. Nothing else builds without this.

- [x] Implement `AcousticElement` ABC and `ComposedElement` in `base.py`
- [x] Test `__add__`, `__radd__`, `sum()` chaining, `matrix()` shape contract `(N_freq, 2, 2)`
- [x] Implement `CylindricalDuct(r, L)` — lossless Munjal formula
  - Test: rigid wall → $|r| = 1$; matched load → $Z_{in} = Z_c$; $\lambda/4$ resonance at correct $f$
- [x] Implement `ConicalDuct(r1, r2, L)` — Munjal exact via apex distances
  - Test: as $r_1 \to r_2$, must converge to `CylindricalDuct`
- [x] Implement `ImpedanceJunction(S1, S2, end_correction=True)`
- [x] Implement `RadiationImpedance(mode='flanged'|'unflanged', r)`
- [x] Implement `RigidWall` and `MatchedLoad` terminations

## Phase 2 — Dissipation (Days 5–7)

- [x] Implement `ViscothermalDuct(r, L)` — Kirchhoff/Stinson using `scipy.special.j0`, `j1`
  - Test: at $r = 5$ mm, attenuation ~0.1 dB/cm at 1 kHz, increasing with frequency
  - Test: converges to `CylindricalDuct` as $r \to \infty$
- [x] Implement simplified `BLIDuct(r, L)` as lightweight alternative for canal segments
- [x] Implement `FlowDuct(r, L, mach=0.0)` — convected wave matrix
  - Test: at $M = 0$, must equal `CylindricalDuct`
- [x] Add `EardrumImpedance` (Shaw model) and `IEC711Coupler` to `boundaries.py`

## Phase 3 — Geometry (Days 8–9)

- [x] Implement `EarCanalBuilder`: discretise into $N$ `ViscothermalDuct` elements, return `sum(segments)`
- [x] Expose `n_segments` and `radius_scale` as parameters for sensitivity testing
- [x] Test: $Z_{in}$ of bare canal + $Z_{TM}$ must show quarter-wave resonance at ~3.4 kHz

## Phase 4 — Simulation 1: Bare Canal (Days 10–11)

```python
canal = EarCanalBuilder(stinson, n_segments=40).build()
Z_TM  = EardrumImpedance().Z(omega)
Z_in  = canal.Z_in(Z_TM, omega)
```

- [x] Plot $|Z_{in}(f)|$ magnitude and phase (100 Hz – 10 kHz)

- [x] Plot pressure transfer $|p_{TM}/p_{in}|(f)$ — expect +5 to +10 dB at resonance

- [x] Sensitivity sweep: vary mean radius ±1.5 mm, verify resonance shift
  
  ## Phase 5 — Porous and Plate Layers (Days 12–14)

- [x] Implement `JCALayer(phi, sigma, alpha_inf, Lambda, Lambda_prime, L, S)` in `porous.py`
  
  - Test: at $\sigma \to 0$, converges to lossless duct
  - Test: absorption coefficient matches literature for melamine foam

- [x] Implement plate-derived series obstruction element in `lumped.py`
  
  - Test: stiffness-controlled below resonance, mass-controlled above

## Phase 6 — Simulation 2: Bare Tube in Canal (Days 15–16)

Isolates the **impedance mismatch of the bore alone** before any lossy material.

```python
bore   = (ImpedanceJunction(S_canal, S_bore)
        + ViscothermalDuct(r_bore=1.5e-3, L_bore=15e-3)
        + ImpedanceJunction(S_bore, S_canal))
system = canal_pre + bore + canal_post
IL_bore = system.IL(Z_open, Z_TM, omega)
```

- [ ] Plot IL — expect 5–10 dB at high frequency from mass reactance, dip at bore resonance
- [ ] **Deliverable:** `2a_bare_tube.ipynb`

## Phase 7 — Simulation 3: Silicone Shell (Days 17–18)

Isolates the **elastic wall contribution** before foam is added.

```python
shell  = PlateSeriesImpedance(area=S_canal,
                              rho_plate=rho_sil,
                              h=t_shell,
                              E=E_sil,
                              nu=nu_sil)
system = canal_pre + shell + bore + canal_post
IL_sil = system.IL(Z_open, Z_TM, omega)
```

- [ ] Plot IL — expect mass-law slope above shell resonance (~200 Hz for 2 mm silicone)
- [ ] **Deliverable:** `2b_silicone_shell.ipynb`

## Phase 8 — Simulation 4: Full Filter IL (Days 19–21)

```python
plug = (ImpedanceJunction(S_canal, S_bore)
      + ViscothermalDuct(r_bore, L_bore)
      + JCALayer(foam_params, L_foam, S_bore)
      + PlateSeriesImpedance(area=S_canal, rho_plate=rho_sil, h=t_shell, E=E_sil, nu=nu_sil)
      + ImpedanceJunction(S_bore, S_canal))
system = canal_pre + plug + canal_post
IL = system.IL(Z_open=canal_pre.Z_in(Z_TM, omega), Z_occ=Z_TM, omega=omega)
```

- [ ] Plot IL(f), TL(f), and $|r_{med}|^2$ correction
- [ ] Verify target: 25–30 dB flat (500 Hz – 4 kHz), $|r_{med}| > 0.8$
- [ ] Sensitivity sweep on JCA $\sigma$ and $\Lambda$
- [ ] **Deliverable:** `2c_filter_model.ipynb`

## Phase 9 — Optimisation (Days 22–24)

- [ ] Implement `optimize.py`: SciPy `minimize` (L-BFGS-B) or `differential_evolution` wrapping IL
- [ ] Objective: $\min_\theta \;\text{RMS}[\text{IL}(\theta, f) - 27.5\,\text{dB}]$ over 500–4000 Hz
- [ ] Free parameters: `r_bore`, `L_bore`, `L_foam`, `sigma`, `Lambda`
- [ ] Constraints: `r_bore` in [0.5, 2.5] mm, `L_foam` in [2, 15] mm
- [ ] Export optimised params to JSON → input for COMSOL geometry and STL for 3D print
- [ ] **Deliverable:** `3_optimization.ipynb` + `optimized_params.json`

## Phase 10 — TMM vs. COMSOL Benchmark (Days 25–27)

- [ ] Implement `tests/test_benchmarks.py`: load COMSOL-exported CSV, compare to `canal.Z_in()`
- [ ] Criterion: RMS error < 1.5 dB on $|Z_{in}|$ and pressure transfer (100 Hz – 10 kHz)
- [ ] Sensitivity table: radius ±10%, BLI on/off, conical vs. successive cylinders
- [ ] **Deliverable:** `1b_benchmark.ipynb` + comparison table PNG

## Phase 11 — Impedance Tube / IEC 711 Integration (Days 28–30)

- [ ] Swap `EardrumImpedance` with `IEC711Coupler` and re-run all IL plots
- [ ] Implement `extract_T_from_measurements(p1, p2, p3, p4, x_mics, omega)` in `metrics.py` (ASTM E2611 four-mic two-load)
- [ ] Add `MeasuredElement(T_array, omega_array)` subclass in `elements/measured.py`
  - `.matrix()` interpolates tabulated complex $\mathbf{T}$ over frequency
  - Drops into `+` chain like any simulated element
- [ ] Cross-validate: `MeasuredElement(foam)` vs `JCALayer(params)` overlay plot
- [ ] Prepare physical measurement protocol document

| Method                    | Standard         | Microphones | Loads | Measures                        |
| ------------------------- | ---------------- | ----------- | ----- | ------------------------------- |
| Two-mic standing wave     | ISO 10534-2      | 2           | 1     | $\alpha$, $Z_{abs}$ only        |
| Four-mic transfer matrix  | ASTM E2611       | 4           | 2     | Full $\mathbf{T}$, TL, $\alpha$ |
| Three-mic two-load (3M2L) | E2611 complement | 3           | 2     | Full $\mathbf{T}$, TL           |

---

## Full Simulation Progression

| Step  | System                    | Load            | Key Observable                                     |
| ----- | ------------------------- | --------------- | -------------------------------------------------- |
| Sim 1 | Bare Stinson canal        | $Z_{TM}$        | Quarter-wave resonance ~3.4 kHz, baseline $Z_{in}$ |
| Sim 2 | Canal + bare bore         | $Z_{TM}$        | Mismatch IL, bore resonance                        |
| Sim 3 | Canal + silicone shell    | $Z_{TM}$        | Shell mass-law slope                               |
| Sim 4 | Canal + [bore + JCA foam] | $Z_{TM}$        | Target 25–30 dB flat IL                            |
| Sim 5 | All above                 | IEC 711 coupler | Lab-calibrated validation proxy                    |

---

## Timeline Summary

| Phase | Content                                           | Duration   |
| ----- | ------------------------------------------------- | ---------- |
| 0     | Setup, data, constants                            | Day 1      |
| 1     | Core engine: base class, lossless ducts, junction | Days 2–4   |
| 2     | Dissipation: viscothermal, BLI, flow, boundaries  | Days 5–7   |
| 3     | Geometry: Stinson loader, canal builder           | Days 8–9   |
| 4     | Sim 1: bare canal baseline                        | Days 10–11 |
| 5     | Porous JCA + thin plate                           | Days 12–14 |
| 6     | Sim 2: bare tube in canal                         | Days 15–16 |
| 7     | Sim 3: silicone shell                             | Days 17–18 |
| 8     | Sim 4: full filter IL                             | Days 19–21 |
| 9     | Optimisation                                      | Days 22–24 |
| 10    | TMM vs. COMSOL benchmark                          | Days 25–27 |
| 11    | Impedance tube / IEC 711 integration              | Days 28–30 |
