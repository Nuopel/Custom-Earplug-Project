# WB9 — Porous Layer Models

This folder studies porous slabs inserted in a duct using equivalent-fluid models.

The goal is to compare:

- different porous constitutive laws
- their TMM implementation in `acoustmm`
- their effect on IEC711 / rigid-end insertion loss

The current examples cover:

- `A0`: JCA porous slab in duct, compared to FEM
- `A1`: JCA vs Miki in the same duct system
- `A2`: Miki sigma sweep in TMM only

## 1. Common TMM Form

Both porous models are implemented as homogeneous equivalent-fluid layers with transfer matrix:

$
T_p(\omega)=\begin{bmatrix}\cos(kL) & j Z_c \sin(kL) \\j \sin(kL) / Z_c & \cos(kL)\end{bmatrix}
$

where:

- \(L\): porous slab thickness
- \(k(\omega)\): complex wavenumber
- \(Z_c(\omega)\): characteristic impedance in the `p/U` state basis

In the code, the state vector is \([p, U]\), so the characteristic impedance is scaled by the duct area.

## 2. JCA Layer

`JCALayer` is the Johnson-Champoux-Allard equivalent-fluid porous slab.

### Inputs

- \(\phi\): porosity
- \(\sigma\): airflow resistivity
- \(\alpha_\infty\): tortuosity
- \(\Lambda\): viscous characteristic length
- \(\Lambda'\): thermal characteristic length
- \(L\): slab thickness
- \(S\): duct cross-sectional area

### Effective properties

The model computes:

- complex effective density \(\rho_\mathrm{eff}(\omega)\)
- complex effective bulk modulus \(K_\mathrm{eff}(\omega)\)

In the implementation, the Johnson-Champoux-Allard formulas are written as:

$
\rho_\mathrm{eff}(\omega)
=
\rho_0 \alpha_\infty
\left(
1
+\frac{\sigma \phi}{j \omega \alpha_\infty \rho_0}
\cdot
\sqrt{
1
+\frac{
4 j \alpha_\infty^2 \eta \rho_0 \omega
}{
\sigma^2 \Lambda^2 \phi^2
}
}
\right)
$

equivalently, using the code form:

$
\rho_\mathrm{eff}(\omega)
=
\rho_0 \alpha_\infty
\left[
1
+\frac{\sigma \phi}{j \omega \alpha_\infty \rho_0}\cdot\sqrt{1+\frac{4 j \alpha_\infty^2 \eta \rho_0 \omega}{
\sigma^2 \Lambda^2 \phi^2
}
}
\right]
$

and:

$
K_\mathrm{eff}(\omega)=\frac{\gamma p_0}{
\gamma-\dfrac{\gamma-1}{
1 + \dfrac{8 \eta}{j \rho_0 \omega \Pr \Lambda'^2}
\sqrt{
1 + \frac{j \rho_0 \omega \Pr \Lambda'^2}{16 \eta}
}
}
}
$

The code uses the same expression in the form:

$
K_\mathrm{eff}(\omega)=\frac{\gamma p_0}{
\gamma-
\dfrac{\gamma-1}{
1 + \dfrac{8 \eta \sqrt{1 + \frac{j \rho_0 \omega \Pr \Lambda'^2}{16 \eta}}}{j \rho_0 \omega \Pr \Lambda'^2}
}
}
$

then:

$
k(\omega)=\omega \sqrt{\frac{\rho_\mathrm{eff}}{K_\mathrm{eff}}}
$

$
Z_\mathrm{char}(\omega)=\sqrt{\rho_\mathrm{eff} K_\mathrm{eff}}
$

### `p/U` normalization used in `acoustmm`

In the implementation:

$
Z_c(\omega)=\frac{Z_\mathrm{char}(\omega)}{S\,\phi}
$

The extra \(\phi\) appears because the porous equivalent-fluid relation is expressed on the pore-scale fluid velocity, while the TMM state uses total volume velocity \(U\).

### Usage in examples

- `A0_jca_in_duct_iec_or_rigidend.py`
- also reused as reference in `A1`

## 3. Miki Layer

`MikiLayer` is an empirical equivalent-fluid porous slab based on the Miki model.

### Inputs

The dominant physical input is:

- \(\sigma\): airflow resistivity

and, for the layer element:

- \(L\): slab thickness
- \(S\): duct cross-sectional area

In the current implementation, `PorousMaterial` is still used as a container, but the Miki law depends essentially on \(\sigma\), \(c_0\), and \(\rho_0\).

### Miki formulas

Define:

$
r = \frac{\rho_0 f}{\sigma}
$

Then:

$
Z_\mathrm{char}(\omega)=\rho_0 c_0\left(1 + 0.0785\, r^{-0.632}- j\,0.120\, r^{-0.632}\right)
$

$
k(\omega)=\frac{\omega}{c_0}\left(1 + 0.122\, r^{-0.618}- j\,0.180\, r^{-0.618}\right)
$

### `p/U` normalization used in `acoustmm`

For `MikiLayer`, the implementation uses:

$
Z_c(\omega)=\frac{Z_\mathrm{char}(\omega)}{S}
$

There is no additional porosity factor in the current empirical Miki-layer implementation.

### Usage in examples

- `A1_miki_vs_jca_in_duct_iec_or_rigidend.py`
- `A2_miki_sigma_sweep_in_duct_iec_or_rigidend.py`

## 4. Physical Interpretation

### Thickness effect

Increasing slab thickness \(L\):

- increases attenuation
- changes the internal phase accumulation
- usually raises insertion loss over a broader band

### Resistivity effect

Increasing \(\sigma\):

- increases viscous resistance
- usually increases dissipation up to a point
- can also make the layer more reflective

So a sigma sweep is useful to locate:

- too-open foams
- useful intermediate damping
- overly resistive layers

## 5. Practical Reading of the Examples

### `A0`

Use this when you want:

- one porous slab
- one FEM comparison
- full duct response with IEC711 / rigid-end outputs

### `A1`

Use this when you want:

- same slab geometry
- same duct
- JCA and Miki compared directly

This isolates the effect of the porous constitutive law itself.

### `A2`

Use this when you want:

- a pure TMM design-space study
- quick sigma and thickness trends
- no FEM dependency

## 6. Summary

WB9 currently covers two porous equivalent-fluid families:

- **JCA**: more physical, more parameters
- **Miki**: empirical, simpler, mostly controlled by \(\sigma\)

Both are inserted into the same duct through the same TMM layer form:

$
T_p(\omega)=
\begin{bmatrix}
\cos(kL) & j Z_c \sin(kL) \\
j \sin(kL) / Z_c & \cos(kL)
\end{bmatrix}
$

The main differences come from how \(k(\omega)\) and \(Z_c(\omega)\) are computed.
