# WBS 6 Theory Note

## Objective

WBS 6 validates the modeling of narrow circular ducts with thermoviscous losses. In the toolbox, the dissipative duct is represented as an equivalent-fluid waveguide: the wall losses are first absorbed into an effective density and an effective bulk modulus, then these effective quantities are used in the standard 1D transfer-matrix form of a uniform duct.

The implementation used here is the `ViscothermalDuct` element, based on the circular Kirchhoff/Stinson model. A simpler asymptotic model, `BLIDuct`, is also used for comparison.

## 1. Lossless Reference

For a circular duct of radius $a$ and section
$
S = \pi a^2,
$
the lossless characteristic impedance and bulk modulus used in the code are
$
Z_{c,0} = \frac{\rho_0 c_0}{S},
\qquad
K_0 = \rho_0 c_0^2.
$

The ratio of specific heats is recovered from
$
\gamma = \frac{\rho_0 c_0^2}{P_0}.
$

## 2. Kirchhoff/Stinson Circular Loss Model

The viscous and thermal boundary-layer effects are introduced through two complex transverse parameters:
$
G_r = \sqrt{\frac{-j\omega\rho_0}{\eta_0}},
\qquad
G_k = \sqrt{\frac{-j\omega Pr\,\rho_0}{\eta_0}}.
$

The associated dimensionless arguments are
$
x_r = a G_r,
\qquad
x_k = a G_k.
$

The code then uses the circular Bessel-function correction factors
$
F_r = \frac{2 J_1(x_r)}{x_r J_0(x_r)},
\qquad
F_k = \frac{2 J_1(x_k)}{x_k J_0(x_k)}.
$

This gives the equivalent-fluid properties
$
\rho_{eq} = \rho_0 \left(1 - F_r\right)^{-1},
$
$
K_{eq} = K_0 \left(1 + (\gamma-1)F_k\right)^{-1}.
$

From these, the effective wavenumber and characteristic impedance are
$
k_{eq} = \omega \sqrt{\frac{\rho_{eq}}{K_{eq}}},
\qquad
Z_{c,eq} = \frac{\sqrt{\rho_{eq}K_{eq}}}{S}.
$

Finally, the propagation constant used by the transfer matrix is
$
\Gamma_{eq} = j k_{eq}.
$

In the implementation, the sign is chosen so that
$
\Re(\Gamma_{eq}) \ge 0,
\qquad
\Re(Z_{c,eq}) \ge 0,
$
which enforces attenuation in the physically correct direction.

## 3. Use in `ViscothermalDuct`

Once $\Gamma_{eq}$ and $Z_{c,eq}$ are known, the duct is treated exactly like a uniform 1D waveguide of length $L$. Its transfer matrix is
$
\mathbf{T}_{vt} =
\begin{bmatrix}
\cosh(\Gamma_{eq}L) & Z_{c,eq}\sinh(\Gamma_{eq}L) \\
\dfrac{\sinh(\Gamma_{eq}L)}{Z_{c,eq}} & \cosh(\Gamma_{eq}L)
\end{bmatrix}.
$

This is the matrix returned by `ViscothermalDuct.matrix(omega)`. In other words, all thermoviscous losses are carried by the complex constitutive quantities $(\Gamma_{eq}, Z_{c,eq})$, while the transfer-matrix structure remains the standard one for a uniform duct.

## 4. Simplified `BLIDuct` Model

For comparison, WBS 6 also uses `BLIDuct`, which is a first-order boundary-layer approximation. The viscous and thermal thicknesses are
$
\delta_v = \sqrt{\frac{2\eta_0}{\rho_0\omega}},
\qquad
\delta_t = \sqrt{\frac{2\eta_0}{\rho_0 Pr\,\omega}}.
$

The model corrects the lossless wavenumber $k=\omega/c_0$ as
$
k_{bli} = k \left[1 + \frac{1-j}{2}\left(\frac{\delta_v}{a} + (\gamma-1)\frac{\delta_t}{a}\right)\right],
$
then uses
$
\Gamma_{bli} = j k_{bli}.
$

An optional first-order correction is also applied to the characteristic impedance:
$
Z_{c,bli} = Z_{c,0}
\left[1 + \frac{1-j}{2}\left(\frac{\delta_v}{a} - (\gamma-1)\frac{\delta_t}{a}\right)\right].
$

This model is lighter but less robust for very small radii. In the code, a warning is issued when the radius becomes too close to the viscous boundary-layer thickness.

## 5. Meaning for WBS 6

The WBS 6 comparisons therefore test three levels of description:

- `CylindricalDuct`: no losses
- `BLIDuct`: asymptotic thermoviscous correction
- `ViscothermalDuct`: equivalent-fluid Kirchhoff/Stinson model

The main practical question is whether the full viscothermal model reproduces the FEM duct response closely enough to be used later as a reliable reduced element in the earplug and filter simulations.
