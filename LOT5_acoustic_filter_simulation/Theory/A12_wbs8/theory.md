# Families of models

## A. Equivalent acoustic impedance models

These models directly prescribe the acoustic series impedance:

$
T_f(\omega) =
\begin{bmatrix}
1 & Z_f(\omega) \\
0 & 1
\end{bmatrix}
$

They are phenomenological models.

***

### A1. Resistive film

$
Z_f = R_f
$

Parameters:

- \(R_f\): acoustic series resistance \([\mathrm{Pa\,s/m^3}]\)

***

### A2. Mass-only film

$
Z_f = j\omega M_f
$

Parameters:

- \(M_f\): acoustic series mass \([\mathrm{Pa\,s^2/m^3}]\)

If starting from a sheet with surface density \(\mu\) over area \(S\):

$
Z_f = j\omega \frac{\mu}{S}
\qquad \Rightarrow \qquad
M_f = \frac{\mu}{S}
$

***

### A3. Mass + resistance

$
Z_f = R_f + j\omega M_f
$

or, from sheet parameters:

$
Z_f = R_f + j\omega \frac{\mu}{S}
$

***

### A4. Mass + resistance + stiffness

$
Z_f = R_f + j\omega M_f + \frac{K_f}{j\omega}
$

This is the generic resonant lumped model.

Parameters:

- \(R_f\): damping
- \(M_f\): effective acoustic mass
- \(K_f\): effective acoustic stiffness

If derived from a membrane-like element, \(M_f\) and \(K_f\) are effective fitted quantities.

***

## B. Bulk elastic models

These models describe a material layer that fully fills the duct cross-section. Unlike membrane or plate models, they are based on **bulk longitudinal motion inside the material**. They differ mainly by the level of approximation used to represent the slab dynamics.

***

### B1. Exact elastic slab

This is the most complete model of the family. The slab is treated as a finite-thickness elastic layer supporting longitudinal wave propagation through its thickness. Its transfer matrix is:

$
T_s(\omega) =
\begin{bmatrix}
\cos(k_L L) & j Z_{c,L}\sin(k_L L) \\
j\sin(k_L L)/Z_{c,L} & \cos(k_L L)
\end{bmatrix}
$

with

$
k_L = \frac{\omega}{c_L}, \qquad
c_L = \sqrt{\frac{M_L}{\rho}}, \qquad
Z_{c,L} = \frac{\rho\, c_L}{S}
$

and

$
M_L = E^* \frac{1-\nu}{(1+\nu)(1-2\nu)}, \qquad E^* = E(1+j\eta).
$

Parameters:

- \(E\): Young's modulus
- \(\nu\): Poisson's ratio
- \(\eta\): loss factor
- \(\rho\): material density
- \(L\): slab thickness
- \(S\): cross-sectional area

This model captures full longitudinal propagation, including phase accumulation, internal resonance effects, and material losses.

***

### B2. Thin-slab first-order approximation

This is a low-frequency approximation of the exact slab model, obtained by a first-order Taylor expansion of the exact transfer matrix for small \(k_L L\). The slab is represented by both a **series inertive term** and a **shunt compliance term**:

$
T_s(\omega) \approx
\begin{bmatrix}
1 & Z_s(\omega) \\
Y_s(\omega) & 1
\end{bmatrix}
$

with

$
Z_s(\omega) = j\omega \frac{\rho L}{S}
$

and

$
Y_s(\omega) = j\omega \frac{L\, S}{M_L}
$

where \(M_L = E^*(1-\nu)/[(1+\nu)(1-2\nu)]\) with \(E^* = E(1+j\eta)\) carries the material losses. Note that losses enter only through the compliance term \(Y_s\) (via the complex modulus \(E^*\)), not the mass term \(Z_s\), since \(\rho\) is a real physical density.

This model is more accurate than the purely inertive approximation because it retains both the mass and compressibility of the slab, while remaining much simpler than the full propagating model.

***

### B3. Lumped inertive approximation

This is the simplest model of the bulk elastic family. The slab is reduced to a pure lumped series impedance accounting only for its inertial effect:

$
T_s(\omega) =
\begin{bmatrix}
1 & Z_s(\omega) \\
0 & 1
\end{bmatrix}
$

with

$
Z_s(\omega) = j\omega \frac{\rho L}{S}
$

A phenomenological structural damping term may optionally be added:

$
Z_s(\omega) = j\omega \frac{\rho L}{S}(1+j\eta)
$

Note that this \((1+j\eta)\) on the mass term is a shorthand engineering convention for added damping; it is **not** derived from the elastic model hierarchy above, where losses enter only through \(E^*\).

This model neglects internal propagation, compliance, and resonance effects. It should be interpreted as the mass-only limit of the elastic slab.

***

### Summary of the bulk elastic family

- **B1. Exact elastic slab**: full distributed longitudinal propagation model.
- **B2. Thin-slab first-order approximation**: retains both inertive and compliance effects.
- **B3. Lumped inertive approximation**: retains only the inertive effect.

The progression is:

$
\text{Exact slab} \;\rightarrow\; \text{thin-slab first-order model} \;\rightarrow\; \text{mass-only lumped model}.
$

***

## C. Structural membrane models

These models describe a thin stretched surface inserted across the acoustic path. The restoring force is mainly due to **tension**, making them appropriate for taut films, stretched membranes, or soft sheets under preload.

***

### C1. Membrane model

#### Governing equation

$ \mu \frac{\partial^2 w}{\partial t^2}+ c_d \frac{\partial w}{\partial t}- T\, \nabla_\perp^2 w= \Delta p$

Parameters:

- $(\mu)$: surface mass density $([\mathrm{kg/m^2}])$
- $(T)$: membrane tension $([\mathrm{N/m}])$
- $(c_d)$: damping coefficient
- Geometry: membrane radius $(a)$

#### Exact model

The exact impedance \(Z_m(\omega)\) is obtained by solving this PDE and relating the volume-averaged membrane velocity to the pressure jump:

$
\Delta p = Z_m(\omega)\cdot U
$

There is no single universal closed form unless the geometry, mode set, and averaging procedure are fully specified.

#### Lumped approximation

For a circular membrane of radius \(a\) and area \(S = \pi a^2\), a practical first approximation is:

$
Z_m(\omega) \approx R_m + j\omega \frac{\mu}{S} + \frac{1}{j\omega}\frac{C_T\, T}{a^2\, S}
$

where \(C_T\) is a dimensionless geometry/mode constant.

#### Useful scalings

$
M_m = \frac{\mu}{S}, \qquad K_m = \frac{C_T\, T}{a^2\, S}
$

so the structural membrane model reduces to the standard lumped form:

$
Z_m(\omega) \approx R_m + j\omega M_m + \frac{K_m}{j\omega}
$

***

## D. Structural flexural plate models

These models describe a thin transverse element whose restoring force is mainly due to **bending stiffness**. They are appropriate for thin clamped plates, stiff foils, polymer disks, or plate-like inserts.

***

### D1. Flexural plate model

#### Plate parameters

- \(E\): Young's modulus
- \(\nu\): Poisson's ratio
- \(h\): thickness
- \(\rho\): density
- \(a\): plate radius, \(S = \pi a^2\)

#### Bending stiffness

$
D = \frac{E\, h^3}{12(1-\nu^2)}
$

#### Mass

$
\mu_p = \rho h, \qquad m_p = \rho h S
$

#### Flexural wavenumber

$
k_p = \left(\frac{\rho h\,\omega^2}{D}\right)^{1/4}
$

#### Acoustic series impedance

$
Z_p(\omega) = -j\omega \frac{m_p}{S^2}
\left(
\frac{
I_1(x)J_0(x) + J_1(x)I_0(x)
}{
I_1(x)J_2(x) - J_1(x)I_2(x)
}
\right), \qquad x = k_p\, a
$

where \(J_n\) and \(I_n\) are the Bessel and modified Bessel functions of order \(n\).

The corresponding TMM element is:

$
T_p(\omega) =
\begin{bmatrix}
1 & Z_p(\omega) \\
0 & 1
\end{bmatrix}
$

***

### D2. Low-frequency flexural plate approximation

This model is the **low-frequency approximation** of the exact flexural plate impedance in D1. It is useful when the plate operates well below its first flexural resonance, or more generally when

$ x = k_p a \ll 1. $

In that regime, the exact Bessel-form impedance can be expanded asymptotically, leading to a simpler closed-form expression. ([perso.univ-lemans.fr][1])

#### Plate parameters

* (E): Young's modulus

* (\nu): Poisson's ratio

* (h): thickness

* (\rho): density

* (a): plate radius, (S = \pi a^2)

#### Bending stiffness

$ D = \frac{E, h^3}{12(1-\nu^2)} $

#### Surface mass

$ \mu_p = \rho h $

#### Flexural wavenumber

$ k_p = \left(\frac{\rho h\omega^2}{D}\right)^{1/4} $

#### Low-frequency acoustic series impedance

$ Z_{p,\mathrm{LF}}(\omega)= -j\omega \mu_p \frac{192}{S}\left(\frac{1}{(k_p a)^4} - \frac{3}{320}\right) $

or equivalently, since $\mu_p = \rho h$,

$ Z_{p,\mathrm{LF}}(\omega)= -j\omega \rho h \frac{192}{S}\left(\frac{1}{(k_p a)^4} -\frac{3}{320}\right). $

This is the standard low-frequency reduction of the exact clamped circular plate impedance.

#### Physical interpretation

Using

$ (k_p a)^4 = \frac{\rho h\omega^2}{D}a^4, $

the leading term behaves like

$ Z_{p,\mathrm{LF}}(\omega) \sim \frac{-jK_{\mathrm{eff}}}{\omega} + j\omega M_{\mathrm{eff}}, $

which shows that the plate behaves like a **stiffness-controlled bending element** at very low frequency, with a higher-order inertial correction. This is why the approximation is valid only in the low-frequency flexural regime. 

#### Corresponding TMM element

$ T_{p,\mathrm{LF}}(\omega) =\begin{bmatrix}1 & Z_{p\mathrm{LF}}(\omega) \\0 & 1\end{bmatrix} $

## Clean summary table

| Ref. | Family                        | Model                                      | Parameters                             | Main formula / governing relation                                                                                                                                                                                                                      | TMM form / output                                                                                          | Typical use                                                                                                          |
| ---- | ----------------------------- | ------------------------------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| A1   | Equivalent acoustic impedance | Resistive film                             | $(R_f)                      $          | $(Z_f = R_f)                                                                                                                                                                                                                                         $ | $ (T_f(\omega)=\begin{bmatrix}1 & Z_f \\ 0 & 1\end{bmatrix})         $                                     | Pure damping layer, mesh, resistive sheet                                                                            |
| A2   | Equivalent acoustic impedance | Mass-only film                             | $(M_f) or (\mu, S)           $         | $(Z_f = j\omega M_f = j\omega \dfrac{\mu}{S})                                                                                                                                                                                                       $  | $(T_f(\omega)=\begin{bmatrix}1 & Z_f \\ 0 & 1\end{bmatrix})           $                                    | Limp sheet, inertive foil, thin mass loading                                                                         |
| A3   | Equivalent acoustic impedance | Mass + resistance                          | $(R_f, M_f) or (R_f, \mu, S)  $        | $(Z_f = R_f + j\omega M_f = R_f + j\omega \dfrac{\mu}{S})                                                                                                                                                                                          $   | $(T_f(\omega)=\begin{bmatrix}1 & Z_f \\ 0 & 1\end{bmatrix})            $                                   | Practical generic thin film with damping and inertia                                                                 |
| A4   | Equivalent acoustic impedance | Mass + resistance + stiffness              | $(R_f, M_f, K_f)              $        | $(Z_f = R_f + j\omega M_f + \dfrac{K_f}{j\omega})                                                                                                                                                                                                 $    | $(T_f(\omega)=\begin{bmatrix}1 & Z_f \\ 0 & 1\end{bmatrix})             $                                  | Lumped resonant thin-element surrogate; equivalent membrane-like model                                               |
| B1   | Bulk elastic slab             | Exact elastic slab                         | $(E,\nu,\eta,\rho,L,S)        $        | $(E^*=E(1+j\eta)), (M_L=E^*\dfrac{1-\nu}{(1+\nu)(1-2\nu)}), (c_L=\sqrt{\dfrac{M_L}{\rho}}), (Z_{c,L}=\dfrac{\rho c_L}{S}), (k_L=\dfrac{\omega}{c_L})                                                                                             $     | $(T_s(\omega)=\begin{bmatrix}\cos(k_LL)&jZ_{c,L}\sin(k_LL)\\ j\sin(k_LL)/Z_{c,L}&\cos(k_LL)\end{bmatrix})$ | Full longitudinal propagation in a bulk layer filling the cross-section                                              |
| B2   | Bulk elastic slab             | Thin-slab first-order approximation        | $(E,\nu,\eta,\rho,L,S)        $        | $(E^*=E(1+j\eta)), (M_L=E^*\dfrac{1-\nu}{(1+\nu)(1-2\nu)}), (Z_s(\omega)=j\omega \dfrac{\rho L}{S}), (Y_s(\omega)=j\omega \dfrac{LS}{M_L})                                                                                                      $      | $(T_s(\omega)\approx\begin{bmatrix}1&Z_s\\ Y_s&1\end{bmatrix})            $                                | Low-frequency slab approximation retaining inertia and compliance                                                    |
| B3   | Bulk elastic slab             | Lumped inertive approximation              | $(\rho,L,S), \text{optional} (\eta)  $ | $(Z_s(\omega)=j\omega \dfrac{\rho L}{S}), optionally (Z_s(\omega)=j\omega \dfrac{\rho L}{S}(1+j\eta))                                                                                                                                          $       | $(T_s(\omega)=\begin{bmatrix}1&Z_s\\0&1\end{bmatrix})                     $                                | Mass-only limit of a bulk slab; simplest reduced plug model                                                          |
| C1   | Structural membrane           | Membrane model (full)                      | $(\mu,T,c_d), \text{geometry} (a)    $ | $(\mu \ddot w + c_d \dot w - T\nabla_\perp^2 w = \Delta p)                                                                                                                                                                                    $        | Solve structural problem, then derive $(\Delta p = Z_m(\omega),U)         $                                | Tension-dominated stretched membrane, taut film                                                                      |
| C2   | Structural membrane           | Membrane model (lumped approximation)      | $(\mu,T,a,S), \text{optional} (R_m)  $ | $(Z_m(\omega)\approx R_m + j\omega \dfrac{\mu}{S} + \dfrac{1}{j\omega}\dfrac{C_T T}{a^2 S}), with (M_m=\dfrac{\mu}{S}), (K_m=\dfrac{C_T T}{a^2 S})                                                                                           $         | $(T_m(\omega)=\begin{bmatrix}1&Z_m\\0&1\end{bmatrix})                       $                              | Low-order equivalent impedance of a circular membrane                                                                |
| D1   | Structural flexural plate     | Flexural plate model                       | $(E,\nu,h,\rho,a)             $        | $(D=\dfrac{Eh^3}{12(1-\nu^2)}), (S=\pi a^2), (m_p=\rho h S), (k_p=\left(\dfrac{\rho h,\omega^2}{D}\right)^{1/4}), (x=k_pa),(Z_p(\omega)=-j\omega \dfrac{m_p}{S^2}\left(\dfrac{I_1(x)J_0(x)+J_1(x)I_0(x)}{I_1(x)J_2(x)-J_1(x)I_2(x)}\right))$           | $(T_p(\omega)=\begin{bmatrix}1&Z_p\\0&1\end{bmatrix})                        $                             | Bending-dominated thin clamped plate, stiff foil, polymer disk                                                       |
| D2   | Structural flexural plate     | Low-frequency flexural plate approximation | $(E,\nu,h,\rho,a)$                     | $(D=\dfrac{Eh^3}{12(1-\nu^2)}), (S=\pi a^2), (k_p=\left(\dfrac{\rho h\,\omega^2}{D}\right)^{1/4}), (x=k_pa), (Z_{p,\mathrm{LF}}(\omega)=-j\omega\,\rho h\,\dfrac{192}{S}\left(\dfrac{1}{(k_pa)^4}-\dfrac{3}{320}\right))$                              | $(T_{p,\mathrm{LF}}(\omega)=\begin{bmatrix}1&Z_{p,\mathrm{LF}}\\0&1\end{bmatrix})$                         | Low-frequency asymptotic approximation of the clamped flexural plate; useful well below the first flexural resonance |
