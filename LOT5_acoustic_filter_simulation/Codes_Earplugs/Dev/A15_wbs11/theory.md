# Dissipative Ducts

## Introduction

Dissipative ducts play a key role in noise reduction across a wide range of industrial and commercial applications, from HVAC systems to gas turbine and jet engine test benches. Modelling their acoustic behaviour within the **Transfer Matrix Method (TMM)** framework allows the efficient design and optimisation of dissipative silencers.

In this note, the lined-duct formulas used  are taken from **Munjal, Section 6.4**. More precisely:

1. the rectangular closed-form approximation follows **Eq. 6.59**,
2. the rectangular attenuation relation follows **Eq. 6.64**,
3. the circular closed-form approximation follows **Eq. 6.62**,
4. the circular attenuation relation follows **Eq. 6.65**.

***

## General Case: Rectangular Duct with Locally Reacting Walls

Consider a rectangular duct of cross-section $(b \times h)$ whose walls present uniform normal impedances $Z_{w,x}$ and $Z_{w,y}$. Applying the wave equation together with impedance boundary conditions yields the following transcendental eigenvalue equations for the transverse wavenumbers $k_x$ and $k_y$:

$
\frac{\cot\!\left(k_x b/2\right)}{k_x} = -j\,\frac{Z_{w,x}}{\rho_0 c_0}\,\frac{1}{k_0}
\qquad
\frac{\cot\!\left(k_y h/2\right)}{k_y} = -j\,\frac{Z_{w,y}}{\rho_0 c_0}\,\frac{1}{k_0}
$

The axial wavenumber associated with mode $(m,n)$ is then:

$
k_{z,m,n} = \left(k_0^2 - k_{x,m}^2 - k_{y,n}^2\right)^{1/2}
$

The general pressure field is a superposition of all modes $(m,n)$. In practice, this full modal sum is rarely tractable analytically and the problem is reduced to the dominant mode, as detailed in the following section.

***

## Plane-Wave Approximation — Mode (0,0)

For most engineering configurations, only the **least attenuated (fundamental) mode** $(0,0)$ is retained. The mode subscripts are therefore dropped throughout the remainder of this chapter. Pressure and axial particle velocity in the duct reduce to:

$
p(z) = A\,e^{+j k_z z} + B\,e^{-j k_z z}
$

$
u_z(z) = \frac{k_z}{\rho_0 c_0 k_0} \Bigl(A\,e^{+j k_z z} - B\,e^{-j k_z z}\Bigr)
$

where $k_x$ and $k_y$ are the **first roots** of the eigenvalue equations above, and:

$
k_z = \left(k_0^2 - k_x^2 - k_y^2\right)^{1/2}
$

**Single lined wall (one pair of absorbing walls).** If only the walls normal to $y$ are lined and those normal to $x$ are rigid, then $Z_{w,x} \to \infty$, giving $k_x = 0$ and:

$
k_z = \left(k_0^2 - k_y^2\right)^{1/2}
$

***

## Closed-Form Approximation for $k_x$ and $k_y$

Following **Munjal, Section 6.4**, the transcendental equations can be approximated in closed form. The expression implemented in `A0_minimal_lined_rectangular_duct.py` is the one used from **Eq. 6.59**:

$
k_x^2 = \frac{2.47 + Q \;\pm\; \sqrt{(2.47 + Q)^2 - 1.87\,Q}}{0.38} \cdot \frac{4}{b^2}
$

$
Q = j\,k_0\,\frac{b}{2}\,\frac{\rho_0 c_0}{Z_{w,x}}
$

This yields two complex roots; the physically relevant one is that which **minimises** $\operatorname{Im}(k_z)$, i.e. the root producing the lowest attenuation. The same procedure applies to $k_y$ using $h$ and $Z_{w,y}$.

The **propagative attenuation coefficient** then follows from the axial-wavenumber relation used in **Munjal Eq. 6.64**:

$
\alpha_0 = -\operatorname{Im}\!\left\{ k_0^2 - k_{x,0}^2 - k_{y,0}^2 \right\}^{1/2}
$

***

## Circular Duct with Locally Reacting Wall

For a circular duct of radius $r_0$ with a uniform locally reacting wall impedance $Z_w$, the pressure field is written in cylindrical coordinates. Retaining the axisymmetric fundamental mode only, the axial wavenumber is:

$
k_z = \left(k_0^2 - k_r^2\right)^{1/2}
$

where $k_r$ is the transverse radial wavenumber associated with the lined boundary condition at $r=r_0$. Following **Munjal, Section 6.4**, the exact characteristic equation is replaced by the closed-form approximation used in `A0_minimal_lined_circular_duct.py`; this is the expression used from **Eq. 6.62**:

$
q = (k_0 r_0)\,\frac{\rho_0 c_0}{Z_w}
$

$
(k_r r_0)^2_{\pm}
=
\frac{
96 + 36j\,q \pm \sqrt{9216 + 2304j\,q - 912\,q^2}
}{
12 + j\,q
}
$

Hence

$
k_{r,\pm}^2 = \frac{(k_r r_0)^2_{\pm}}{r_0^2}
\qquad\text{and}\qquad
k_{z,\pm} = \left(k_0^2 - k_{r,\pm}^2\right)^{1/2}
$

The physically relevant branch is chosen as the one that gives the lower attenuation:

$
\alpha_\pm = -\operatorname{Im}(k_{z,\pm})
\qquad\Rightarrow\qquad
k_z =
\begin{cases}
k_{z,+}, & \alpha_+ \le \alpha_- \\
k_{z,-}, & \alpha_- < \alpha_+
\end{cases}
$

This is the model compared against FEM in `A1_compare_lined_circular_duct_fem.py`. As for the rectangular case, the attenuation constant of the lined circular duct follows **Munjal Eq. 6.65**:

$
\alpha_0 = -\operatorname{Im}(k_z)
$

***

## Transfer Matrix Formulation

Combining the expressions for $p(z)$ and $u_z(z)$, the transfer matrix of a dissipative duct element of length $l$ is:

$
\begin{bmatrix} p(0) \\ v_z(0) \end{bmatrix}
=
\begin{bmatrix}
\cos(k_z l) & j\,Y\sin(k_z l) \\
\dfrac{j\sin(k_z l)}{Y} & \cos(k_z l)
\end{bmatrix}
\begin{bmatrix} p(l) \\ v_z(l) \end{bmatrix}
$

with the specific admittance:

$
Y = \frac{k_0}{k_z}\,\frac{c_0}{S}
$

where $S$ is the duct cross-sectional area: $S=bh$ for a rectangular duct and $S=\pi r_0^2$ for a circular duct.

This matrix is **structurally identical** to that of a rigid uniform tube. The only modification is that $k_z$ is now complex, its imaginary part directly encoding the acoustic energy dissipated by the lining.

***

## Wall Impedance from the Porous Layer

The wall impedance $Z_w$ is computed by treating the lining as a homogeneous porous layer of thickness $l$ backed by a rigid wall. Starting from the TMM input impedance formula:

$
Z_\mathrm{in} = \frac{T_{11}\,Z_\mathrm{out} + T_{12}}{T_{21}\,Z_\mathrm{out} + T_{22}}
$

and taking the rigid-backing limit $Z_\mathrm{out} \to \infty$, one obtains:

$
Z_w = \frac{T_{11}}{T_{21}} = -j\,Z_c\cot(k_c\,l)
$

where $Z_c$ and $k_c$ are respectively the characteristic impedance and the complex wavenumber of the porous material.

***

## Porous Material Model — Miki

The characteristic impedance $Z_c$ and wavenumber $k_c$ are described using the **Miki model**, a well-established refinement of the Delany–Bazley model, parameterised solely by the **airflow resistivity** $\sigma$

***

## Computation Pipeline

Modelling a dissipative duct within the TMM reduces to the following sequential steps:

1. Given $\sigma$ and lining thickness $l$, compute $Z_c$ and $k_c$ via the Miki model.
2. Evaluate the wall impedance $Z_w = -jZ_c\cot(k_c l)$.
3. For a rectangular duct, solve for transverse wavenumbers $k_x$, $k_y$ using the closed-form $Q$-approximation, selecting the root that minimises $\operatorname{Im}(k_z)$.
4. For a circular duct, solve for the radial transverse wavenumber $k_r$ using the two-branch closed-form expression and retain the branch with minimum attenuation.
5. Compute the complex axial wavenumber $k_z$ from $k_0^2$ minus the retained transverse contribution.
6. Assemble the transfer matrix and cascade it with the remaining system elements.

This approach integrates dissipative effects into the TMM with minimal overhead, preserving the compact matrix structure and making it directly suitable for silencer design and optimisation.

***

## Equivalent-Slab Interpretation

The previous sections assume that the lining is described directly by a wall impedance $Z_w$, typically obtained from a porous model such as Miki. A second route, used in `A2_Slab_duct_eq.py`, is to start from the **transfer matrix of a slab section itself** and retrieve an equivalent homogeneous medium described by an effective wavenumber $k_{eq}$ and an effective characteristic impedance $Z_{c,eq}$.

Assume that a slab of known thickness $l_s$ has already been isolated as a two-port in the $[p,U]$ basis:

$
\mathbf{T}_s =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix}
$

If this slab is represented by an equivalent homogeneous section, its transfer matrix must be of the standard form:

$
\mathbf{T}_{eq} =
\begin{bmatrix}
\cos(k_{eq} l_s) & j\,Z_{c,eq}\sin(k_{eq} l_s) \\
\dfrac{j}{Z_{c,eq}}\sin(k_{eq} l_s) & \cos(k_{eq} l_s)
\end{bmatrix}
$

Comparing both matrices gives the retrieval formulas used in `A2_Slab_duct_eq.py`:

$
\cos(k_{eq} l_s) = \frac{A + D}{2}
$

$
k_{eq} = \frac{1}{l_s}\arccos\!\left(\frac{A+D}{2}\right)
$

$
Z_{c,eq} = \sqrt{\frac{B}{C}}
$

Because $\arccos(\cdot)$ and the square root are multi-valued, the retrieved quantities are not unique unless the branch is tracked continuously with frequency. In practice, the physically meaningful branch is selected by continuity and by checking that the reconstructed matrix

$
\mathbf{T}_{eq}(k_{eq}, Z_{c,eq})
$

matches the original slab matrix as closely as possible over the full frequency range.

An important point is that the retrieved $k_{eq}$ and $Z_{c,eq}$ are **effective section parameters**, not necessarily intrinsic bulk material constants. They describe the acoustic behaviour of the slab as seen inside the duct configuration from which the matrix was extracted.

***

## Recovering Intrinsic Medium Parameters

Once $k_{eq}$ and $Z_{c,eq}$ have been identified, they can be converted back into effective medium properties for a slab cross-sectional area $S_s$. The first step is to recover the **specific** characteristic impedance:

$
z_{c,eq} = Z_{c,eq}\,S_s
$

Then the equivalent density and bulk modulus are obtained from the one-dimensional relations:

$
\rho_{eq} = \frac{z_{c,eq}\,k_{eq}}{\omega}
\qquad
K_{eq} = \frac{\omega\,z_{c,eq}}{k_{eq}}
$

From these, one reconstructs a consistent intrinsic wavenumber and specific characteristic impedance:

$
k_m = \omega\sqrt{\frac{\rho_{eq}}{K_{eq}}}
\qquad
z_{c,m} = \sqrt{\rho_{eq}K_{eq}}
$

As in the previous step, sign and branch consistency must be enforced so that $k_m$ and $z_{c,m}$ remain aligned with the originally retrieved $(k_{eq}, z_{c,eq})$.

***

## Wall Impedance from Retrieved Slab Parameters

After recovering the equivalent medium of the slab, the lining wall impedance is obtained exactly as for a rigid-backed layer, but now using the retrieved medium instead of a prescribed porous law. For a slab thickness $t$:

$
Z_w = -j\,z_{c,m}\cot(k_m t)
$

This is the key bridge between the **slab description** and the **lined-duct description**:

1. extract the slab matrix,
2. retrieve $k_{eq}$ and $Z_{c,eq}$,
3. recover $k_m$ and $z_{c,m}$,
4. compute $Z_w$,
5. inject this $Z_w$ into the lined circular duct model to compute $k_z$ and the TMM matrix of the lined duct.
