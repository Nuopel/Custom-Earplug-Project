## WBS 2 — Define robust port quantities and matrix conventions

### 2.1 Objective

Define a clear and reusable two-port framework for acoustic matrix inversion, parameter identification, and de-embedding. The objective is to ensure that the quantities extracted from COMSOL are strictly compatible with the transfer-matrix formulation used in the TMM framework.

---

### 2.2 Port definition and state variables

The slab system is modeled as a **two-port acoustic element** placed between an upstream section and a downstream section.

- **Port 1**: upstream face of the slab
- **Port 2**: downstream face of the slab

At each port, the acoustic state is described by:

- $p$: section-averaged acoustic pressure
- $U$: volume velocity through the port section

The reference state vector is therefore

$ \begin{bmatrix}
p \\
U
\end{bmatrix} $

Using $U$ rather than particle velocity $v$ avoids ambiguity when cross-sections change and is directly compatible with standard transfer-matrix formulations and impedance definitions.

---

### 2.3 Transfer-matrix convention

The convention to freeze is

$ \begin{bmatrix}
p_1 \\
U_1
\end{bmatrix}
=
\mathbf{T}
\begin{bmatrix}
p_2 \\
U_2
\end{bmatrix} $

with

$ \mathbf{T} =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix} $

so that

- $p_1 = A p_2 + B U_2$
- $U_1 = C p_2 + D U_2$

This convention means that the matrix propagates the downstream state back to the upstream state.

---

### 2.4 Sign convention

A single sign convention must be imposed for all models and extractions:

- pressures $p_1$ and $p_2$ are scalar acoustic pressures
- $U_1$ and $U_2$ are defined as **positive in the global propagation direction**, from port 1 to port 2

The main requirement is to use the **same orientation everywhere**.

---

### 2.5 Port quantities to extract from COMSOL

For each port, COMSOL quantities must be reduced to 1D equivalent port variables consistent with TMM.

#### Pressure

Use the **section-averaged pressure**

$ \bar{p} = \frac{1}{S}\int_S p \, dS $

#### Volume velocity

Use the **surface-integrated normal velocity**

$ U = \int_S \mathbf{v} \cdot \mathbf{n} \, dS $

If needed, the section-averaged normal velocity is

$ \bar{v}_n = \frac{1}{S}\int_S \mathbf{v}\cdot\mathbf{n}\, dS $

with

$ U = S \, \bar{v}_n $

For the matrix framework, $U$ is the preferred variable.

---

### 2.6 Matrix inversion principle

Once the port convention is fixed, the unknown matrix

$ \mathbf{T} =
\begin{bmatrix}
A & B \\
C & D
\end{bmatrix} $

can be identified from two independent load cases.

For each load case $k$, COMSOL provides the port states

$ \begin{bmatrix}
p_1^{(k)} \\
U_1^{(k)}
\end{bmatrix}=
\mathbf{T}
\begin{bmatrix}
p_2^{(k)} \\
U_2^{(k)}
\end{bmatrix} $

Using two independent downstream states gives four scalar equations, sufficient to determine $A$, $B$, $C$, and $D$ frequency by frequency.

In compact form:

$ \begin{bmatrix}
p_1^{(1)} & p_1^{(2)} \\
U_1^{(1)} & U_1^{(2)}
\end{bmatrix}=
\mathbf{T}
\begin{bmatrix}
p_2^{(1)} & p_2^{(2)} \\
U_2^{(1)} & U_2^{(2)}
\end{bmatrix} $

hence

$ \mathbf{T}=
\begin{bmatrix}
p_1^{(1)} & p_1^{(2)} \\
U_1^{(1)} & U_1^{(2)}
\end{bmatrix}
\left(
\begin{bmatrix}
p_2^{(1)} & p_2^{(2)} \\
U_2^{(1)} & U_2^{(2)}
\end{bmatrix}
\right)^{-1} $

This is the core inversion formula for matrix identification.

---

### 2.7 Condition for a valid inversion

The inversion is valid only if the two downstream state vectors are linearly independent, i.e. if

$ \begin{bmatrix}
p_2^{(1)} & p_2^{(2)} \\
U_2^{(1)} & U_2^{(2)}
\end{bmatrix} $

is invertible.

In practice, the two load cases must produce sufficiently different acoustic terminations.

---

### 2.8 De-embedding principle

If the measured system is

$ \mathbf{T}_{\mathrm{tot}} = \mathbf{T}_{L}\mathbf{T}_{x}\mathbf{T}_{R} $

where $\mathbf{T}_{x}$ is the target subsystem, then

$ \mathbf{T}_{x} = \mathbf{T}_{L}^{-1}\mathbf{T}_{\mathrm{tot}}\mathbf{T}_{R}^{-1} $

This is the basis for removing known air sections from a COMSOL-identified matrix in order to isolate the slab contribution alone.

### 2.8 Conditioning and regularization

In practice, the matrix inversion can become unstable if the two load cases are too similar, or if the extracted COMSOL quantities contain numerical noise. Even if the downstream state matrix is formally invertible, it may be **poorly conditioned**, which leads to large errors in the identified coefficients.

Let

$ \mathbf{X}_2 =
\begin{bmatrix}
p_2^{(1)} & p_2^{(2)} \\
U_2^{(1)} & U_2^{(2)}
\end{bmatrix} $

and

$ \mathbf{X}_1 =
\begin{bmatrix}
p_1^{(1)} & p_1^{(2)} \\
U_1^{(1)} & U_1^{(2)}
\end{bmatrix} $

The direct inversion writes

$ \mathbf{T} = \mathbf{X}_1 \mathbf{X}_2^{-1} $

but this expression should only be used when $\mathbf{X}_2$ is sufficiently well conditioned.

A practical robustness criterion is to monitor the condition number

$ \kappa(\mathbf{X}_2) = \|\mathbf{X}_2\| \, \|\mathbf{X}_2^{-1}\| $

If $\kappa(\mathbf{X}_2)$ becomes too large, the identified matrix may become unreliable.

A more robust approach is to replace the direct inverse by a **regularized inverse** or pseudo-inverse. For example, a Tikhonov-type inversion may be written

$ \mathbf{T}=\mathbf{X}_1
\mathbf{X}_2^{H}
\left(
\mathbf{X}_2 \mathbf{X}_2^{H} + \lambda \mathbf{I}
\right)^{-1} $

where:

- $\mathbf{X}_2^{H}$ is the Hermitian transpose
- $\lambda > 0$ is a regularization parameter
- $\mathbf{I}$ is the identity matrix

This regularization limits the amplification of numerical noise when the two load cases are nearly linearly dependent.

In practice, robustness should be improved by combining:

- sufficiently distinct load cases
- careful port averaging in COMSOL
- condition-number monitoring
- regularized inversion when needed

The regularization must remain weak enough not to distort the physical matrix, but strong enough to suppress purely numerical instabilities.

---

### 2.9 Consistency rules for the framework

The same convention must be used everywhere:

- same port numbering
- same propagation direction
- same sign convention for $U$
- same definition of section-averaged pressure
- same definition of integrated normal flow rate
- same matrix form $ \begin{bmatrix}
  p_1 \\
  U_1
  \end{bmatrix}
  = \mathbf{T}
  \begin{bmatrix}
  p_2 \\
  U_2
  \end{bmatrix} $

This ensures direct comparability between analytical TMM elements, COMSOL-extracted matrices, de-embedded matrices, and reconstructed responses.

---

### 2.10 Validation criteria

The convention is considered validated if:

- the same formulation works for slab, duct, filter, and full cascades
- reassembled matrices reconstruct the original COMSOL responses
- de-embedding of known air sections gives the expected isolated subsystem
- identified matrices remain stable when using different valid load pairs
