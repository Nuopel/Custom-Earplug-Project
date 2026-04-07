# Extracting the 2×2 Transfer Matrix from $(p_1, u_1, p_2, u_2)$

We want the 2-port relation

$$
\begin{pmatrix}
p_1 \\
u_1
\end{pmatrix}
=
\begin{pmatrix}
A & B \\
C & D
\end{pmatrix}
\begin{pmatrix}
p_2 \\
u_2
\end{pmatrix}
$$

that is,

$$
p_1 = A\,p_2 + B\,u_2
\qquad\text{and}\qquad
u_1 = C\,p_2 + D\,u_2
$$

## Important point

A **single configuration** gives only 2 equations, so it is **not enough** to determine the 4 unknowns $A,B,C,D$.

You need **2 independent configurations**.

---

## General extraction with two configurations

Let:

- configuration $a$: $\left(p_1^{(a)}, u_1^{(a)}, p_2^{(a)}, u_2^{(a)}\right)$
- configuration $b$: $\left(p_1^{(b)}, u_1^{(b)}, p_2^{(b)}, u_2^{(b)}\right)$

Then

$$
\begin{pmatrix}
A & B \\
C & D
\end{pmatrix}
=
\begin{pmatrix}
p_1^{(a)} & p_1^{(b)} \\
u_1^{(a)} & u_1^{(b)}
\end{pmatrix}
\begin{pmatrix}
p_2^{(a)} & p_2^{(b)} \\
u_2^{(a)} & u_2^{(b)}
\end{pmatrix}^{-1}
$$

---

## Explicit formulas

Define

$$
\Delta = p_2^{(a)}u_2^{(b)} - p_2^{(b)}u_2^{(a)}
$$

Then:

$$
A = \frac{p_1^{(a)}u_2^{(b)} - p_1^{(b)}u_2^{(a)}}{\Delta}
$$

$$
B = \frac{-p_1^{(a)}p_2^{(b)} + p_1^{(b)}p_2^{(a)}}{\Delta}
$$

$$
C = \frac{u_1^{(a)}u_2^{(b)} - u_1^{(b)}u_2^{(a)}}{\Delta}
$$

$$
D = \frac{-u_1^{(a)}p_2^{(b)} + u_1^{(b)}p_2^{(a)}}{\Delta}
$$

with the condition

$$
\Delta \neq 0
$$

so the two configurations must be linearly independent.

---

## Very convenient special case

### Configuration 1: rigid termination

If the output is rigid:

$$
u_2 = 0
$$

then

$$
p_1 = A\,p_2
\qquad\text{and}\qquad
u_1 = C\,p_2
$$

so

$$
A = \frac{p_1}{p_2}
\qquad\text{and}\qquad
C = \frac{u_1}{p_2}
$$

---

### Configuration 2: pressure-release termination

If the output is pressure-release:

$$
p_2 = 0
$$

then

$$
p_1 = B\,u_2
\qquad\text{and}\qquad
u_1 = D\,u_2
$$

so

$$
B = \frac{p_1}{u_2}
\qquad\text{and}\qquad
D = \frac{u_1}{u_2}
$$

---

## Final minimal summary

With two configurations:

$$
\boxed{
\mathbf{T}
=
\begin{pmatrix}
A & B \\
C & D
\end{pmatrix}
=
\begin{pmatrix}
p_1^{(a)} & p_1^{(b)} \\
u_1^{(a)} & u_1^{(b)}
\end{pmatrix}
\begin{pmatrix}
p_2^{(a)} & p_2^{(b)} \\
u_2^{(a)} & u_2^{(b)}
\end{pmatrix}^{-1}
}
$$

Special simple case:

- **rigid end** $\rightarrow u_2=0$ gives $A$ and $C$
- **pressure-release end** $\rightarrow p_2=0$ gives $B$ and $D$
