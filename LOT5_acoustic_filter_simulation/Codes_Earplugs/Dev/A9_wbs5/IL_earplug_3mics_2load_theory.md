# Analytical model and determination of earplug acoustic properties

This document merges the notes on the analytical insertion loss model and the determination of earplug acoustic properties from the three-microphone impedance tube method. It consolidates the reconstructed equations, the interpretation of the model, and the practical measurement procedure.
It originate from `An impedance tube technique for estimating the insertion
loss of earplugs` K. Carillo

## 1. Analytical model of insertion loss

The insertion loss (IL) is defined as the difference in sound pressure level at the eardrum between the open and occluded earcanal configurations. The model uses one-dimensional plane-wave propagation in a cylindrical earcanal and includes viscothermal losses through the equivalent complex wavenumber $k_{eq}^{EC}$.

### 1.1 Open earcanal

For the open earcanal, the pressure field is written as a superposition of forward and backward waves, including multiple reflections between the entrance and the tympanic membrane:

$
p_{open}(x) =
A_{open}
\frac{
e^{-j k_{eq}^{EC} x}+
R_{TM} e^{-2 j k_{eq}^{EC} l_{EC}} e^{j k_{eq}^{EC} x}
}{
1 - R_R R_{TM} e^{-2 j k_{eq}^{EC} l_{EC}}
}
$

The entrance-wave amplitude is:

$
A_{open} = 2 P_0 \frac{Z_{eq}^{EC}}{Z_R + Z_{eq}^{EC}}
$

where $Z_R$ is the radiation impedance at the canal entrance and $Z_{TM}$ is the eardrum impedance, both rewritten through the reflection coefficients $R_R$ and $R_{TM}$.

### 1.2 Occluded earcanal

In the occluded case, the earplug is represented by two acoustic quantities: the transmission coefficient $\tau_{EP}$ and the medial reflection coefficient $R_{EP}$. The pressure field becomes:

$
p_{occl}(x) =
A_{occl}
\frac{
e^{-j k_{eq}^{EC} x}
+R_{TM} e^{-2 j k_{eq}^{EC} l_{EC}} e^{j k_{eq}^{EC} x}
}{
1 - R_{EP} R_{TM} e^{-2 j k_{eq}^{EC}(l_{EC} - l_{ID})}
}
$

with

$
A_{occl} = \tau_{EP} P_0
$

and $l_{ID}$ the insertion depth of the earplug from the canal entrance.

### 1.3 IL decomposition

Evaluated at the eardrum, the model leads to a decomposition of insertion loss into an earplug transmission term and a cavity/reflection term:

$
\mathrm{IL} = TL_{EP} + IL_c
$

with

$
TL_{EP} = -20 \log_{10} |\tau_{EP}|
$

and

$
IL_c =
20 \log_{10}
\left|
\frac{
2 \dfrac{Z_{eq}^{EC}}{Z_R + Z_{eq}^{EC}}
\cdot
\left(1 - R_{EP} R_{TM} e^{-2 j k_{eq}^{EC}(l_{EC} - l_{ID})}\right)^{-1}
}{
1 - R_R R_{TM} e^{-2 j k_{eq}^{EC} l_{EC}}
}
\right|^{-1}
$

This decomposition is useful because it separates attenuation caused by transmission through the earplug from attenuation caused by cavity resonances and reflections inside the occluded earcanal.

### 1.4 Physical interpretation

The cavity term $IL_c$ can be negative at low frequency, which corresponds to the occlusion-effect regime where the rigid medial side of the earplug increases low-frequency pressure in the cavity. Around the open-ear quarter-wavelength resonance, the same term can become positive because the earplug suppresses the resonance that would otherwise amplify pressure at the eardrum.

From a design viewpoint, improving low-frequency attenuation requires reducing the acoustic impedance seen at the medial side of the earplug so that the cavity term is less detrimental and the total IL is driven more directly by the transmission loss $TL_{EP}$.

## 2. Determination of earplug acoustic properties

The second part concerns how to determine the two earplug properties required by the IL model, namely $R_{EP}$ and $\tau_{EP}$, from impedance tube measurements. The measurement is performed with the earplug mounted in a sample holder reproducing the earcanal geometry.

### 2.1 Sample holder and cavity correction

The impedance tube measures the transfer matrix of the combined system formed by the earplug and the residual cavity behind it, noted $T^{EP,SH}$. If the earplug insertion depth $l_{ID}$ is known, the transfer matrix of the earplug alone is recovered by removing the cavity contribution:

$
T^{EP} = T^{EP,SH} \cdot (T^{cav})^{-1}
$

where $T^{cav}$ is the standard transfer matrix of an air cavity of length $l_{cav} = l_{SH} - l_{ID}$.

If $l_{ID}$ is not known, the notes indicate that one may bypass this correction and directly use the acoustic properties extracted from the combined sample-holder system in the IL equations, while replacing $l_{ID}$ by $l_{SH}$.

### 2.2 Extraction of $R_{EP}$ and $\tau_{EP}$

Once $T^{EP}$ is known, the two acoustic coefficients are obtained analytically from the matrix coefficients and the equivalent earcanal impedance $Z_{eq}^{EC}$:

$
R_{EP} =
\frac{
T^{EP}_{11} + T^{EP}_{12}/Z_{eq}^{EC} - T^{EP}_{21} Z_{eq}^{EC} - T^{EP}_{22}
}{
T^{EP}_{11} + T^{EP}_{12}/Z_{eq}^{EC} + T^{EP}_{21} Z_{eq}^{EC} + T^{EP}_{22}
}
$

$
\tau_{EP} =
\frac{
2 e^{j k_{eq}^{EC} l_{ID}}
}{
T^{EP}_{11} + T^{EP}_{12}/Z_{eq}^{EC} + T^{EP}_{21} Z_{eq}^{EC} + T^{EP}_{22}
}
$

Because the earplug may be asymmetric, two downstream acoustic loads are required in order to fully determine the transfer matrix.

## 3. Three-microphone method

The transfer matrix is identified from measurements made with three microphones: two upstream of the sample holder and one downstream near a rigid termination. Two different downstream lengths are used, indexed by $i = a, b$, in order to generate two independent load conditions.

### 3.1 Geometric parameters

The main geometric parameters used in the formulation are:

- $l_1$: distance between microphones 1 and 2.
- $l_{2,i}$: distance from microphone 2 to the sample entrance for load $i$.
- $l_{3,i}$: distance from the sample exit to microphone 3 for load $i$.
- $l_{SH}$: sample-holder length.

### 3.2 Transfer matrix from two loads

Using the two load configurations, the transfer matrix of the earplug-plus-holder system is assembled as follows:

$
T^{EP,SH}
=
\frac{1}{p^a(l_{SH})v^b(l_{SH}) - p^b_{SH}v^a(l_{SH})}
\begin{bmatrix}
p^a(0)v^b(l_{SH}) - p^b(0)v^a(l_{SH})
&
p^b(0)p^a(l_{SH}) - p^a(0)p^b(l_{SH})
\\
v^a(0)v^b(l_{SH}) - v^b(0)v^a(l_{SH})
&
p^a(l_{SH})v^b(0) - p^b(l_{SH})v^a(0)
\end{bmatrix}
$

All entries are built from the boundary pressures and velocities reconstructed for each load from the microphone transfer functions.

## 4. Eq. (13): reconstruction of pressures and velocities

For each load $i \in \{a,b\}$, the four boundary quantities are reconstructed from the measured transfer functions $H_{12,i} = p_2^i/p_1^i$ and $H_{13,i} = p_3^i/p_1^i$.
Note that there is an error in the equation of the pressure which miss a $j$. This error is not missing on the original and reference paper : `Complement to standard method for measuring
normal incidence sound transmission loss with three microphones` Y. Salissou
### 4.1 Pressure at sample entrance

The upstream pressure at the sample entrance is:

$
p^i(0) =
-2 j e^{j k_{eq}^{tube} l_{2,i}}
\frac{
H_{12,i}\sin\!\left(k_{eq}^{tube}(l_1+l_{2,i})\right)
-\sin\left(k_{eq}^{tube} l_{2,i}\right)
}{
H_{12,i} e^{-j k_{eq}^{tube} l_{1}} - 1
}
$

An equivalent exponential-form expression also appears in the notes, but this sine-form version is the cleaner reconstructed form.

### 4.2 Velocity at sample entrance

The upstream velocity contains the key modification introduced by the method, namely the correction by the area ratio $S_{tube}/S_{eff}$ to preserve volume-flow continuity at the section change:

$
v^i(0) =
\frac{S_{tube}}{S_{eff}}
\cdot
\frac{1}{Z_{eq}^{tube}}
2 e^{j k_{eq}^{tube} l_{2,i}}
\frac{
H_{12,i}\cos\left(k_{eq}^{tube}(l_1+l_{2,i})\right)- \cos\left(k_{eq}^{tube} l_{2,i}\right)
}{H_{12,i} e^{-j k_{eq}^{tube} l_{1}} - 1}
$

This correction is the central novelty emphasized in the notes because it allows the method to account for the change between the large upstream tube and the effective radiating area of the earplug entrance.

### 4.3 Pressure at sample exit

The downstream pressure at the sample-holder exit is reconstructed from the downstream microphone relation using the equivalent earcanal wavenumber $k_{eq}^{EC}$. The attached notes contain a partially reconstructed expression, written as:

$
p^i(l_{SH}) =
-2 je^{j k_{eq}^{tube} l_{2,i}} \cdot
\frac{H_{13,i}\,\sin\!\left(k_{eq}^{tube} l_{1}\right)\cos\!\left(k_{eq}^{EC} l_{3,i}\right)
}{
H_{12,i} e^{-j k_{eq}^{tube} l_{1}} - 1
}
$

The notes also indicate that the Markdown source originally garbled this part of Eq. (13), so this expression should be treated as the reconstructed form provided in the attached material rather than a checked transcription from the original paper.

### 4.4 Velocity at sample exit

The exit velocity is then obtained directly from the downstream characteristic impedance:

$
v^i(l_{SH}) = \frac{1}{Z_{eq}^{tube}}2 e^{j k_{eq}^{tube} l_{2,i}}
\frac{
H_{13,i}\sin\!\left(k_{eq}^{tube}l_1\right)
\sin\left(k_{eq}^{tube} l_{3,i}\right)
}{
H_{12,i} e^{-j k_{eq}^{tube} l_{1}} - 1
}
$

## 5. Effective area correction

A major point in the method is the definition of the effective area $S_{eff}$, which depends on whether the earplug is flush-mounted or protrudes outside the earcanal.

| Case                  | Effective area                                       | Meaning                                                         |
|:--------------------- |:---------------------------------------------------- |:--------------------------------------------------------------- |
| Flush-mounted earplug | $S_{EC} = \pi r_{EC}^2$                              | Standard continuity with the earcanal cross-section .           |
| Protruding earplug    | Total exposed lateral surface of the protruding part | Captures the real radiating area of the exposed plug geometry . |

Neglecting this correction can produce large errors for soft foam earplugs, with deviations reported above 10 dB in the notes, whereas the effect is much smaller for stiffer silicone earplugs.

## 6. Use of different equivalent media parameters

The upstream and downstream sections do not use the same equivalent propagation parameters because their radii differ. The notes explicitly distinguish the large tube section from the narrow earcanal-like downstream section.

| Section            | Radius               | Wavenumber      | Impedance       |
|:------------------ |:-------------------- |:--------------- |:--------------- |
| Upstream section   | $r_{tube} = 14.5$ mm | $k_{eq}^{tube}$ | $Z_{eq}^{tube}$ |
| Downstream section | $r_{EC}$             | $k_{eq}^{EC}$   | $Z_{eq}^{EC}$   |

This distinction is essential because viscothermal losses are stronger in the narrow downstream section, so using the same propagation constants everywhere would distort the reconstruction and the extracted earplug properties.

## 7. Consolidated interpretation

Taken together, the analytical IL model and the three-microphone measurement framework form a complete chain: first determine $R_{EP}$ and $\tau_{EP}$ experimentally, then inject them into the one-dimensional earcanal model to predict insertion loss. The decomposition into $TL_{EP}$ and $IL_c$ is especially useful because it shows separately what comes from the material/transmission behavior of the earplug and what comes from acoustic interaction with the occluded cavity.

The most distinctive experimental contribution in the merged notes is the $S_{tube}/S_{eff}$ correction, which adapts the standard transfer-matrix measurement to earplugs with protruding geometries and makes the model much more realistic for soft deformable plugs.
