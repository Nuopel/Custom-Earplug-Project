# WB8 Example Map

This folder validates thin inserted elements in a duct by comparing:

- TMM models in `acoustmm`
- FEM 2-port exports
- IEC711 / rigid-end acoustic observables

The progression goes from simple lumped impedance films to structural elements.

## Example List

### `A0_Resistive_film_in_duct_iec_or_rigidend.py`

Pure resistive series film:

$
T_f(\omega)=
\begin{bmatrix}
1 & R_f \\
0 & 1
\end{bmatrix}
$

Use this to validate:

- constant series resistance
- FEM interior impedance boundary with real constant impedance
- end-pressure / IL agreement for the simplest case

### `A1_ResistiveMass_film_in_duct_iec_or_rigidend.py`

Resistive + inertive mesh-like film:

$
T_f(\omega)=
\begin{bmatrix}
1 & R_f + j\omega M_f \\
0 & 1
\end{bmatrix}
$

Use this to compare several mesh-style cases against FEM.

### `A2_RKM_in_duct_iec_or_rigidend.py`

Generic lumped film:

$
T_f(\omega)=
\begin{bmatrix}
1 & Z_f(\omega) \\
0 & 1
\end{bmatrix},
\qquad
Z_f(\omega)=R_f + j\omega M_f + \frac{K_f}{j\omega}
$

This is the generic surrogate layer for:

- resistive-dominated films
- mass-dominated films
- resonant lumped films

### `A3_bulksilicone_in_duct_iec_or_rigidend.py`

Bulk elastic slab validation with three TMM levels:

- `ElasticSlab` = exact slab model
- `ElasticSlabThin` = thin-slab approximation
- `ElasticSlabSeries` = inertive-only limit

This covers the bulk/silicone-like inserted element case.

### `A4_membrane_in_duct_iec_or_rigidend.py`

Membrane surrogate based on physical membrane parameters:

$
Z_m(\omega)\approx R_m + j\omega \frac{\mu}{S} + \frac{1}{j\omega}\frac{C_T T}{a^2 S}
$

Implemented through `MembraneSeriesImpedance`.

Current script focus:

- silicone-like membrane surrogate
- IEC711 comparison
- optional FEM overlay

### `A5_flexuralplate_rkm_in_duct_iec_or_rigidend_lossless.py`

Flexural plate comparison in duct:

- exact plate model `D1`
- low-frequency plate approximation `D2`
- comparison to one FEM plate case

This is the main structural plate-in-duct validation script.

### `A6_flexuralplate_rkm_in_duct_iec_or_rigidend_withlosses.py`

Flexural plate case with losses.

Use this once the lossless plate behavior is understood and you want to study damping effects or lossy FEM comparisons.

### `A7_compare_exact_vs_lowfreq_plate_impedance.py`

Direct impedance-level comparison for the flexural plate:

- exact `D1` impedance
- low-frequency `D2` impedance
- alternative low-frequency asymptotic forms

This script works directly on `Z(\omega)` and plots:

- `|Z|`
- `phase(Z)`
- `|Re(Z)|`
- `|Im(Z)|`

It is useful to inspect the validity range of the low-frequency plate approximation before embedding the plate in the full duct system.

## Recommended Order

Run the examples in this order:

1. `A0` for pure resistance
2. `A1` for resistance + mass
3. `A2` for generic `RKM`
4. `A3` for bulk slab
5. `A4` for membrane surrogate
6. `A5` for exact vs low-frequency flexural plate in duct
7. `A6` for lossy flexural plate
8. `A7` for direct impedance-level inspection of plate formulas

## Interpretation

The folder now covers three modeling layers:

- lumped impedance films: `A0`, `A1`, `A2`
- real structural surrogates: `A3`, `A4`
- flexural plate models: `A5`, `A6`, `A7`

The main validation logic is:

- first verify TMM vs FEM when both use the same impedance law
- then move toward structural models
- then compare exact structural models with their simplified approximations
