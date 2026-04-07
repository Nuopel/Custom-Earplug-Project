---
# 2. Work Breakdown Structure (WBS)

---

## WBS 1 — Clarify and demonstrate the slab-model discrepancy

### 1.1 Objective

Establish clearly why the initial COMSOL and TMM results differ.

### 1.2 Tasks

* [x] Build the reference configuration: air + silicone slab + air cavity
* [x] Run COMSOL with fixed lateral constraint
* [x] Run COMSOL without fixed lateral constraint
* [x] Run the corresponding TMM slab model
* [x] Extract pressure at the end of the cavity
* [x] Extract velocity or volume velocity at the end of the cavity
* [x] Extract pressure at slab entrance and slab exit
* [x] Compute IL if relevant
* [x] Compare the three cases on the same plots
* [x] Write a short physical interpretation of the discrepancy

### 1.3 Deliverables

* [x] Comparison plot: $p_{\text{end}}(f)$

* [x] Comparison plot: $v_{\text{end}}(f)$ or $U_{\text{end}}(f)$

* [x] Comparison plot: $IL(f)$

* [x] Short technical note:
  
  * [x] Why standard TMM is not equivalent to fixed-constraint COMSOL
  * [x] Why fixed walls tend to overestimate low-frequency IL

### 1.4 Validation criteria

* [x] Discrepancy between the three cases is clearly visible
* [x] Explanation is physically consistent with boundary-condition reasoning

---

## WBS 2 — Define robust port quantities and matrix conventions

### 2.1 Objective

Prepare a clean and reusable port-based framework for inversion and de-embedding.

### 2.2 Tasks

* [x] Define the two acoustic ports of the slab model

* [x] Decide the state variables to use at each port
  
  * [x] Section-averaged pressure
  * [x] Averaged normal velocity
  * [x] Volume velocity

* [x] Fix the transfer-matrix convention

* [x] Fix the sign convention for port variables

* [x] Document all conventions in the framework

Reference convention to freeze explicitly:
$
\begin{bmatrix}
p_1 \\
U_1
\end{bmatrix}
=
\mathbf{T}
\begin{bmatrix}
p_2 \\
U_2
\end{bmatrix}
$

### 2.3 Deliverables

* [x] Port-definition note
* [x] Transfer-matrix convention note

### 2.4 Validation criteria

* [x] Same convention works for slab, duct, filter, and merged systems
* [x] Reconstructed responses from matrices are self-consistent

---

## WBS 3 — Equivalent slab identification by two-load inversion

### 3.1 Objective

Estimate the slab as a **frequency-dependent equivalent two-port** directly from COMSOL.

### 3.2 Tasks

* [x] Define two distinct load cases for the slab system
  
  * [x] Rigid termination
  * [x] Open / radiation-like / non-rigid termination

* [x] Run COMSOL for both loads

* [x] Extract upstream and downstream port states

* [x] Solve for the equivalent matrix

* [x] Check numerical conditioning of the inversion

* [x] Add diagnostic plots for matrix coefficients

* [x] Add determinant tracking

* [x] Add sensitivity / conditioning diagnostics if needed

Equivalent matrix:
$
\mathbf{T}_{\text{slab}}(f)=
\begin{bmatrix}
A(f) & B(f) \\
C(f) & D(f)
\end{bmatrix}
$

### 3.3 Deliverables

* [x] Equivalent slab matrix versus frequency

* [x] Inversion notebook / script

* [x] Comparison between:
  
  * [x] COMSOL direct result
  * [x] Equivalent-matrix reconstruction
  * [x] Naive analytical slab TMM

### 3.4 Validation criteria

* [x] Reconstructed pressure matches COMSOL closely
* [x] Reconstructed velocity matches COMSOL closely
* [x] Equivalent matrix is physically plausible
* [x] Equivalent matrix is numerically stable

---

## WBS 4 — De-embedding of surrounding air sections

### 4.1 Objective

Isolate the slab response from the surrounding air domains.

### 4.2 Tasks

* [x] Identify upstream air section in COMSOL
* [x] Identify downstream air section in COMSOL
* [x] Represent both air sections with TMM blocks
* [x] Implement de-embedding / uncascade operation
* [x] Remove surrounding air sections from the total equivalent matrix
* [x] Check invariance versus chosen air-section lengths

Target operation:
$
\mathbf{T}_{\text{slab}}
=
\mathbf{T}*{\text{total}}
\mathbf{T}_{\text{air,down}}^{-1}
$

### 4.3 Framework development

* [x] Add matrix uncascade operator
* [x] Add inverse-cascade operation between TMM objects
* [x] Add slab-only extraction helper

### 4.4 Deliverables

* [x] De-embedding method note
* [x] Framework implementation
* [x] Slab-only equivalent matrix after air removal

### 4.5 Validation criteria

* [x] De-embedded slab response is stable
* [x] Reconstructed full system matches original COMSOL
* [x] Reconstructed full system matches TMM cascade

---

## WBS 5 — IL-focused identification using Carrillo-style method

### 5.1 Objective

Implement a reduced IL-estimation workflow based on the Carrillo-style three-microphone method, using COMSOL as the primary physical model for the slab because the current 1D TMM slab does not capture the fixed border condition.

The aim is not to replace the previous WBS3 work, but to build a lighter identification route focused on the quantities needed for IL:

* identify the earplug/slab-plus-holder response from two COMSOL load cases
* recover the transmission and reflection terms needed by the reduced IL formula
* compare the reduced IL prediction against the full two-load identification already obtained in WBS3

### 5.2 Tasks

* [x] Review the Carrillo-inspired method

* [x] Build a minimal COMSOL identification bench for the slab / sample-holder system

* [x] Represent the slab with the required fixed border condition in COMSOL

* [x] Define and parameterize the three microphone positions:
  
  * mic 1 upstream
  * mic 2 upstream
  * mic 3 downstream near the rigid end

* [x] Run two downstream rigid-back load cases with different lengths:
  
  * [ ] load `a`
  * [ ] load `b`

* [x] Export complex pressure at microphones 1, 2, and 3 for both load cases

* [x] Implement Python post-processing of the COMSOL microphone data

* [x] Reconstruct the boundary states from the three-microphone data using the WBS5 formulas

* [x] Identify `T_EP,SH` from the two load cases

* [x] Remove the residual holder cavity contribution if needed to recover `T_EP`

* [x] Extract reduced earplug/slab quantities:
  
  * [x] medial reflection term `R_EP`
  * [x] transmission term `tau_EP`

* [x] Compute IL from the reduced parameter set

* [x] Compare against:
  
  * [x] Full two-load matrix inversion from WBS3 / A8
  * [x] Naive slab TMM
  * [x] COMSOL reference IL when available

* [ ] Document assumptions, conventions, and limits of the reduced method

### 5.3 Deliverables

* [x] Minimal COMSOL three-microphone identification model
* [x] Exported microphone datasets for load `a` and load `b`
* [x] Python implementation of the WBS5 reduced identification
* [x] Recovered `T_EP,SH` and, if applicable, `T_EP`
* [x] Extracted `R_EP` and `tau_EP`
* [x] Side-by-side IL comparison versus WBS3, naive TMM, and COMSOL reference
* [ ] Note on assumptions, sign conventions, and limitations

### 5.4 Validation criteria

* [x] The COMSOL identification bench is stable for two distinct load cases
* [x] The reconstructed boundary states are consistent with the exported COMSOL response
* [x] The extracted `tau_EP` is consistent with the previous WBS3 / A8 result
* [x] Reduced IL matches the full two-load inversion closely enough for design use
* [x] The gap between reduced IL and the full inversion is quantified over frequency
* [ ] The limits due to plane-wave assumptions and geometry conventions are clearly identified

---

## WBS 6 — Filter-only modeling: straight lossy duct

### 6.1 Objective

Create and validate a first reduced model of the filter.

### 6.2 Tasks

* [x] Model the filter as a straight small duct
* [x] Include viscous / thermoviscous losses
* [x] Build equivalent configuration in COMSOL
* [x] Build equivalent configuration in TMM
* [x] Compare transfer response
* [x] Compare end pressure

### 6.3 Deliverables

* [x] Straight-duct filter model
* [x] COMSOL vs TMM comparison plots
* [ ] Technical note on losses and approximation limits

### 6.4 Validation criteria

* [x] Reduced model reproduces COMSOL trends with acceptable accuracy

---

## WBS 7 — Filter-only modeling: variable-radius duct

### 7.1 Objective

Refine the filter model to account for geometric transitions.

### 7.2 Tasks

* [x] Model a variable-radius duct
* [x] Implement corresponding TMM element(s)
* [x] Compare with COMSOL
* [x] Compare IL under IEC loading
* [x] Assess the effect of area discontinuities

### 7.3 Deliverables

* [x] Variable-radius filter model
* [x] Comparison plots
* [x] Note on geometry-transition effects

### 7.4 Validation criteria

* [x] Improved model is better than straight-duct model when geometry variation matters

---

## WB8 — Validate films / impedance elements

### 8.1 Objective

Validate the correspondence between **TMM** and **COMSOL** for thin acoustic elements inserted in the duct, from simple lumped impedance films to more physical structural models.

### 8.2 Tasks

* [x] Define the validation sequence from simplest to most physical elements
* [x] Introduce a **series impedance film** element in TMM
* [x] Define matching **film / impedance representations** in COMSOL
* [x] Validate **pure resistive films**
* [x] Validate **resistive + inertive films**
* [x] Validate **generic R-K-M films**
* [x] Compare **bulk elastic slab** models in TMM and COMSOL
* [x] Compare **structural membrane** models and lumped surrogates
* [x] Compare **structural flexural plate** models and lumped surrogates
* [x] Compare impact on **IL**
* [x] Compare **transfer function / pressure ratio**
* [x] Compare **resonance frequency shifts**
* [x] Compare **bandwidth / damping**
* [x] Compare **equivalent impedance** when relevant

### 8.3 Deliverables

* [x] Film-impedance module in TMM
* [x] Matching COMSOL film / impedance implementations
* [x] Validation set for **pure R**, **R+M**, and **R-K-M** films
* [x] Comparison set for **slab**, **membrane**, and **plate** cases
* [x] Comparison plots for IL, transfer response, resonance shift, and damping
* [x] Interpretation note on the validity range of lumped surrogates

### 8.4 Validation criteria

* [x] TMM and COMSOL agree when the same impedance law is imposed
* [x] Pure resistive and generic lumped-film cases are reproducible in both models
* [x] The role of resistance, inertance, and compliance is clearly identified
* [x] The low-frequency validity range of slab approximations is identified
* [x] The first-mode behavior of membrane / plate elements can be approximated by lumped surrogates over a limited band
* [ ] Suitable parameter ranges are identified for each element family
* [x] The limits of lumped-film representations versus full structural models are clearly established

---

## WBS 9 — Add foam in the filter

### 9.1 Objective

Introduce porous damping inside the filter.

### 9.2 Tasks

* [x] Model foam in COMSOL with an appropriate porous / equivalent-fluid approach
* [x] Implement corresponding TMM porous block
* [x] Compare IL
* [x] Compare resonance smoothing
* [x] Compare broadband attenuation
* [x] Study sensitivity to porous parameters

### 9.3 Deliverables

* [x] Foam model
* [x] COMSOL vs TMM comparison
* [x] Sensitivity note for porous parameters

### 9.4 Validation criteria

* [x] Porous block gives physically credible damping behavior
* [x] Agreement is acceptable over the frequency band of interest

---

# WBS 10 — Merge slab and filter into a full reduced model

## 10.1 Objective

Build the full reduced model of the occluding system using either:

* validated **parallel slab + filter combination**, or
* **black-box COMSOL matrix** for the full assembly.

## 10.2 Tasks

* [x] Run 3 COMSOL cases: slab+filter, rigid+filter, slab-only with blocked filter

* [x] Identify $ \mathbf{T}_A $, $ \mathbf{T}_B $, $ \mathbf{T}_C $ with two-load inversion

* [x] Reconstruct the parallel equivalent from slab-only and rigid+filter

* [x] Compare $ \mathbf{T}_{\text{parallel}} $ to $ \mathbf{T}_A $

* [x] Decide the merging strategy:
  
  * [ ] use **parallel admittance** if error $ \leq 10% $
  * [x] use **black-box full COMSOL matrix** if error $ > 10% $

* [x] Build the full cascade with optional film / foam / cavity / radiation blocks

* [x] Choose the most stable formalism (T, Y, or hybrid) based on conditioning

* [x] Compute IL, end pressure, and end volume velocity

* [x] Compare reduced model results to COMSOL

* [ ] Diagnose discrepancies: slab, filter, interaction, or boundary condition error

## 10.3 Deliverables

* [x] Python reduced-order model with conversion / parallel / cascade utilities
* [ ] COMSOL vs TMM comparison plots
* [ ] Short discrepancy analysis note

## 10.4 Validation criteria

* [ ] IL mean error < 2 dB over 100–5000 Hz
* [ ] IL peak error < 5 dB over 100–5000 Hz
* [ ] Parallel model used only if full-assembly error $ \leq 10% $
* [ ] No ill-conditioned frequency point: $ \kappa(\mathbf{T}) \leq 10^6 $
* [ ] No near-singular point: $ |\det(\mathbf{T})| \geq 10^{-12} $
* [ ] Passive and physically plausible response

---

## WBS 11 — Lined-duct reduced model and slab-equivalent transferability

### 11.1 Objective

Build and validate a reduced lined-duct model for dissipative earplug elements, then assess whether slab-equivalent parameters extracted from porous, elastic, or COMSOL-derived slab matrices can be reused as locally reacting lining impedances.

### 11.2 Tasks

- [x] Implement minimal lined-duct reduced models
  - [x] Circular lined duct
  - [x] Rectangular lined duct
  - [x] One-sided and two-sided rectangular variants
- [x] Validate the lined-duct approximation against reference cases
  - [x] Compare circular lined duct against COMSOL
  - [x] Compare rectangular lined duct against COMSOL
  - [x] Assess the level of agreement and limitations
- [x] Implement slab-equivalent parameter retrieval
  - [x] Extract $k_{eq}$ and $Z_{c,eq}$ from an isolated slab matrix
  - [x] Reconstruct the equivalent slab matrix for self-consistency checks
  - [x] Verify branch tracking and continuity of the retrieved quantities
- [x] Test the transfer from slab-equivalent parameters to lining impedance
  - [x] Recover wall impedance from a porous Miki slab
  - [x] Recover wall impedance from an elastic silicone slab
  - [x] Reuse the recovered wall impedance inside the lined circular duct model
- [x] Test the same workflow on COMSOL-extracted slab matrices
  - [x] Isolate the slab contribution by decascading the surrounding elements
  - [x] Retrieve equivalent slab parameters from the COMSOL matrix
  - [x] Reinsert the recovered equivalent lining into the reduced filter model
- [x] Evaluate the validity domain of the approach
  - [x] Confirm the method on a porous reference case
  - [x] Identify the failure of transferability for the silicone / COMSOL slab case
  - [x] Document why an equivalent through-slab does not necessarily define a valid local lining impedance

### 11.3 Deliverables

- [x] Minimal lined circular and rectangular duct models
- [x] FEM / TMM comparison scripts for lined ducts
- [x] Slab-equivalent retrieval workflow for $(k_{eq}, Z_{c,eq})$
- [x] Verification cases for porous, elastic, and COMSOL-extracted slabs
- [x] Technical conclusion on the validity and limitation of reusing slab-equivalent parameters as lining properties

### 11.4 Validation criteria

- [x] The lined circular duct reproduces the COMSOL reference with acceptable agreement
- [x] The lined rectangular duct comparison reveals and documents the limits of the simplified model
- [x] The slab retrieval reconstructs the isolated slab matrix consistently
- [x] The porous-slab transfer test remains self-consistent
- [x] The silicone / COMSOL slab transfer test is shown to be non-predictive when reused as a lining, and the limitation is clearly identified

## WBS 12 — Design-space exploration and IL targeting

### 12.1 Objective

Use the validated reduced model for design.

### 12.2 Tasks

* [ ] Define IL targets
  
  * [ ] Flat $-10$ dB
  * [ ] Flat $-20$ dB
  * [ ] Other target shape if needed

* [ ] Select optimization variables
  
  * [ ] Slab thickness
  * [ ] Slab effective stiffness
  * [ ] Filter length
  * [ ] Filter diameter
  * [ ] Film impedance
  * [ ] Foam parameters

* [ ] Build target-error metrics
  
  * [ ] Mean error
  * [ ] Flatness error
  * [ ] Bandwidth constraints

* [ ] Run parameter sweeps

* [ ] Run optimization

* [ ] Check physical manufacturability of candidate designs

### 12.3 Deliverables

* [ ] Target definitions
* [ ] Optimization workflow
* [ ] Candidate designs
* [ ] Trade-off plots

### 12.4 Validation criteria

* [ ] At least one design reaches target behavior in simulation
* [ ] Optimized solution remains physically manufacturable

---

## WBS 13 — Experimental validation strategy

### 13.1 Objective

Validate the model chain experimentally.

### 13.2 Tasks

* [ ] Define measurement protocol

* [ ] Plan progressive characterization
  
  * [ ] Slab only
  * [ ] Tube / filter only
  * [ ] Film only if relevant
  * [ ] Foam-filled filter only
  * [ ] Full assembly

* [ ] Fabricate at least one prototype from optimized design

* [ ] Measure response

* [ ] Compare measurements with COMSOL

* [ ] Compare measurements with TMM

* [ ] Update parameter estimates if required

### 13.3 Deliverables

* [ ] Measurement plan
* [ ] Prototype test matrix
* [ ] Measured vs simulated comparison
* [ ] Updated parameter estimates

### 13.4 Validation criteria

* [ ] Discrepancies are localized to specific sub-blocks when present
* [ ] A clear model-refinement path is identified from measurements

---

# 3. Cross-cutting framework development

## WBS F1 — TMM framework improvements

### Tasks

* [ ] Add matrix de-embedding / uncascade operations

* [ ] Add robust port-state handling

* [ ] Add equivalent-matrix identification utilities

* [ ] Add comparison utilities
  
  * [ ] TMM vs COMSOL overlays
  * [ ] Matrix determinant tracking
  * [ ] Conditioning diagnostics

* [ ] Add support for duct losses

* [ ] Add support for variable section

* [ ] Add support for film impedance

* [ ] Add support for porous blocks

* [ ] Add admittance / hybrid formulations if required

### Deliverables

* [ ] Upgraded framework
* [ ] Reusable object operations for future studies

---

# 6. Risks and watch points

## R1 — Ill-conditioned two-load inversion

* [ ] Risk identified
* [ ] Conditioning tracker implemented
* [ ] Loads sufficiently separated
* [ ] Redundancy strategy considered if needed

## R2 — Inconsistent port definitions

* [ ] Risk identified
* [ ] Conventions frozen early
* [ ] Reconstruction tests added systematically

## R3 — Overinterpreting fixed-wall FE as “real ear”

* [ ] Risk identified
* [ ] Fixed-wall FE treated as limiting / simplified case
* [ ] Interpretation note written clearly

## R4 — Premature full-system complexity

* [ ] Risk identified
* [ ] Sub-block validation enforced before full assembly
* [ ] Integration order kept progressive

---

# 

---
