# Earplug Lot 5 Simulation

This document provides a compact narrative for the earplug simulation work developed in `Codes_Earplugs`.

The project should not be read as a single linear implementation. It evolved from an initial broad modeling ambition into a progressively validated reduced-model framework. The code therefore contains both reusable models and the intermediate steps that were needed to understand which simplifications were reliable.

## Project Context

Two planning documents coexist at project level:

- `original_plan.md` records the initial modeling ambition and theoretical reference framework.
- `new_plan.md` reflects the later and more realistic work breakdown after the project focus evolved.

The correct overall interpretation is therefore not “one final model built from the start”, but “a reduced-model methodology progressively validated against more detailed references”.

## Problem Evolution

The initial objective was to build a Transfer Matrix Method framework able to simulate an earplug system from standard one-dimensional acoustic building blocks: ducts, terminations, discontinuities, films, porous layers, and compliant elements.

The first major difficulty was the discrepancy between direct slab TMM modeling and FEM results for the silicone slab configuration. This was not just a small mismatch. It showed that some physical boundary conditions and coupling effects were not captured by the simplest reduced description.

This led to an important shift in method:

1. reduced models could not simply be assumed valid,
2. matrix identification and de-embedding became central tools,
3. effective parameters had to be treated carefully, especially when transferred from one configuration to another.

The project therefore moved from direct modeling alone toward a workflow combining reduced models, FEM comparison, equivalent matrix extraction, and explicit validity checks.

## Methodology

The work is built on four complementary layers:

1. elementary TMM blocks for ducts, discontinuities, radiation, films, plates, and porous or equivalent-fluid layers,
2. FEM reference models whenever the limits of the simplest reduced models are uncertain,
3. port-based matrix identification and decascade workflows,
4. system-level metrics such as transmission loss and insertion loss under rigid-end or IEC711 loading.

The general validation logic is simple:

1. define a minimal physical configuration,
2. build the corresponding reduced model,
3. compare it to a FEM or otherwise controlled reference,
4. retain the reduced description only if its validity domain is clear.

This is why the repository contains many examples that may look partly redundant. In practice, they document successive decisions about which reduced descriptions are reliable and which are not.

## Main Results

The main outcomes are not limited to one final earplug model.

First, the toolbox now provides a usable set of validated acoustic elements: straight and viscothermal ducts, section changes, conical approximations, films, plates, porous layers, and relevant terminations. This gives a solid basis for reduced earplug studies.

Second, the slab discrepancy was a productive turning point. It forced the project away from an overly direct use of analytical submodels and toward a stricter identification workflow.

Third, equivalent matrix identification became a key practical tool. It made it possible to represent more complex subassemblies as reusable two-port elements inside the reduced framework, even when a direct analytical derivation was not sufficiently trustworthy.

Fourth, the project established an important negative result: not every equivalent quantity is transferable. In particular, parameters identified for a slab as a through element do not necessarily define a valid local wall impedance when reused as a lined-duct boundary condition. This limit is part of the contribution, not a side issue.

Fifth, the framework is already usable for early design work. The optimization examples show that the reduced models can already be used to formulate target-driven studies, even though some parameters remain effective rather than fully physically identified.

## Current Limits

The framework should not yet be presented as a fully identified predictive design tool for final earplug products.

The main limitations are:

- some retrieved equivalent parameters are configuration-dependent rather than intrinsic,
- simple fitted `R/M/K` parameters may not yet correspond to physically realizable components,
- sensitivity to detailed geometry and loading still needs tighter constraint for realistic earplug configurations,
- some reduced descriptions have only been validated on specific subproblems, not in every coupled situation.

## Limits and improovements

Although the codebase was developed following a class-based, object-oriented strategy, with systematic testing and FEM validation, many of the current example scripts could be turned into proper integration and non-regression tests. Refactoring in that direction would clearly improve the overall robustness and maintainability of the project.

Some of the latest work packages also rely on fairly cumbersome code. While this could definitely be improved, it reflects a common trade-off between rapid prototyping and over-engineered development. A cleanup phase would be valuable, but at this stage it seems more relevant to prioritize the experimental development rather than spend too much time polishing the simulation code.

## Next Steps

The next major step is measurement-informed refinement.

Measurement data should make it possible to:

- calibrate realistic parameter ranges,
- replace purely effective fits by more physical identified values,
- constrain the design space by manufacturability and measurable properties,
- validate the most promising reduced architectures experimentally.

The framework is also naturally extensible beyond the current film-based showcase. Candidate directions already visible in the codebase include Helmholtz resonators, quarter-wave resonators, porous fillings, section changes, parallel branches, and non-flat target responses.

## Code Structure

The main technical backbone is in:

- `Codes_Earplugs/Dev/`
- `Codes_Earplugs/Showcases/`
- `Codes_Earplugs/Theory/`

Their roles are different:

- `Dev/` preserves the research trail, validation steps, and work-package logic.
- `Showcases/` provides shorter and easier entry points for readers who want quick runnable examples.
- `Theory/` regroups the longer derivations and theory notes referenced by several work packages.

For a short technical reading path through the development results, the most relevant folders are:

1. `A10_wbs6`
2. `A11_wbs7`
3. `A12_wbs8`
4. `A13_wbs9`
5. `A14_wbs10`
6. `A15_wbs11`
7. `A16_wbs12_optimisation`

For the methodological history, earlier validation folders and WBS notes in `Dev/` remain important because they document how the modeling choices were established.

## Running the Code

From the repository root:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e LOT5_acoustic_filter_simulation/toolkitsd/Toolkitsd_porous
uv pip install -e LOT5_acoustic_filter_simulation/toolkitsd/Toolkitsd_acoustmm
uv run --active python LOT5_acoustic_filter_simulation/Codes_Earplugs/Showcases/A0_elementary_pieces/B0_rigid_duct.py
```

For more runnable entry points, see `Codes_Earplugs/Showcases/README.md`.

## Publication Position

The strongest publication narrative is not that a single final earplug model was completed directly. It is that a coherent reduced-model methodology was built, tested, and delimited.

The codebase now supports:

- validated reduced studies for several acoustic subproblems,
- explicit identification of validity limits and non-transferable approximations,
- first target-driven design and optimization workflows.

The next stage is therefore not a conceptual restart, but a measurement-informed continuation of an already coherent modeling framework.
