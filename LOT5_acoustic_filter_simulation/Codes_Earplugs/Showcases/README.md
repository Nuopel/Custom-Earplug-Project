# Earplug Showcases

This folder is a compact entry point to the earplug simulation work.

The goal is not to expose the full research and development history, but to provide a few simple scripts that are easier to read and reuse. Each subfolder highlights one aspect of the project with reduced, self-contained examples.

## Folder Structure

### `A0_elementary_pieces`

This folder presents the main acoustic building blocks one by one.

The scripts show how to assemble and run simple systems such as:
- rigid ducts
- lossy ducts
- section changes
- conical ducts
- films
- Helmholtz resonators
- plates
- porous layers

The objective is to make the available elements easy to understand before moving to more advanced identification or design tasks.

### `A1_3mics_2load_param_extraction`

This folder presents the parameter-extraction workflow in a minimal way.

The scripts show how to:
- synthesize or load microphone data
- reconstruct a transfer matrix from a 3-microphone / 2-load method
- compare the extracted matrix to a direct reference matrix
- test simplified equivalent representations such as parallel combinations

The objective is to show how an identified acoustic element can be recovered from boundary measurements or FEM-derived data, without carrying the full complexity of the development scripts.

### `A2_optimisation`

This folder presents the design-oriented side of the project.

The scripts show how reduced acoustic models can be used to tune a filter toward a target insertion loss. The current examples focus on simple parameterized elements, but the same logic can be extended to more complex devices.

The objective is to illustrate that the toolbox is not only a simulation framework, but also a starting point for target-driven design.

## How To Read This Folder

The recommended order is:

1. Start with `A0_elementary_pieces` to understand the main acoustic elements.
2. Then look at `A1_3mics_2load_param_extraction` to see how equivalent matrices can be identified.
3. Finish with `A2_optimisation` to see how the reduced models can be used for design exploration.

For the full development history, validations, and report notes, see the corresponding material in `../Dev/` and the grouped theory notes in `../Theory/`.
