# WB9 — Porous layer report comment

## Scope

WB9 focuses on porous slabs inserted in a duct, using equivalent-fluid models in TMM. The implemented examples cover:

- `A0`: JCA layer compared with FEM
- `A1`: JCA layer compared with Miki layer in the same duct configuration
- `A2`: Miki-only parameter sweep on airflow resistivity and slab thickness
- `A3`: sensitivity of the IL to the source-side impedance model

The aim is not to validate every porous formulation exhaustively, but to establish a clean working framework for porous inserts in the same spirit as WB8.

---

## A0 — JCA porous slab against FEM

**Comment:**
Case `A0` is the reference validation case for WB9. It checks that a porous slab described by the JCA equivalent-fluid law can be inserted in the same duct/IEC711 environment and compared consistently with FEM.

This case should be interpreted as the main validation anchor of the porous workflow:

- the porous layer is no longer a simple local impedance, but a finite-thickness transfer element,
- the TMM implementation uses the expected complex wavenumber and characteristic impedance structure,
- and the FEM comparison confirms that the layer-based representation is usable in the current acoustic chain.

So `A0` plays for WB9 the same role that `A0–A2` played for WB8: it establishes confidence in the modeling and comparison procedure before moving to parameter studies.

---

## A1 — JCA vs Miki

**Comment:**
Case `A1` is not primarily a validation case. It is a constitutive-law comparison case.

The purpose is to show that:

- the two porous models can be embedded in exactly the same duct system,
- their transfer matrices can be compared directly,
- and the resulting IEC711 insertion loss can be contrasted without changing the surrounding acoustic setup.

This example is useful because it separates two questions that are often mixed together:

1. is the duct-side TMM implementation correct?
2. what difference comes purely from the porous constitutive law?

With `A0` providing the JCA reference, `A1` then becomes a compact way to assess whether Miki is close enough for fast studies, or whether the extra physical detail of JCA matters for the intended application.


---

## A2 — Miki sigma and thickness sweep

**Comment:**
Case `A2` is the first real design-space exploration case of WB9.

It shows the effect of:

- airflow resistivity `sigma`,
- porous slab thickness,
- and load condition (`IEC711` versus rigid end)

on insertion loss.

This is important because porous performance is controlled at least as much by parameter scale as by model choice. In practice:

- increasing thickness tends to increase attenuation and broaden the useful band,
- increasing resistivity tends to increase dissipation up to a point,
- but very large resistivity can also push the response toward stronger reflection rather than purely useful absorption.

So `A2` is the script that gives the first practical reading of what parameter ranges are acoustically interesting before going further into fitting or optimization.

---

## A3 — Source impedance sensitivity

**Comment:**
Case `A3` checks a secondary but important modeling detail: the effect of using a simple plane-wave source impedance `Z0` versus a flanged radiation impedance at the inlet.

This does not change the porous law itself, but it changes the surrounding system seen by the layer. The example is therefore useful to separate:

- effects intrinsic to the porous slab,
- from effects caused by the way the source boundary is modeled.

This is a good sanity check, especially when insertion-loss differences are modest and boundary assumptions may influence the result significantly.

---

## Global synthesis

> `A0` establishes the porous-slab TMM workflow with a JCA reference comparison against FEM.
>
> `A1` then shows how the same setup can be used to compare two equivalent-fluid constitutive laws, JCA and Miki, without changing the surrounding duct model.
>
> `A2` turns the porous model into a practical exploration tool by sweeping airflow resistivity and thickness, which are the two most immediate parameters controlling the attenuation trend.
>
> `A3` adds a boundary-condition sensitivity check by comparing a simple plane-wave source impedance with a flanged-radiation source impedance.
>
> Overall, WB9 provides a coherent porous-layer extension of the WB8 workflow: first validate one reference layer model, then compare reduced models, then explore parameter sensitivity, and finally check the influence of the acoustic environment around the porous insert.
