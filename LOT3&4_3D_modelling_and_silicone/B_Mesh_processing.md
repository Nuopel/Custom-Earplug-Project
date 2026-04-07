# Notes — Processing of 3D Scans of Ear Impressions

## Note on the Organization of the Lots

Since Lots 3 and 4 are strongly interdependent, the different iterations were carried out jointly. For this reason, they are treated as a coherent whole in the analysis below.

---

## B) 3D Mesh Processing

The processing of the ear-impression meshes obtained from 3D scanning relies on a toolchain combining **Blender** for basic geometric operations and **Meshmixer** for reconstruction, cleanup, and mesh finishing.

---

## 1. Pre-processing in Blender

### Import and Scaling

The raw mesh is first imported into Blender. The following operations are performed:

* correction of normal orientation to ensure mesh consistency,
* scaling of the model from a measured reference object present in the scan (flat element with known length).

This step ensures that the real dimensions required for ear-fit applications are preserved.

---

### Trimming the Impression

The mesh is then cleaned in order to keep only the useful geometry:

* use of the **Bisect** tool to remove the base of the scan,
* trimming and cleaning of the end of the ear canal to obtain a clean and usable termination.

---

### Mesh Reconstruction and Smoothing

Two modifiers are applied successively.

#### Voxel remesh

Objectives:

* homogenize the triangulation,
* remove local irregularities,
* produce a robust and uniform mesh.

The voxel-remesh parameters will be documented later for reproducibility.

#### Global smoothing

A **Smooth** modifier is then applied in order to:

* attenuate reconstruction artifacts,
* reduce surface irregularities,
* obtain a continuous surface suitable for manufacturing an ear insert.

---

### Export

Once these operations are completed, the model is exported in STL format for the next processing steps.

---

## 2. Post-processing in Meshmixer

Advanced mesh processing is carried out in Meshmixer, which provides tools specialized for reconstruction and finishing of organic surfaces.

The following video resources were used as methodological references:

* https://www.youtube.com/watch?v=qpsfMe0-fM8
* https://www.youtube.com/watch?v=bF5XvugnWkQ

---

### 2.1 Mesh Reconstruction

Ear-impression scans generally present:

* very high triangle density,
* local imperfections,
* topological defects.

A reconstruction is therefore performed using:

**Edit → Make Solid**

This operation allows:

* homogenization of the triangulation,
* removal of artifacts,
* reduction of mesh complexity,
* generation of a continuous and stable surface.

High accuracy and mesh-density settings are used in order to preserve geometric precision.

---

### 2.2 Manual Cleanup

Some degraded areas or residual artifacts are removed manually using the selection tools.

The corrected surfaces are then rebuilt with a new **Make Solid** operation to ensure mesh continuity.

---

### 2.3 Model Smoothing

Smoothing is an essential step for comfort and for the quality of the future device.

Two smoothing levels are applied.

#### Local smoothing

Used to:

* remove sharp edges,
* avoid pressure zones during wear.

#### Global smoothing

A **Deform → Smooth** operation is applied to the whole model in order to:

* homogenize the surface,
* reproduce the traditional finishing effect obtained by wax immersion,
* improve the sealing of the earplug.

---

### 2.4 Dimensional Adjustment

A slight expansion of the model is applied through normal surface offset in order to:

* ensure better retention,
* improve acoustic isolation.

### 2.5 Earplug generation

The earplug is created by extending the end using **Extract** with a normal offset.
This step can also be carried out locally with the **Replace** tool (`F` shortcut).

<img src="images/meshmixer.png" alt="Meshmixer workflow" width="600">

---

## 3. Generation of the Printing Mold

Once the final model is validated:

1. the STL is reimported into Blender,
2. the impression is converted into a negative shape inside a volume block using a boolean operation,
3. the resulting model is exported for 3D printing to be printed with a bambu A1 3D printer.

<img src="images/moule_blender.png" alt="Blender mold generation" width="600">

---

## 4. Automation Prospects

The initial objective was to perform the whole processing chain in Blender so that the workflow could eventually be automated with Python scripts.

However:

* some operations such as organic reconstruction, **Make Solid**, and advanced smoothing are significantly more effective in Meshmixer,
* full automation therefore remains limited at this stage.

A possible improvement path would be to:

* use the Meshmixer API,
* develop a semi-automated pipeline controlled from Python,
* reduce the amount of manual processing time.

---

## C) Earplug Tests — Geometry Iteration

Several iterative test series were carried out in parallel with the mesh-processing work in order to converge toward usable molded earplugs. In practice, one or two geometry iterations were generally sufficient to obtain a comfortable full-silicone prototype after the first test with 3-4 diffrent test subject feedback (after the initial test on myself).

These tests confirm that the overall workflow can produce meaningful results in terms of fit and manufacturability. However, the current prototypes remain preliminary, since they are still full silicone parts without an integrated acoustic filter. Further work is therefore needed to improve the design and move toward functional filtered earplug configurations.

<img src="images/Result_test.jpg" alt="Overview of molded earplug tests" width="443">

## Conclusion

The current workflow efficiently combines:

* the geometric precision and modeling tools of Blender,
* the organic reconstruction and surface-finishing capabilities of Meshmixer.

This hybrid approach makes it possible to obtain robust, smooth, and dimensionally reliable meshes suitable for manufacturing custom ear inserts.
