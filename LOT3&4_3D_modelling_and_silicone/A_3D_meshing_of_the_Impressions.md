# Report — 3D Scan of Ear Impressions

**Note:** Lots 3 and 4 are strongly connected, so the iterations were carried out jointly. It is therefore more relevant to consider them as a single combined workflow in the analysis.

---

## A) 3D Acquisition of the Impressions

### 1. Tools Tested

The acquisition was carried out mainly with the **RealityScan** smartphone application.
**Meshroom** was also identified as a potential alternative for local processing on a PC, but it has not yet been tested in this context.

---

### 2. RealityScan — Feedback

#### Main advantages

* **Ease of use**: quick to learn and simple workflow.
* **Real-time tracking mode**:
  allows the user to visualize image alignment during capture and immediately assess the quality of the future mesh.
* **Pre-cropping before export**, which simplifies post-processing.

---

### 3. Experimental Conditions and Setup

The acquisition protocol relies on:

* a **rotating platform mounted on a tripod**, making it easy to move around the object;
* the addition of a **scale reference** (round piece or flat 3D-printed shape) to ensure reliable scaling during processing;
* homogeneous and diffuse lighting.

#### Influence of the support

Two types of support plates were tested:

| Support               | Result                                    |
| --------------------- | ----------------------------------------- |
| Black plate           | Difficult alignment, many rejected images |
| Textured wooden plate | Much better alignment, more stable mesh   |

Conclusion: a **textured surface** significantly improves photogrammetric reconstruction quality.

---

### 4. Processing Time

Observed times at the current stage:

* Acquisition: **30–45 min** per impression
* Cloud processing: **5–10 min**

Further optimization of the protocol could reduce:

* total time to around **15–20 min**, although it seems difficult to go much below that with this approach.

---

### 5. Improvement Paths

Several possible improvements are being considered:

* smartphone acquisition with **local PC processing** in order to:
  
  * reduce dependence on cloud processing,
  * enable a more customizable workflow,
  * directly generate a usable STL export.

* testing **Meshroom** or other local photogrammetry solutions.

---

### 6. Result Quality

In most cases:

* the exported models are **clean and detailed**;
* the accuracy is sufficient for use in a modeling and manufacturing workflow.

### RealityScan export examples (PNG)

Right ear:

<img src="images/ear_result_D.png" alt="RealityScan export - right ear" width="212">

Left ear:

<img src="images/ear_result_G.png" alt="RealityScan export - left ear" width="380">

---

## 7. Outlook Toward More Professional Solutions

The current smartphone photogrammetry approach offers an excellent cost-to-flexibility ratio, but some limitations remain:

* relatively long acquisition time,
* dependence on setup quality and environmental conditions,
* variable precision and reproducibility.

Several improvement paths are therefore being considered, ranging from improved DIY solutions to dedicated industrial systems.

---

### 7.1 Semi-professional Solutions (Dedicated Photogrammetry Setup)

A first possible evolution is to keep the photogrammetry approach while using a more optimized hardware setup.

#### Automated rotating platform

Using a motorized rotating platform could:

* improve capture regularity,
* reduce acquisition time,
* increase reproducibility.

However, the real gain in accuracy remains uncertain because the main limitations are still linked to:

* image resolution,
* hard-to-reconstruct areas such as the deep ear canal,
* texture-related artifacts.

An estimated budget of around **€500** would allow this type of improvement to be tested.

#### OpenScan Mini solution

![Image](https://openscan.eu/cdn/shop/files/DSC07842.jpg?v=1710924489&width=3162)

The **OpenScan Mini** is an interesting alternative:

* open-source photogrammetry scanner,
* automated rotating platform,
* local PC workflow,
* direct STL export,
* moderate cost (~€200–400).

This solution could be a good compromise between:

* DIY flexibility,
* improved reproducibility,
* independence from cloud services.

---

### 7.2 Professional Solutions — Scanning Physical Ear Impressions

These systems keep the silicone impression step but use dedicated scanners for very fast and highly accurate digitization.

![Image](https://static.wixstatic.com/media/adca4e_72b6d3363c7749a1a5dc0a6b80bf8f62~mv2.jpg/v1/fill/w_568%2Ch_474%2Cal_c%2Cq_80%2Cusm_0.66_1.00_0.01%2Cenc_avif%2Cquality_auto/adca4e_72b6d3363c7749a1a5dc0a6b80bf8f62~mv2.jpg)

#### General characteristics

* industrial accuracy: **12 to 30 µm**,
* very fast acquisition: a few seconds,
* direct STL export,
* stable and reproducible workflow.

#### Typical solutions

* **3Shape A3**: historical reference in hearing-aid laboratories.
* **3Shape A2**: more accessible version.
* **3DIFY JME3**: more compact and modern alternative for R&D use.

These systems provide major gains in:

* speed,
* precision,
* workflow standardization.

The main limitation remains their high cost, generally above **€10,000**.

---

### 7.3 Professional Solutions — Direct In-Ear Scanning

These systems perform acquisition directly in the ear canal, without a silicone impression.

#### Advantages

* elimination of the impression step,
* rapid acquisition,
* very high point density.

#### Limitations

* often closed ecosystems,
* high cost,
* workflows that may be less flexible for R&D use.

#### Typical solutions

* **Natus Otoscan**: widely used in clinical audiology.
* **Empress3D (Aurality)**: more open solution with STL export.
* **Scane[a]r (Earow)**: French structured-light solution.

---

### 7.4 Comparative Summary of the Solution Levels

| Level                                | Cost      | Accuracy  | Time       | Flexibility   |
| ------------------------------------ | --------- | --------- | ---------- | ------------- |
| Smartphone photogrammetry            | Very low  | Medium    | Long       | Very high     |
| OpenScan / dedicated turntable setup | Low       | Medium+   | Medium     | High          |
| Dedicated lab impression scanner     | High      | Very high | Very short | Medium        |
| Direct in-ear scan                   | Very high | Very high | Very short | Low to medium |

---

### 7.5 Direction for the Next Project Steps

Considering the current needs (prototyping, R&D, cost control), the most relevant path appears to be:

**Short term:**
optimize the current photogrammetry workflow.

**Medium term:**
test an OpenScan-type system or equivalent.

**Long term:**
evaluate a dedicated impression scanner if regular production is envisioned.
