# AstroFormer

**A Unified Framework for Integrating Photometry and Imaging in Astronomy using Cross-Attention Transformers**


AstroFormer explores modern deep learning architectures, specifically Cross-Attention Vision Transformers, to synergistically integrate heterogeneous astronomical data types—photometric parameters and multi-band imaging data. AstroFormer aims to enhance the performance, efficiency, and generalization capabilities of models applied to fundamental astronomical tasks by processing these modalities within a single, unified framework, moving beyond traditional methods that often process data streams separately before late-stage fusion. The core idea is to leverage photometric information (like brightness, colours, and concentration indices) to *guide* the feature extraction process from imaging data. This is achieved by employing a cross-attention mechanism where photometric features act as *queries*, probing the *keys* and *values* derived from image patches. This allows the model to learn intricate cross-modal correlations and focus on visual details most relevant given the photometric context of the source.


## Applications

AstroFormer provides a flexible framework adaptable to astronomical problems requiring the fusion of tabular (photometric) and visual (imaging) data. Key applications currently explored or planned include:

### 1. Source Classification: Star-Galaxy-Quasar Separation

*   **Goal:** Accurately distinguish between stars, galaxies, and quasars, a fundamental step in catalog generation and numerous downstream scientific analyses (e.g., cosmology, galaxy evolution studies, Galactic structure mapping).
*   **Challenge:** Classification becomes particularly difficult for faint and compact objects where traditional morphological or color-based methods often struggle. Point-like quasars can be mistaken for stars, and compact galaxies can resemble point sources at large distances or faint magnitudes.
*   **AstroFormer Approach (`MargFormer`):** By using photometry to query image features, the model can better disentangle subtle morphological differences or color-space overlaps that might confuse models relying solely on one data type or simple fusion. This is expected to improve accuracy and robustness, especially for the challenging faint/compact regimes critical for upcoming deep surveys.

### 2. Regression: Photometric Redshift Estimation

*   **Goal:** Estimate the redshift (a proxy for distance) of celestial objects using only photometric data and associated images, bypassing the need for time-consuming spectroscopic observations. Accurate photo-z's are crucial for large-scale structure mapping, cosmological parameter estimation (e.g., Dark Energy equation of state), and galaxy evolution studies over cosmic time.
*   **Challenge:** Achieving high precision (low scatter) and accuracy (low bias), and minimizing the rate of catastrophic outliers across a wide range of redshifts and object types, remains a significant challenge.
*   **AstroFormer Approach:** Integrating morphological information from images (e.g., galaxy type, size, profile) directly with photometric colors and magnitudes through cross-attention could provide complementary information to break degeneracies inherent in photometry-only methods, potentially leading to more accurate and robust redshift predictions. The model can learn how specific morphologies correlate with photometric properties at different redshifts.

## Implemented Models & Datasets

### 1. `MargFormer`: Classification Model

*   **Task:** Star-Galaxy-Quasar classification.
*   **Description:** `MargFormer` is the specific implementation of the AstroFormer concept tailored for the SGQ classification task. It utilizes the cross-attention mechanism where photometric features query image patch embeddings.
*   **Dataset:** The model is developed and evaluated using data products from the **Sloan Digital Sky Survey (SDSS) Data Release 16 (DR16)** [1]. This includes:
    *   Derived photometric features (magnitudes, colors, etc.).
    *   Corresponding FITS images in u, g, r, i, z filters.
    *   Ground-truth spectroscopic classifications from the official SDSS pipeline.
*   **Focus:** The evaluation specifically targets challenging populations of faint and compact objects, using selection criteria and experimental setups identical to those in Chaini et al. (2023) [2] to ensure direct comparability and demonstrate improvements in generalization.
*   **References:**
    *   [1] Ahumada, R., et al. (2020). *The 16th Data Release of the Sloan Digital Sky Surveys: First Release from the APOGEE-2 Southern Survey and Full Release of eBOSS Spectra*. ApJS, 249(1), 3. ([DOI: 10.3847/1538-4365/ab929e](https://doi.org/10.3847/1538-4365/ab929e))
    *   [2] Chaini, P., et al. (2023). *MargNet: A Machine Learning Framework for Photometric Classification of Faint Compact Sources in SDSS DR16*. MNRAS, 521(3), 3788–3799. ([DOI: 10.1093/mnras/stad719](https://doi.org/10.1093/mnras/stad719))


Experiment | Classification | Accuracy | Precision | Recall 
--- | --- | --- | --- | --- 
Ex1 | Star-Galaxy | 98.1 ± 0.1 | 98.1 ± 0.1 | 98.1 ± 0.1
_ | Star-Galaxy-Quasar | 93.2 ± 0.1 | 93.3 ± 0.1 | 93.2 ± 0.1
Ex2 | Star-Galaxy | 97.1 ± 0.1 | 97.1 ± 0.1 | 97.1 ± 0.1
_ | Star-Galaxy-Quasar | 86.7 ± 0.1 | 86.8 ± 0.1 | 86.7 ± 0.1
Ex3 | Star-Galaxy | 92.7 ± 0.1 | 93.2 ± 0.1 | 92.7 ± 0.1
_ | Star-Galaxy-Quasar | 75.2 ± 0.1 | 77.9 ± 0.1 | 75.3 ± 0.1

### The confusion matrix for MargFormer with CLS desired using Photometric Features and alone is used as a query in the attention are presented for three different experiments.
### For Star-Galaxy Classification.

| Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|
| ![Plot 1](./MargFormer/Trained_Models/EX1_SG_ViTCLSPFCA_CM.png) | ![Plot 2](./MargFormer/Trained_Models/EX2_SG_ViTCLSPFCA_CM.png) | ![Plot 3](./MargFormer/Trained_Models/EX3_SG_ViTCLSPFCA_CM.png) |

### For Star-Galaxy-Quasar Classification.

| Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|
| ![Plot 1](./MargFormer/Trained_Models/EX1_SGQ_ViTCLSPFCA_CM.png) | ![Plot 2](./MargFormer/Trained_Models/EX2_SGQ_ViTCLSPFCA_CM.png) | ![Plot 3](./MargFormer/Trained_Models/EX3_SGQ_ViTCLSPFCA_CM.png) |

2. Regression Galaxy Redshift Estimation using HSC Data
