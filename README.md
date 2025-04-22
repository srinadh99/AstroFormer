# AstroFormer
Integrate photometric data and imaging using Cross-Attention Vision Transformers to enhance astronomical image processing.

## Applications

1. Classification - Star-Galaxy-Quasar Separation
2. Regression - Photometric Redshift Estimation

# Results

1. Classification
Using Sloan Digital Sky Survey (SDSS) Data Release 16

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
