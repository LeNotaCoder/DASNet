# DASNet: A Dual Adaptive Subtle-Feature Network for Enhanced Diabetic Retinopathy Detection in Fundus Images

<p align="center">
  <img src="/images/architecture.png" alt="DASNet Architecture" width="80%">
</p>

---

## 📄 Overview

**DASNet (Dual Adaptive Subtle-feature Network)** is designed to capture complex feature representations in fundus images using a robust preprocessing pipeline to enhance image quality.  
The proposed architecture employs a **dual-branch convolutional neural network** that integrates **MaxPooling**, **Adaptive MaxPooling**, and **Spatial Pyramid Pooling (SPP)** to simultaneously capture **subtle and dominant retinal features**.

This model has been **proposed at the International Conference on Pattern Recognition Applications and Methods (ICPRAM’26)** and is currently under review.

---

## 🧠 Key Contributions

- **High Accuracy:**  
  DASNet achieves **95.34%** on the BiDR dataset, **95.65%** on APTOS, and **97.46%** on the Eye Disease Image (EDI) dataset.  
  It outperforms existing models by:
  - +1.85–6.69% over DenseNet121  
  - +2.08–5.27% over VGG16  
  - +2.80–11.28% over ResNet50  
  - +2.74–8.06% over InceptionV3  

- **Interpretability:**  
  Visualizes feature maps from each branch to show how different lesion types are detected.

- **Scalability:**  
  Suitable for **automated DR screening** in **telemedicine systems**.

---

## 📊 Results

| Dataset | DASNet Accuracy | Dataset Size |
|----------|------------------|--------------|
| BiDR     | 95.34%           | 2816 images  |
| APTOS    | 95.65%           | 3610 images  |
| EDI      | 97.46%           | 2048 images  |

---

## 🔍 Ablation Study

### APTOS Dataset

| Model | Accuracy | F1 Score | Precision | Recall |
|:------|:----------|:----------|:-----------|:--------|
| **DASNet** | 0.9534 | 0.9534 | 0.9536 | 0.9534 |
| DASNet (w/o Pyramid Pooling) | 0.9293 | 0.9293 | 0.9304 | 0.9293 |
| MaxPool CNN | 0.9489 | 0.9488 | 0.9499 | 0.9489 |
| AdaptivePool CNN | 0.9528 | 0.9527 | 0.9536 | 0.9528 |

### BiDR Dataset

| Model | Accuracy | F1 Score | Precision | Recall |
|:------|:----------|:----------|:-----------|:--------|
| **DASNet** | 0.9565 | 0.9565 | 0.9568 | 0.9565 |
| DASNet (w/o Pyramid Pooling) | 0.9338 | 0.9337 | 0.9346 | 0.9338 |
| MaxPool CNN | 0.9488 | 0.9488 | 0.9489 | 0.9488 |
| AdaptivePool CNN | 0.9493 | 0.9493 | 0.9499 | 0.9493 |

### EDI Dataset

| Model | Accuracy | F1 Score | Precision | Recall |
|:------|:----------|:----------|:-----------|:--------|
| **DASNet** | 0.9746 | 0.9746 | 0.9750 | 0.9746 |
| DASNet (w/o Pyramid Pooling) | 0.9311 | 0.9311 | 0.9322 | 0.9311 |
| MaxPool CNN | 0.9717 | 0.9720 | 0.9717 | 0.9717 |
| AdaptivePool CNN | 0.9673 | 0.9673 | 0.9677 | 0.9673 |

---

## 🛠️ Model Architecture

Input images are processed in parallel through **two branches**:

- **Branch 1:** Uses **MaxPooling** to extract dominant structures such as lesion edges.  
- **Branch 2:** Uses **Adaptive MaxPooling** to preserve subtle features like microaneurysms.  

The extracted feature maps are concatenated and passed through convolutional layers with **ReLU** and **MaxPooling** for feature enrichment.  
A **Spatial Pyramid Pooling (SPP)** layer then uses multiple pooling levels (1×1, 2×2, 4×4) to generate a fixed-length vector, capturing both **global and local lesion context**.

Finally, the flattened features are passed through **fully connected layers with dropout** and a **binary classification head (Healthy vs DR)**.  

**Total Trainable Parameters:** 46,607,746

<p align="center">
  <img src="/images/pipeline.png" alt="DASNet Pipeline" width="80%">
</p>

---

## 🖼️ Preprocessing Pipeline

1. Resize images to `224×224`  
2. Convert **BGR → HSV**  
3. Apply **CLAHE** on Value (V) channel  
4. Convert back to **RGB**

This preprocessing enhances **subtle lesion visibility** and **illumination consistency** across samples.

---

## 📁 Datasets Used

1. **BiDR**
2. **APTOS 2019**
3. **Eye Disease Image (EDI)**
4. **Unified Dataset** (Eyepacs, APTOS, Messidor)

All datasets were standardized to **binary labels**:  
`0 = Healthy`, `1 = DR`.

---

## 🧪 Training Setup

- **Optimizer:** Adam  
- **Loss Function:** Binary Cross-Entropy  
- **Epochs:** 25  
- **Batch Size:** 32  
- **Cross Validation:** 3-Fold  
- **Hardware:** NVIDIA A100 (80GB)

---

## 📈 Visualization

<p align="center">
  <img src="/images/DR.png" alt="Grad-CAM Visualization" width="80%">
</p>

<p align="center">
  <img src="/images/normal.png" alt="Grad-CAM Visualization" width="80%">
</p>

**Grad-CAM visualizations for fundus images.**  
The top image shows a **diabetic retinopathy (DR)** sample, while the bottom image shows a **normal** sample.  
These heatmaps illustrate the regions the model focuses on for classification.

---
