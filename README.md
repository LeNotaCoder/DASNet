# 🚀 DASNet: Dual Adaptive Subtle-Feature Network for Enhanced Diabetic Retinopathy Detection

![DASNet Architecture](/images/model.png) <!-- Replace with actual path to your architecture image -->

## 🧩 Overview

**DASNet (Dual Adaptive Subtle-Feature Network)** is a novel deep learning architecture designed to capture both **dominant and subtle retinal features** for diabetic retinopathy (DR) detection from fundus images.  
It employs a **dual-branch CNN** integrated with **MaxPooling**, **Adaptive MaxPooling**, and **Spatial Pyramid Pooling (SPP)** to achieve robust and scale-invariant feature representation.  

This work has been **proposed at the International Conference on Pattern Recognition Applications and Methods (ICPRAM 2026)** and is currently **under review**.

---

## 🌟 Key Contributions

- **Dual-Branch Design:**  
  - *Branch 1*: Extracts prominent retinal structures using standard MaxPooling.  
  - *Branch 2*: Captures subtle features (e.g., microaneurysms) using Adaptive MaxPooling.

- **Spatial Pyramid Pooling (SPP):**  
  Integrates multi-scale contextual features (1×1, 2×2, 4×4 pooling levels) to handle variable image resolutions.

- **Performance Gains:**  
  - Achieves **95.34% (BiDR)**, **95.65% (APTOS)**, and **97.46% (EDI)** accuracy.  
  - Outperforms existing models with improvements of:  
    - +1.85–6.69% over **DenseNet121**  
    - +2.08–5.27% over **VGG16**  
    - +2.80–11.28% over **ResNet50**  
    - +2.74–8.06% over **InceptionV3**

- **Interpretability:**  
  Visual feature maps from each branch highlight how different lesions are detected and processed.

---

## 📊 Results Summary

| **Dataset** | **Accuracy (%)** | **Dataset Size** |
|--------------|------------------|------------------|
| BiDR         | 95.34            | 2,816 images     |
| APTOS 2019   | 95.65            | 3,610 images     |
| EDI          | 97.46            | 2,048 images     |

---

## 🧪 Ablation Study

### **On the APTOS Dataset**

| Model Variant                        | Accuracy | F1 Score | Precision | Recall |
|-------------------------------------|-----------|-----------|------------|---------|
| **DASNet (Full)**                   | 0.9534    | 0.9534    | 0.9536     | 0.9534  |
| DASNet (w/o Pyramid Pooling)        | 0.9293    | 0.9293    | 0.9304     | 0.9293  |
| MaxPool CNN                         | 0.9489    | 0.9488    | 0.9499     | 0.9489  |
| AdaptivePool CNN                    | 0.9528    | 0.9527    | 0.9536     | 0.9528  |

### **On the BiDR Dataset**

| Model Variant                        | Accuracy | F1 Score | Precision | Recall |
|-------------------------------------|-----------|-----------|------------|---------|
| **DASNet (Full)**                   | 0.9565    | 0.9565    | 0.9568     | 0.9565  |
| DASNet (w/o Pyramid Pooling)        | 0.9338    | 0.9337    | 0.9346     | 0.9338  |
| MaxPool CNN                         | 0.9488    | 0.9488    | 0.9489     | 0.9488  |
| AdaptivePool CNN                    | 0.9493    | 0.9493    | 0.9499     | 0.9493  |

### **On the EDI Dataset**

| Model Variant                        | Accuracy | F1 Score | Precision | Recall |
|-------------------------------------|-----------|-----------|------------|---------|
| **DASNet (Full)**                   | 0.9746    | 0.9746    | 0.9750     | 0.9746  |
| DASNet (w/o Pyramid Pooling)        | 0.9311    | 0.9311    | 0.9322     | 0.9311  |
| MaxPool CNN                         | 0.9717    | 0.9720    | 0.9717     | 0.9717  |
| AdaptivePool CNN                    | 0.9673    | 0.9673    | 0.9677     | 0.9673  |

---

## 🧠 Model Architecture

DASNet processes input images through two parallel branches:

1. **Branch 1 (MaxPooling):**  
   Extracts prominent features such as lesion edges and vessel boundaries.

2. **Branch 2 (Adaptive MaxPooling):**  
   Preserves fine-grained lesion details (e.g., microaneurysms, exudates).

Both branches' outputs are **concatenated** and passed through a **shared convolutional block** with ReLU activation and MaxPooling for deeper feature enrichment.  
A **Spatial Pyramid Pooling (SPP)** layer follows, aggregating features at multiple scales (1×1, 2×2, 4×4), producing a **fixed-length vector** independent of input image size.  

Finally, fully connected layers with **ReLU** activations and **dropout** regularization are used before a **binary classification output layer** (`Healthy` vs. `DR`).  

> 🧮 Total trainable parameters: **46,607,746**

---

![DASNet Architecture](/images/pipeline.png)

---

## 🧼 Preprocessing Pipeline

1. **Resize** images to `224×224`
2. **Convert** from BGR → HSV color space
3. **Apply CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the Value (V) channel
4. **Reconvert** to RGB for model input

> This enhances local contrast and makes subtle lesions more visible under variable lighting.

---

## 📚 Datasets Used

| Dataset | Description | Label Format |
|----------|--------------|--------------|
| **BiDR** | Binary DR dataset (private) | 0 = Healthy, 1 = DR |
| **APTOS 2019** | Public Kaggle dataset | 0 = Healthy, 1 = DR |
| **Eye Disease Image (EDI)** | Retinal fundus dataset | 0 = Healthy, 1 = DR |
| **Unified Dataset** | Combination of Eyepacs, Aptos, Messidor | 0 = Healthy, 1 = DR |

---

## ⚙️ Training Setup

| Setting | Value |
|----------|--------|
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |
| **Epochs** | 25 |
| **Batch Size** | 32 |
| **Cross-Validation** | 3-Fold |
| **Hardware** | NVIDIA A100 (80 GB) |

---

## 🖼️ Feature Visualization

### Adaptive Pooling Branch
![Adaptive Pooling branch](/images/adaptive_pool.png)

### MaxPooling Branch
![MaxPooling branch](/images/maxpool.png)

---

