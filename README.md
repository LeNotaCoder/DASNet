# 🚀 DASNet: Dual Adaptive Subtle-Feature Network for Enhanced Diabetic Retinopathy Detection

![DASNet Architecture](/images/model.png) <!-- Replace with actual path to your architecture image -->

---

## 🧩 Overview

**DASNet (Dual Adaptive Subtle-Feature Network)** is a deep learning architecture developed to enhance diabetic retinopathy (DR) detection from fundus images.  
The model integrates **dual-branch convolutional processing** and **multi-scale pooling** to simultaneously capture **dominant retinal structures** and **subtle lesion features**.  
It is designed for robustness, interpretability, and scalability in telemedicine-based DR screening systems.

This work has been **proposed at the International Conference on Pattern Recognition Applications and Methods (ICPRAM 2026)** and is currently **under review**.

---

## 🌟 Key Contributions

- **Dual-Branch Design**
  - *Branch 1 (MaxPooling):* Extracts dominant retinal structures and lesion edges.  
  - *Branch 2 (Adaptive MaxPooling):* Captures subtle features such as microaneurysms and small exudates.

- **Spatial Pyramid Pooling (SPP):**  
  Enables scale-invariant feature extraction by aggregating information at multiple pooling levels (1×1, 2×2, 4×4).

- **Performance Improvements:**  
  - DASNet outperforms baseline models with accuracy gains of:  
    - +1.85–6.69% over **DenseNet121**  
    - +2.08–5.27% over **VGG16**  
    - +2.80–11.28% over **ResNet50**  
    - +2.74–8.06% over **InceptionV3**

- **Interpretability:**  
  Visual feature maps from both branches show how DASNet distinguishes between subtle and dominant retinal lesions.

---

## 🧠 Model Architecture

DASNet processes each input fundus image through **two parallel branches**:

1. **Branch 1 (MaxPooling)** — extracts dominant retinal structures (vessel patterns, lesion boundaries).  
2. **Branch 2 (Adaptive MaxPooling)** — retains subtle lesion details (microaneurysms, soft exudates).  

Feature maps from both branches are **concatenated** and passed through a **shared convolutional block** with ReLU activation and MaxPooling for deeper representation learning.  
A **Spatial Pyramid Pooling (SPP)** layer follows, combining multi-scale features (1×1, 2×2, 4×4 pooling) into a **fixed-length feature vector** regardless of image resolution.  

The resulting vector passes through **fully connected layers** with **ReLU** and **dropout** for regularization, and finally a **binary classification output layer** (`Healthy` vs. `DR`).  

> 🧮 Total trainable parameters: **46,607,746**

---

## 🧼 Preprocessing Pipeline

Each fundus image undergoes the following steps before being fed into DASNet:

1. **Resize** to `224×224`  
2. **Convert** color space from BGR → HSV  
3. **Apply CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the **Value (V)** channel  
4. **Reconvert** back to RGB  

> This enhances local contrast and makes subtle lesions more visible under varying illumination conditions.

---

## 📚 Datasets Used

| **Dataset** | **Description** | **Label Format** |
|--------------|----------------|------------------|
| **BiDR** | Binary DR dataset (private) | 0 = Healthy, 1 = DR |
| **APTOS 2019** | Kaggle DR dataset | 0 = Healthy, 1 = DR |
| **Eye Disease Image (EDI)** | Public fundus dataset | 0 = Healthy, 1 = DR |
| **Unified Dataset** | Combined Eyepacs, Aptos, Messidor | 0 = Healthy, 1 = DR |

---

## ⚙️ Training Setup

| **Parameter** | **Value** |
|----------------|-----------|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Epochs | 25 |
| Batch Size | 32 |
| Validation Method | 3-Fold Cross Validation |
| Hardware | NVIDIA A100 (80GB) |

---

## 📈 Results Summary

| **Dataset** | **Accuracy (%)** | **Dataset Size** |
|--------------|------------------|------------------|
| BiDR         | 95.34            | 2,816 images     |
| APTOS 2019   | 95.65            | 3,610 images     |
| EDI          | 97.46            | 2,048 images     |

> DASNet demonstrates consistent performance across diverse datasets, achieving high accuracy and stability.

---

## 🔍 Ablation Study

### **APTOS Dataset**

| Model Variant | Accuracy | F1 Score | Precision | Recall |
|----------------|-----------|-----------|------------|---------|
| **DASNet (Full)** | 0.9534 | 0.9534 | 0.9536 | 0.9534 |
| DASNet (w/o Pyramid Pooling) | 0.9293 | 0.9293 | 0.9304 | 0.9293 |
| MaxPool CNN | 0.9489 | 0.9488 | 0.9499 | 0.9489 |
| AdaptivePool CNN | 0.9528 | 0.9527 | 0.9536 | 0.9528 |

### **BiDR Dataset**

| Model Variant | Accuracy | F1 Score | Precision | Recall |
|----------------|-----------|-----------|------------|---------|
| **DASNet (Full)** | 0.9565 | 0.9565 | 0.9568 | 0.9565 |
| DASNet (w/o Pyramid Pooling) | 0.9338 | 0.9337 | 0.9346 | 0.9338 |
| MaxPool CNN | 0.9488 | 0.9488 | 0.9489 | 0.9488 |
| AdaptivePool CNN | 0.9493 | 0.9493 | 0.9499 | 0.9493 |

### **EDI Dataset**

| Model Variant | Accuracy | F1 Score | Precision | Recall |
|----------------|-----------|-----------|------------|---------|
| **DASNet (Full)** | 0.9746 | 0.9746 | 0.9750 | 0.9746 |
| DASNet (w/o Pyramid Pooling) | 0.9311 | 0.9311 | 0.9322 | 0.9311 |
| MaxPool CNN | 0.9717 | 0.9720 | 0.9717 | 0.9717 |
| AdaptivePool CNN | 0.9673 | 0.9673 | 0.9677 | 0.9673 |

---

## 🖼️ Visualization

### **Adaptive Pooling Branch**
![Adaptive Pooling Branch](/images/normal.png)

### **MaxPooling Branch**
![MaxPooling Branch](/images/DR.png)

> Visualization highlights how the two branches specialize in detecting distinct retinal lesion patterns.

---

## 🔮 Future Work

- Integrate **clinical metadata** (e.g., age, blood sugar, blood pressure) for multimodal prediction.  
- Extend DASNet for **multi-class DR severity grading (0–4 levels)**.  
- Develop a **lightweight, mobile-ready version** for real-time telemedicine applications.

---

## 📚 Citation

If you use or reference DASNet in your work, please cite:

```bibtex
@inproceedings{sonale2026dasnet,
  title={DASNet: Dual Adaptive Subtle-Feature Network for Enhanced Diabetic Retinopathy Detection in Fundus Images},
  author={Sonale, Yadynesh and others},
  booktitle={Proceedings of the International Conference on Pattern Recognition Applications and Methods (ICPRAM)},
  year={2026}
}
