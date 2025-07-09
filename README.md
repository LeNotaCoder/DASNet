# DASNet: A Dual Adaptive Subtle-Feature Network for Enhanced Diabetic Retinopathy Detection in Fundus Images

![DASNet Architecture](/images/model.png) <!-- Replace with actual path to your model architecture image -->

## 📄 Overview

**DASNet** is a novel deep learning architecture designed to improve the early detection of **Diabetic Retinopathy (DR)** using color fundus images (CFIs). The model addresses the challenge of capturing both **subtle lesions** (e.g., microaneurysms) and **prominent features** (e.g., hemorrhages, optic disc changes) by introducing a **dual-branch CNN** that leverages **Adaptive MaxPooling** and standard **MaxPooling** in parallel.

This model was proposed at the **12th Indian Conference on Computer Vision, Graphics and Image Processing (ICVGIP’25)** and is currently under review.

---

## 🧠 Key Contributions

- **Dual-Branch CNN (DASNet):** Combines adaptive and standard pooling to effectively capture fine-grained and coarse features.
- **Robust Preprocessing Pipeline:** Utilizes HSV conversion and CLAHE on the V-channel to enhance contrast and brightness.
- **Superior Performance:** Outperforms state-of-the-art models (AlexNet, VGG16, ResNet50, InceptionV3) on four datasets.
- **Interpretability:** Visualizes feature maps from each branch to illustrate how different lesions are detected.

---

## 📊 Results

| Dataset     | DASNet Accuracy | Dataset Size|
|-------------|------------------|-------------|
| BiDR        | 95.34%           | 2816 images|
| APTOS       | 94.77%           | 3610 images |
| EDI         | 94.51%           | 2048 images |

---

## 🔍 Ablation Study
On the APTOS dataset,

- Removing **Adaptive Pooling** reduced APTOS accuracy to **92.1%**
- Removing **Max Pooling** reduced APTOS accuracy to **93.6%**
- Removing **Preprocessing** reduced BiDR accuracy to **94.19%**
- Combining both branches and preprocessing led to the highest performance

---

## 🛠️ Model Architecture

DASNet has the following structure:

- Two parallel CNN branches:
  - Branch 1: Conv layers + **Adaptive MaxPooling**
  - Branch 2: Conv layers + **MaxPooling**
- Feature map concatenation
- Deep convolutional stack + FC layers
- Dropout regularization
- Final binary classification (Healthy vs DR)

---

![DASNet Architecture](/images/pipeline.png)

## 🖼️ Preprocessing Pipeline

1. Resize images to `224×224`
2. Convert BGR → HSV
3. Apply **CLAHE** on Value (V) channel
4. Reconvert to RGB

This enhances subtle lesion visibility across lighting conditions.

---

## 📁 Datasets Used

1. **BiDR**
2. **APTOS 2019**
3. **Eye Disease Image (EDI)**
4. **Unified Dataset** (Eyepacs, Aptos, Messidor)

All datasets were normalized into binary labels: `0 = Healthy`, `1 = DR`.

---

## 🧪 Training Setup

- Optimizer: `Adam`
- Loss: `Binary Cross-Entropy`
- Epochs: `25`
- Batch Size: `32`
- Hardware: `NVIDIA A100 (80GB)`
- Train/Test Split: `80/20`

---

## 📈 Visualization

![Adaptive Pooling branch](/images/adaptive_pool.png)

![MaxPooling branch](/images/maxpool.png)

Feature maps from:
To interpret DASNet’s internal workings, the 10 most activated feature maps from both branches were visualized. The Adaptive MaxPooling branch captured fine-grained, subtle features such as microaneurysms and vessel irregularities, with dense, spatially distributed activations. In contrast, the MaxPooling branch highlighted prominent anatomical structures like the optic disc and hemorrhages, with sharper, more localized activations. These complementary visual patterns validate DASNet's dual-branch design, demonstrating its ability to capture both subtle and dominant retinal features, enhancing both performance and model interpretability.

---
