# HVX-Net-DR-Classification
HVX-Net: Hybrid CNN-ViT-XGBoost for Diabetic Retinopathy grading from retinal fundus images. APTOS 2019: 95.07% | EyePACS: 93.00% accuracy.
<div align="center">

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=E94560&center=true&vCenter=true&width=600&lines=HVX-Net" alt="HVX-Net"/>
</h1>

<h3>🩺 Hybrid CNN – ViT – XGBoost for Diabetic Retinopathy Diagnosis</h3>

<br/>

[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![timm](https://img.shields.io/badge/timm-EfficientNet--B4%20%7C%20ViT-orange?style=for-the-badge)](https://timm.fast.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

<br/>

| 🏆 Benchmark | 📊 Accuracy | 🎯 F1-Score | 📈 ROC-AUC |
|:---:|:---:|:---:|:---:|
| **APTOS 2019** | ![95.07%](https://img.shields.io/badge/95.07%25-brightgreen?style=flat-square) | ![0.93](https://img.shields.io/badge/0.93-brightgreen?style=flat-square) | ![0.98](https://img.shields.io/badge/0.98-brightgreen?style=flat-square) |
| **EyePACS** | ![93.00%](https://img.shields.io/badge/93.00%25-brightgreen?style=flat-square) | ![0.91](https://img.shields.io/badge/0.91-brightgreen?style=flat-square) | ![0.96](https://img.shields.io/badge/0.96-brightgreen?style=flat-square) |

<br/>

> **D. HarshaVardhana Reddy · Shaik Haseena · S. Dev Kumar**
> 
> *Department of Computer Science and Engineering*
> *Vignan's Foundation for Science, Technology and Research, Guntur, AP, India*

</div>

---

## 📋 Table of Contents

- [🔬 Overview](#-overview)
- [🧠 Architecture](#-architecture)
- [📂 Datasets](#-datasets)
- [📊 Results](#-results)
- [⚙️ Requirements](#️-requirements)
- [🚀 Getting Started](#-getting-started)
- [📁 Repository Structure](#-repository-structure)
- [🏅 Comparison with State-of-the-Art](#-comparison-with-state-of-the-art)
- [📜 Citation](#-citation)
- [📄 License](#-license)

---

## 🔬 Overview

**Diabetic Retinopathy (DR)** is a leading cause of preventable blindness, affecting over 100 million people worldwide. Accurate automated grading of DR severity from retinal fundus images is critical for early clinical intervention.

**HVX-Net** is a novel three-stage hybrid deep learning architecture that combines the complementary strengths of:

| Stage | Component | Role |
|:---:|:---:|:---|
| 1️⃣ | **EfficientNet-B4** | Extracts fine-grained *local* lesion features — microaneurysms, exudates, haemorrhages |
| 2️⃣ | **ViT-B/16** | Models *global* spatial context via multi-head self-attention across 196 retinal patches |
| 3️⃣ | **XGBoost** | Classifies the 2560-dim fused feature vector with boosted tree ensembles, handling class imbalance |

> 💡 The XGBoost classifier on fused features consistently outperforms a standard Softmax head by **+7 percentage points** on APTOS 2019.

---

## 🧠 Architecture

```
                    Retinal Fundus Image (224×224×3)
                               │
               ┌───────────────┴────────────────┐
               │                                │
       ┌───────▼───────┐                ┌───────▼───────┐
       │ EfficientNet  │                │   ViT-B/16    │
       │     B4        │                │   (P = 16)    │
       │               │                │               │
       │ Conv Layers   │                │ 196 Patches   │
       │ + BN + ReLU   │                │ + CLS Token   │
       │ + GAP         │                │ + MHSA × 12   │
       └───────┬───────┘                └───────┬───────┘
               │                                │
        F_CNN ∈ ℝ¹⁷⁹²                   F_ViT ∈ ℝ⁷⁶⁸
               │                                │
               └──────────┬─────────────────────┘
                          │  Concatenation
                          │
                  F_fused ∈ ℝ²⁵⁶⁰
                          │
                  ┌───────▼───────┐
                  │   XGBoost     │
                  │  Classifier   │
                  │ (300 trees)   │
                  └───────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       No DR            Mild          Moderate
                                          │
                              ┌───────────┴──────────┐
                           Severe             Proliferative
```

### 📐 Model Summary

| Component | Backbone | Output Dim | Parameters |
|:---|:---|:---:|---:|
| 🔵 CNN Branch | EfficientNet-B4 | 1,792 | 17.6 M |
| 🟢 ViT Branch | ViT-B/16 (patch 16, img 224) | 768 | 86.4 M |
| 🔴 Fused Feature | Concatenation | **2,560** | — |
| ⚡ Classifier | XGBoost (n=300, depth=6) | 5 classes | — |
| **Total (backbone)** | | | **~104 M** |

**Feature Fusion Formula:**
$$F_{fused} = [F_{CNN} \| F_{ViT}] \in \mathbb{R}^{d_1 + d_2} = \mathbb{R}^{2560}$$

---

## 📂 Datasets

<details>
<summary><b>📌 APTOS 2019 — Click to expand</b></summary>

| Class | Label | Total | Train | Test |
|:---|:---:|---:|---:|---:|
| No DR | 0 | 1,805 | 1,444 | 361 |
| Mild | 1 | 370 | 285 | 85 |
| Moderate | 2 | 999 | 809 | 190 |
| Severe | 3 | 193 | 160 | 33 |
| Proliferative | 4 | 295 | 231 | 64 |
| **Total** | | **3,662** | **2,929** | **733** |

🔗 [Download from Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)

</details>

<details>
<summary><b>📌 EyePACS — Click to expand</b></summary>

| Class | Label | Total | Train | Test |
|:---|:---:|---:|---:|---:|
| No DR | 0 | 25,810 | 20,646 | 5,164 |
| Mild | 1 | 2,443 | 1,932 | 511 |
| Moderate | 2 | 5,292 | 4,246 | 1,046 |
| Severe | 3 | 873 | 700 | 173 |
| Proliferative | 4 | 708 | 576 | 132 |
| **Total** | | **35,126** | **28,100** | **7,026** |

🔗 [Download from Kaggle](https://www.kaggle.com/c/diabetic-retinopathy-detection)

</details>

**🔧 Preprocessing Pipeline:**

```
Raw Image → Gaussian Blur (5×5, σ=1.0) → Resize (224×224) → Normalize (ImageNet μ/σ)
```

**🔄 Training Augmentation:** Random H/V flip · ±15° rotation · Colour jitter (brightness, contrast)

---

## 📊 Results

### 🏆 APTOS 2019 — Test Set Performance

| Metric | No DR | Mild | Moderate | Severe | Proliferative | **Macro Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Precision | 0.97 | 0.91 | 0.95 | 0.88 | 0.96 | **0.94** |
| Recall | 0.96 | 0.92 | 0.96 | 0.88 | 0.94 | **0.93** |
| F1-Score | 0.97 | 0.91 | 0.95 | 0.88 | 0.95 | **0.93** |
| **Accuracy** | | | | | | **95.07%** |
| **ROC-AUC** | | | | | | **0.98** |

### 🏆 EyePACS — Test Set Performance

| Metric | No DR | Mild | Moderate | Severe | Proliferative | **Macro Avg** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Precision | 0.96 | 0.85 | 0.91 | 0.87 | 0.94 | **0.91** |
| Recall | 0.96 | 0.84 | 0.92 | 0.86 | 0.92 | **0.90** |
| F1-Score | 0.96 | 0.85 | 0.91 | 0.87 | 0.93 | **0.91** |
| **Accuracy** | | | | | | **93.00%** |
| **ROC-AUC** | | | | | | **0.96** |

### 📉 Ablation Study (APTOS 2019)

| Model Variant | Accuracy | Precision | Recall | F1 | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| CNN only (EfficientNet-B4) | 88.00% | 0.86 | 0.85 | 0.85 | 0.92 |
| ViT only (ViT-B/16) | 86.00% | 0.84 | 0.83 | 0.83 | 0.91 |
| CNN + ViT + Softmax | 88.00% | 0.87 | 0.86 | 0.86 | 0.93 |
| ✅ **CNN + ViT + XGBoost (HVX-Net)** | **95.07%** | **0.94** | **0.93** | **0.93** | **0.98** |

### 🔁 5-Fold Cross-Validation

| Dataset | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean ± Std** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| APTOS 2019 | 96.5% | 95.9% | 96.1% | 97.3% | 97.1% | **96.6% ± 0.56** |
| EyePACS | 92.1% | 93.9% | 93.0% | 91.1% | 93.0% | **92.6% ± 1.04** |

---

## ⚙️ Requirements

```txt
Python        >= 3.9
PyTorch       >= 2.0.0
torchvision   >= 0.15.0
timm          >= 0.9.0
xgboost       >= 1.7.0
scikit-learn  >= 1.2.0
opencv-python >= 4.7.0
numpy         >= 1.24.0
pandas        >= 1.5.0
matplotlib    >= 3.7.0
seaborn       >= 0.12.0
```

```bash
pip install timm xgboost scikit-learn torchvision matplotlib seaborn opencv-python-headless
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/221fa04470/HVX-Net-DR-Classification.git
cd HVX-Net-DR-Classification
```

### 2️⃣ Prepare Datasets

Organise your data as follows:

```
DR_Data/
├── aptos2019/
│   ├── train.csv              # columns: id_code, diagnosis
│   └── train_images/          # PNG retinal fundus images
└── eyepacs/
    ├── train_labels.csv       # columns: image, level
    └── train/                 # JPEG retinal fundus images
```

### 3️⃣ Update Dataset Paths

In **Cell 3** of the notebook, update:

```python
APTOS_CSV = '/content/drive/MyDrive/DR_Data/aptos2019/train.csv'
APTOS_DIR = '/content/drive/MyDrive/DR_Data/aptos2019/train_images/'
EYE_CSV   = '/content/drive/MyDrive/DR_Data/eyepacs/train_labels.csv'
EYE_DIR   = '/content/drive/MyDrive/DR_Data/eyepacs/train/'
```

### 4️⃣ Run

**Google Colab (recommended):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Upload the notebook to Colab → Mount Drive → **Runtime → Run all**
> Free T4 GPU is sufficient · Full training ≈ 3–4 hours

**Local Jupyter:**
```bash
jupyter notebook HVX_Net_DR_Classification_FINAL.ipynb
```

---

## 📁 Repository Structure

```
HVX-Net-DR-Classification/
│
├── 📓 HVX_Net_DR_Classification_FINAL.ipynb   # Full pipeline: training + evaluation
├── 📄 README.md                               # Project documentation
├── 📋 requirements.txt                        # Python dependencies
├── 📜 LICENSE                                 # MIT License
└── 🔒 .gitignore                              # Python gitignore
```

---

## 🏅 Comparison with State-of-the-Art

| # | Author / Method | Dataset | Accuracy | Prec. | Recall | F1 | AUC |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Pratt et al. — CNN | APTOS | 91.3% | 0.90 | 0.89 | 0.89 | 0.94 |
| 2 | Abbasi et al. — Adaptive CNN | EyePACS | 92.1% | 0.91 | 0.90 | 0.90 | 0.95 |
| 3 | Jabbar et al. — Transfer Learning | EyePACS | 92.4% | 0.91 | 0.91 | 0.90 | 0.96 |
| 4 | Geetha & Hema — Joint CNN | APTOS | 92.6% | 0.92 | 0.91 | 0.91 | 0.96 |
| 5 | Dharrao et al. — EfficientNetB0 | APTOS | 93.0% | 0.92 | 0.92 | 0.92 | 0.97 |
| 6 | Akhtar et al. — DL Grading | EyePACS | 92.8% | 0.91 | 0.91 | 0.91 | 0.96 |
| 7 | Sathya & Valaramathi — Agentic AI | APTOS | 93.2% | 0.92 | 0.92 | 0.92 | 0.97 |
| 8 | Pandey & Alsheikheh — CNN | APTOS | 92.9% | 0.91 | 0.91 | 0.91 | 0.96 |
| 9 | Bhutnal & Moparthi — Lightweight CNN | EyePACS | 92.5% | 0.91 | 0.90 | 0.90 | 0.96 |
| 🥇 | **HVX-Net — Proposed** | **APTOS** | **95.07%** | **0.94** | **0.93** | **0.93** | **0.98** |
| 🥇 | **HVX-Net — Proposed** | **EyePACS** | **93.00%** | **0.91** | **0.90** | **0.91** | **0.96** |

> 🚀 HVX-Net achieves **+1.87 pp** improvement over the next-best method on APTOS 2019.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{hvxnet2026,
  title        = {HVX-Net: A Hybrid Approach Based on CNN-ViT-XGBoost for
                  Diagnosis of Diabetic Retinopathy from Retinal Fundus Images},
  author       = {HarshaVardhana Reddy, D. and Haseena, Shaik and Dev Kumar, S.},
  journal      = {Journal of Computer Science and Engineering},
  year         = {2026},
  institution  = {Vignan's Foundation for Science, Technology and Research,
                  Guntur, AP, India}
}
```

---

## 📄 License

This project is released under the [MIT License](LICENSE) © 2026 D. HarshaVardhana Reddy, Shaik Haseena, S. Dev Kumar.

---

<div align="center">

**⭐ If this work helped you, please consider starring the repository ⭐**

<br/>

Made with ❤️ at **Vignan's Foundation for Science, Technology and Research**, Guntur, India

---
*© 2026 Vignan's Foundation for Science, Technology and Research, Guntur, India*

</div>
</div>nan's Foundation for Science, Technology and Research, Guntur, India
</p>
