# HVX-Net-DR-Classification
HVX-Net: Hybrid CNN-ViT-XGBoost for Diabetic Retinopathy grading from retinal fundus images. APTOS 2019: 95.07% | EyePACS: 93.00% accuracy.
# HVX-Net: Hybrid CNN–ViT–XGBoost for Diabetic Retinopathy Diagnosis

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.3.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/timm-EfficientNet--B4%20%7C%20ViT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/XGBoost-Classifier-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/APTOS%202019-Acc%3A%2095.07%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/EyePACS-Acc%3A%2093.00%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

> **HVX-Net: A Hybrid Approach Based on CNN-ViT-XGBoost for Diagnosis of Diabetic Retinopathy from Retinal Fundus Images**
>
> D. HarshaVardhana Reddy, Shaik Haseena, S. Dev Kumar
> Dept. of Computer Science and Engineering, Vignan's Foundation for Science, Technology and Research, Guntur, AP, India

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Comparison with State-of-the-Art](#comparison-with-state-of-the-art)
- [Citation](#citation)

---

## Overview

Diabetic Retinopathy (DR) is a leading cause of preventable blindness worldwide. Early, accurate grading of DR severity from retinal fundus images is critical for clinical intervention. HVX-Net addresses this challenge through a novel three-stage hybrid architecture:

1. **Local feature extraction** — EfficientNet-B4 captures fine-grained lesion patterns (microaneurysms, exudates, haemorrhages)
2. **Global context modelling** — Vision Transformer (ViT-B/16) encodes long-range spatial dependencies via multi-head self-attention across 196 retinal patches
3. **Robust classification** — Fused features (2560-dimensional) are fed into an XGBoost classifier that handles class imbalance and outperforms a standard Softmax head

The model is evaluated on two public benchmarks — **APTOS 2019** and **EyePACS** — and outperforms all reported single-model baselines on both.

---

## Architecture

```
Retinal Fundus Image  (224 × 224 × 3)
         │
         ├──────────────────────┬──────────────────────────┐
         │                      │                          │
  EfficientNet-B4          ViT-B/16 (P=16)                │
  (CNN Branch)             (Transformer Branch)            │
  Global Avg Pool          CLS Token                       │
  F_CNN ∈ R^1792           F_ViT ∈ R^768                  │
         │                      │                          │
         └──────── Concat ───────┘                         │
                    │                                      │
           F_fused ∈ R^2560                                │
                    │                                      │
              XGBoost Classifier                           │
                    │                                      │
     DR Grade ∈ {0-Normal, 1-Mild, 2-Moderate,            │
                  3-Severe, 4-Proliferative}               │
```

| Component | Model | Output Dim | Parameters |
|-----------|-------|-----------|-----------|
| CNN Branch | EfficientNet-B4 | 1792 | 17.6 M |
| ViT Branch | ViT-B/16 (224) | 768 | 86.4 M |
| Fused Feature | Concatenation | **2560** | — |
| Classifier | XGBoost | 5 classes | — |
| **Total (backbone)** | | | **~104 M** |

---

## Datasets

| Dataset | Class | Total | Train | Test |
|---------|-------|------:|------:|-----:|
| APTOS 2019 | 0 – No DR | 1,805 | 1,444 | 361 |
| | 1 – Mild | 370 | 285 | 85 |
| | 2 – Moderate | 999 | 809 | 190 |
| | 3 – Severe | 193 | 160 | 33 |
| | 4 – Proliferative | 295 | 231 | 64 |
| **APTOS Total** | | **3,662** | **2,929** | **733** |
| EyePACS | 0 – No DR | 25,810 | 20,646 | 5,164 |
| | 1 – Mild | 2,443 | 1,932 | 511 |
| | 2 – Moderate | 5,292 | 4,246 | 1,046 |
| | 3 – Severe | 873 | 700 | 173 |
| | 4 – Proliferative | 708 | 576 | 132 |
| **EyePACS Total** | | **35,126** | **28,100** | **7,026** |

**Preprocessing pipeline:** Gaussian blur (kernel 5×5, σ=1.0) → Resize 224×224 → Normalize (ImageNet mean/std). Training augmentation includes random horizontal/vertical flips, ±15° rotation, and colour jitter.

---

## Results

### Performance on APTOS 2019

| Metric | Score |
|--------|------:|
| Accuracy | **95.07%** |
| Precision (macro) | **0.94** |
| Recall (macro) | **0.93** |
| F1-Score (macro) | **0.93** |
| ROC-AUC (macro OvR) | **0.98** |

### Performance on EyePACS

| Metric | Score |
|--------|------:|
| Accuracy | **93.00%** |
| Precision (macro) | **0.91** |
| Recall (macro) | **0.90** |
| F1-Score (macro) | **0.91** |
| ROC-AUC (macro OvR) | **0.96** |

### 5-Fold Cross-Validation Accuracy

| Dataset | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|---------|-------:|-------:|-------:|-------:|-------:|-----:|
| APTOS 2019 | 96.5% | 95.9% | 96.1% | 97.3% | 97.1% | **96.6%** |
| EyePACS | 92.1% | 93.9% | 93.0% | 91.1% | 93.0% | **92.6%** |

### Ablation Study (APTOS 2019)

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|--------:|--------:|--------:|
| CNN only (EfficientNet-B4) | 88.00% | 0.85 | 0.92 |
| ViT only (ViT-B/16) | 86.00% | 0.83 | 0.91 |
| CNN + ViT + Softmax | 88.00% | 0.86 | 0.93 |
| **CNN + ViT + XGBoost (HVX-Net)** | **95.07%** | **0.93** | **0.98** |

---

## Requirements

```
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

Install all dependencies:

```bash
pip install timm xgboost scikit-learn torchvision matplotlib seaborn opencv-python-headless
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/HVX-Net-DR-Classification.git
cd HVX-Net-DR-Classification
```

### 2. Prepare datasets

Download the datasets and organise them as follows:

```
DR_Data/
├── aptos2019/
│   ├── train.csv          # columns: id_code, diagnosis
│   └── train_images/      # .png retinal fundus images
└── eyepacs/
    ├── train_labels.csv   # columns: image, level
    └── train/             # .jpeg retinal fundus images
```

- **APTOS 2019:** [Kaggle — APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- **EyePACS:** [Kaggle — Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection)

### 3. Run in Google Colab (recommended)

Upload `HVX_Net_DR_Classification_FINAL.ipynb` to Google Colab, mount your Drive, update the dataset paths in **Cell 3**, then **Runtime → Run all**.

A free T4 GPU is sufficient. Full training takes approximately 3–4 hours.

### 4. Run locally

```bash
jupyter notebook HVX_Net_DR_Classification_FINAL.ipynb
```

Update `APTOS_CSV`, `APTOS_DIR`, `EYE_CSV`, `EYE_DIR` in Cell 3 to point to your local dataset paths.

---

## Repository Structure

```
HVX-Net-DR-Classification/
│
├── HVX_Net_DR_Classification_FINAL.ipynb   # Full training + evaluation notebook
├── README.md                               # This file
└── requirements.txt                        # Python dependencies
```

---

## Comparison with State-of-the-Art

| Author / Method | Dataset | Accuracy | Precision | Recall | F1 | AUC |
|----------------|---------|--------:|--------:|------:|---:|----:|
| Pratt et al. — CNN | APTOS | 91.3% | 0.90 | 0.89 | 0.89 | 0.94 |
| Abbasi et al. — Adaptive CNN | EyePACS | 92.1% | 0.91 | 0.90 | 0.90 | 0.95 |
| Jabbar et al. — Transfer Learning | EyePACS | 92.4% | 0.91 | 0.91 | 0.90 | 0.96 |
| Geetha & Hema — Joint CNN | APTOS | 92.6% | 0.92 | 0.91 | 0.91 | 0.96 |
| Dharrao et al. — EfficientNetB0 | APTOS | 93.0% | 0.92 | 0.92 | 0.92 | 0.97 |
| Akhtar et al. — DL Grading | EyePACS | 92.8% | 0.91 | 0.91 | 0.91 | 0.96 |
| Sathya & Valaramathi — Agentic AI | APTOS | 93.2% | 0.92 | 0.92 | 0.92 | 0.97 |
| Pandey & Alsheikheh — CNN | APTOS | 92.9% | 0.91 | 0.91 | 0.91 | 0.96 |
| Bhutnal & Moparthi — Lightweight CNN | EyePACS | 92.5% | 0.91 | 0.90 | 0.90 | 0.96 |
| ★ **HVX-Net (Proposed)** | **APTOS** | **95.07%** | **0.94** | **0.93** | **0.93** | **0.98** |
| ★ **HVX-Net (Proposed)** | **EyePACS** | **93.00%** | **0.91** | **0.90** | **0.91** | **0.96** |

HVX-Net achieves the highest accuracy on APTOS 2019, surpassing the next best method by **+1.87 percentage points**.

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{hvxnet2024,
  title   = {HVX-Net: A Hybrid Approach Based on CNN-ViT-XGBoost for
             Diagnosis of Diabetic Retinopathy from Retinal Fundus Images},
  author  = {HarshaVardhana Reddy, D. and Haseena, Shaik and Dev Kumar, S.},
  journal = {Journal of Computer Science and Engineering},
  year    = {2024},
  institution = {Vignan's Foundation for Science, Technology and Research,
                 Guntur, AP, India}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ at Vignan's Foundation for Science, Technology and Research, Guntur, India
</p>
