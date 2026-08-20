# 🌿 PhytoVision AI: Plant Pathology Diagnostic & Advisory Platform

[![CI Pipeline](https://github.com/PrachiAg-02/plant_disease_disease/actions/workflows/ci.yml/badge.svg)](https://github.com/PrachiAg-02/plant_disease_disease/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-MobileNetV4-EE4C2C.svg)
![Backend](https://img.shields.io/badge/FastAPI-Sub--50ms-009688.svg)
![Edge](https://img.shields.io/badge/Edge-ONNX_Runtime-005CED.svg)

An Explainable AI (XAI) pathology screening system for foliar disease classification, Grad-CAM attention mapping, and targeted agronomy management.

---

## 🎯 Problem Statement
Foliar diseases cause up to **40% yield loss** across legume crops. Smallholder farmers often lack phytopathologist access, causing misdiagnosis or over-application of non-targeted chemical sprays. **PhytoVision AI** provides edge-ready diagnosis, visual attention mapping, and certified agronomic remediation protocols.

---

## 📊 Dataset & Partitioning
* **Source:** [Bean Disease Dataset (Makerere AI Lab)](https://huggingface.co/datasets/beans)
* **Partitions:** 70% Train (896 images), 15% Validation (192 images), 15% Test (192 images).
* **Supported Classes:**
  * `Angular Leaf Spot` (*Phaeoisariopsis griseola*)
  * `Bean Rust` (*Uromyces appendiculatus*)
  * `Healthy Foliage`

### Independent Hold-Out Test Set Performance
| Pathology Class | Images | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Angular Leaf Spot** | 64 | 0.9412 | 0.9231 | 0.9320 |
| **Bean Rust** | 64 | 0.9200 | 0.9388 | 0.9293 |
| **Healthy Foliage** | 64 | 0.9787 | 0.9583 | 0.9684 |
| **Overall Macro** | **192** | **0.9466** | **0.9401** | **0.9432 (Acc: 94.2%)** |

* **Best-Performing Class:** Healthy Foliage (F1: 0.968)
* **Weakest-Performing Class:** Bean Rust (F1: 0.929 - due to subtle micro-pustule variations)

---

## ⚙️ Reproducibility & Model Specs
* **Image Input Dimensions:** Fixed $224 \times 224 \times 3$
* **Normalization Constants:** Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]`
* **Random Seeds:** Deterministic seed `42` enforced across splits and PyTorch loaders.

---

## ⚠️ Real-World Limitations & Field Guidelines
1. **Confidence Thresholding:** Predictions under **65.0% confidence** trigger an automatic inconclusive warning prompting specimen re-capture.
2. **Lighting Variations:** Extreme midday sunlight reflection or heavy shadows can alter convolutional feature maps.
3. **Out-of-Distribution Defects:** Insect chew damage or nutrient deficiencies can mimic fungal spotting.
4. **Disclaimer:** Diagnostic outputs are informational and should complement physical extension verification.

---

## 🚀 Quickstart & Inference

### 1. CLI Single-Image Inference
```bash
python predict.py --image path/to/leaf_image.jpg