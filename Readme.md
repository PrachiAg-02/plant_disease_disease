# 🌿 PhytoVision AI: Plant Pathology Diagnostic & Explainable Advisory Platform

[![CI Pipeline](https://github.com/PrachiAg-02/plant_disease_disease/actions/workflows/ci.yml/badge.svg)](https://github.com/PrachiAg-02/plant_disease_disease/actions)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/PyTorch-MobileNetV4-EE4C2C.svg)
![Backend](https://img.shields.io/badge/FastAPI-Sub--50ms-009688.svg)
![UI](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![Edge](https://img.shields.io/badge/Edge-ONNX_Runtime-005CED.svg)

An Explainable AI (XAI) computer vision pipeline engineered for foliar crop disease diagnosis, localized lesion heatmapping, and structured agronomic intervention plans.

---

## 🏗️ System Architecture

```text
Image upload
     │
     ▼
Image validation (Format: JPG/PNG, Size: <10MB)
     │
     ▼
Preprocessing (Resize 224x224, ImageNet Normalization)
     │
     ▼
ONNX / MobileNetV4 inference (Sub-10ms latency)
     │
     ▼
Confidence threshold (Threshold: 65.0%)
     ├── Low confidence (<65%)  ──► Inconclusive warning & retake guidance
     └── Accepted result (≥65%) ──► Disease classification + Grad-CAM heatmap + Agronomic Advisory PDF