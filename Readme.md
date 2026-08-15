# 🌿 PhytoVision AI: Plant Pathology Diagnostic & Advisory Platform

An end-to-end deep learning plant disease identification and agronomic remediation platform powered by **MobileNetV4**, **FastAPI**, and **Streamlit**.

---

## 🌟 Key Features

- **Deep Learning Classifier:** Fine-tuned MobileNetV4 convolutional backbone trained on high-resolution crop foliar disease datasets.
- **Sub-50ms Inference API:** Asynchronous FastAPI backend delivering real-time predictions and softmax probability distributions.
- **Automated Agronomic Advisory:** Structured treatment protocols categorizing chemical, organic bio-control, and cultural prevention measures.
- **Modern Interactive Dashboard:** Dark-theme glassmorphism UI built with Streamlit and Plotly telemetry charts.
- **Enterprise Quality Verification:** Fully automated `pytest` test suite integrated with GitHub Actions CI/CD.

---

## 🏗️ Architecture

```text
[ User / Field Image ] 
         │
         ▼
[ Streamlit Web UI (Port 8501) ]
         │ (HTTP POST /diagnose)
         ▼
[ FastAPI Backend (Port 8000) ]
         │
         ├──► [ MobileNetV4 Model (.pth) ] ──► Softmax Probabilities
         │
         └──► [ Agronomy Rules / Fallback ] ──► Remediation Protocols