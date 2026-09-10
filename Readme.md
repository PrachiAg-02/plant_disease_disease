# 🌾 PhytoVision AI: Enterprise Agritech Diagnostic & Severity Engine

> **Solution Challenge 2026 – Build with AI**  
> An enterprise-grade, containerized computer vision platform designed for real-time crop disease detection, severity quantification, and automated agronomic reporting.

---

## 🚀 Overview
PhytoVision AI bridges the gap between advanced deep learning and practical agricultural management. Utilizing optimized **ONNX runtime models** combined with a high-performance **FastAPI** backend and an interactive **Streamlit** enterprise dashboard, the platform enables agronomists and farmers to rapidly diagnose crop pathologies, calculate precise infection severities, and generate actionable multi-lingual PDF reports.

---

## 🛠️ Tech Stack & Architecture

* **Frontend:** Streamlit (Enterprise Dashboard UI)
* **Backend:** FastAPI (Async RESTful API Services)
* **Core Intelligence:** Computer Vision, ONNX Runtime (Optimized Inference Engine)
* **Deployment & Containerization:** Docker & Docker Compose (Multi-container orchestration)
* **Reporting:** Automated PDF generation with localized insights

---

## 📂 Project Structure
```text
plant_disease_disease/
├── app.py                  # Streamlit Frontend Application
├── main.py                 # FastAPI Backend & Inference Engine
├── Dockerfile.backend      # Container configuration for FastAPI
├── Dockerfile.frontend     # Container configuration for Streamlit
├── docker-compose.yml      # Multi-container orchestration setup
├── requirements.txt        # Python dependencies
└── .dockerignore           # Build optimization rules