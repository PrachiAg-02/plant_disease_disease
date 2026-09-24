# 🌾 PhytoVision AI Enterprise

**B2B Agritech Diagnostic & Severity Intelligence Gateway**

PhytoVision AI is an enterprise-grade, containerized computer vision platform engineered to deliver real-time plant pathology inference, explainable AI diagnostics, and persistent operational telemetry for commercial agriculture. 

Developed as a capstone engineering solution for the Google Solution Challenge 2026 and refined with principles from the Gen AI Academy APAC Edition, this platform bridges the gap between raw machine learning models and production-ready B2B software.

**Lead Developer:** Prachi Agarwal

---

## 🚀 Enterprise Features

- **Edge-Optimized Inference:** Powered by MobileNetV4 and ONNX Runtime for ultra-fast, lightweight diagnostic execution.
- **Visual Explainability (XAI):** Generates real-time localized lesion heatmaps using computer vision to prove *why* the AI made its diagnosis, building trust with agronomists.
- **Severity Quantification:** Algorithmic calculation of infected tissue percentage to categorize operational risk (Mild, Moderate, Severe).
- **Automated Agronomic Reporting:** Dynamically generates downloadable, professional PDF reports embedding visual evidence and treatment directives.
- **Data Persistence & Analytics:** SQLite integration for secure tenant telemetry, tracking historical field health and severity trends over time.
- **MLOps Active Learning Queue:** Built-in human-in-the-loop flagging mechanism to route hard samples back to a storage queue for continuous model retraining.
- **Containerized Architecture:** Fully decoupled FastAPI backend and Streamlit frontend orchestrated via Docker Compose.

---

## 🏗️ System Architecture

1. **Frontend (Streamlit):** Multi-tenant B2B dashboard with isolated routing for diagnostics and historical analytics.
2. **Gateway (FastAPI):** High-performance, asynchronous REST API secured via `.env` API key headers.
3. **Vision Engine (OpenCV):** Custom segmentation pipeline for isolating biological matter and quantifying necrotic lesions.
4. **Persistence Layer (SQLAlchemy):** Relational database mapping Organizations, Farms, and Diagnostic Events.

---

## 🛠️ Quickstart Deployment

Deploy the entire microservice ecosystem locally using Docker.

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/phytovision-ai.git](https://github.com/yourusername/phytovision-ai.git)
cd phytovision-ai