import os
import io
import base64
import tempfile
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from fpdf import FPDF

# Load secure environment configuration
load_dotenv()

st.set_page_config(
    page_title="PhytoVision AI Enterprise",
    page_icon="🌾",
    layout="wide"
)

# Configuration from environment with secure defaults
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("PHYTOVISION_API_KEY", "key_drone_corp_2026")

# ----------------- PDF GENERATOR ENGINE -----------------
def generate_pdf_report(diagnosis, confidence, severity, risk, action, original_image_bytes, heatmap_base64):
    """Generates a professional B2B agronomic PDF report including visual evidence."""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, txt="PhytoVision AI - Official Agronomic Report", ln=True, align='C')
    pdf.set_font("Arial", 'I', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, txt="Automated AI Pathology Diagnostic & Severity Analysis", ln=True, align='C')
    pdf.ln(5)
    
    # Visual Evidence Handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_orig:
        tmp_orig.write(original_image_bytes)
        orig_path = tmp_orig.name
        
    heatmap_path = None
    if heatmap_base64:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_heat:
            tmp_heat.write(base64.b64decode(heatmap_base64))
            heatmap_path = tmp_heat.name

    # Determine layout based on whether a heatmap exists
    if heatmap_path:
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(95, 10, txt="Original Specimen", align='C')
        pdf.cell(95, 10, txt="Lesion Activation / Thermal Mapping", ln=True, align='C')
        
        pdf.image(orig_path, x=15, y=pdf.get_y(), w=80)
        pdf.image(heatmap_path, x=115, y=pdf.get_y(), w=80)
        pdf.ln(70) 
    else:
        pdf.set_font("Arial", 'B', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, txt="Original Specimen (No Necrotic Lesions Detected)", ln=True, align='C')
        pdf.image(orig_path, x=65, y=pdf.get_y(), w=80)
        pdf.ln(70)

    # Data Table
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    
    pdf.cell(60, 10, txt="Primary Diagnosis:", border=1, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(130, 10, txt=f" {diagnosis} ({confidence:.2f}% Confidence)", border=1, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, txt="Severity Index:", border=1, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(130, 10, txt=f" {severity:.1f}%", border=1, ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, txt="Operational Risk:", border=1, fill=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(130, 10, txt=f" {risk}", border=1, ln=True)
    
    # Advisory Directive
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="Agronomic Advisory Directive:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 8, txt=action)
    
    # Cleanup & Export
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            pdf_bytes = f.read()
            
    os.remove(orig_path)
    if heatmap_path:
        os.remove(heatmap_path)
    os.remove(tmp_pdf.name)
        
    return pdf_bytes

# ----------------- SIDEBAR -----------------
st.sidebar.title("🌾 PhytoVision AI")
st.sidebar.markdown("**Enterprise Agritech Diagnostic Engine**")
st.sidebar.markdown("---")

api_endpoint = st.sidebar.text_input("Backend Gateway", value=BACKEND_URL)
tenant_key = st.sidebar.text_input("Tenant API Key", value=API_KEY, type="password")

try:
    health_resp = requests.get(f"{api_endpoint}/health", timeout=2)
    if health_resp.status_code == 200:
        st.sidebar.success("Gateway: Online (200 OK)")
    else:
        st.sidebar.warning(f"Gateway: Status {health_resp.status_code}")
except Exception:
    st.sidebar.error("Gateway: Disconnected")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Platform Specs:**
    - Architecture: `MobileNetV4`
    - Engine: `ONNX Runtime`
    - Explainability: `Lesion Heatmap`
    """
)

# ----------------- MAIN UI ROUTING -----------------
st.title("Enterprise Crop Health Gateway")

tab_scanner, tab_analytics = st.tabs(["🧬 Diagnostic Terminal", "📊 Enterprise Analytics"])

# === TAB 1: DIAGNOSTIC TERMINAL ===
with tab_scanner:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("1. Ingest Plant Specimen")
        uploaded_file = st.file_uploader("Upload Leaf Specimen (JPG, PNG)", type=["jpg", "jpeg", "png"])
        inference_requested = False
        if uploaded_file:
            inference_requested = st.button("Run Diagnostic Inference", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Ingested Specimen")
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="RGB Raw Specimen", use_container_width=True)
        else:
            st.info("Awaiting specimen ingestion...")

    if inference_requested and uploaded_file:
        st.markdown("---")
        st.subheader("3. Diagnostic Intelligence")
        headers = {"x-api-key": tenant_key}
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

        with st.spinner("Executing neural inference & lesion localization..."):
            try:
                response = requests.post(f"{api_endpoint}/analyze", headers=headers, files=files, timeout=20)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Diagnosis", result.get("diagnosis", "N/A"))
                    m2.metric("Confidence", f"{result.get('confidence', 0.0):.2f}%")
                    m3.metric("Severity", f"{result.get('severity', 0.0):.1f}%", delta=f"Class: {result.get('severity_class', 'N/A')}", delta_color="inverse")
                    m4.metric("Operational Risk", result.get("risk_level", "N/A"))
                    
                    # Visuals
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.markdown("**Original Specimen**")
                        st.image(Image.open(uploaded_file), use_container_width=True)
                    with vis_col2:
                        st.markdown("**Pathology Explainability (Lesion Heatmap)**")
                        heatmap_data = result.get("heatmap_base64")
                        if heatmap_data:
                            heatmap_bytes = base64.b64decode(heatmap_data)
                            st.image(Image.open(io.BytesIO(heatmap_bytes)), caption="Lesion Activation / Thermal Mapping", use_container_width=True)
                        else:
                            st.success("No active necrotic lesions identified on crop specimen.")

                    # Action & PDF
                    st.markdown("### Agronomic Advisory Directive")
                    if result.get("diagnosis") == "Healthy":
                        st.success(f"**Directive:** {result.get('action')}")
                    else:
                        st.warning(f"**Directive:** {result.get('action')}")
                    
                    pdf_bytes = generate_pdf_report(
                        diagnosis=result.get("diagnosis"), 
                        confidence=result.get("confidence"), 
                        severity=result.get("severity"), 
                        risk=result.get("risk_level"), 
                        action=result.get("action"),
                        original_image_bytes=uploaded_file.getvalue(),
                        heatmap_base64=result.get("heatmap_base64")
                    )
                    
                    st.download_button(
                        label="📄 Download Official PDF Report",
                        data=pdf_bytes,
                        file_name="PhytoVision_Diagnostic_Report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

                    # --- ADD THIS DIRECTLY BELOW YOUR PDF DOWNLOAD BUTTON ---
                    st.markdown("---")
                    st.markdown("### Continuous Learning (MLOps)")
                    st.caption("Human-in-the-loop validation. Flag this sample if the AI diagnosis appears incorrect.")
                    
                    if st.button("🚩 Flag as Hard Sample (Send to MLOps Queue)"):
                        with st.spinner("Transmitting to active learning queue..."):
                            flag_files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                            flag_data = {"actual_diagnosis": result.get("diagnosis", "Unknown")}
                            
                            try:
                                flag_resp = requests.post(
                                    f"{api_endpoint}/flag", 
                                    headers=headers, 
                                    files=flag_files,
                                    data=flag_data
                                )
                                if flag_resp.status_code == 200:
                                    st.success("Sample securely vaulted for agronomist review and future model retraining.")
                                else:
                                    st.error("Failed to flag sample.")
                            except Exception as e:
                                st.error(f"Queue Error: {e}"
                                         )
                elif response.status_code == 403:
                    st.error("Authentication Denied: Check Enterprise API Key.")
                else:
                    st.error(f"Inference Failure ({response.status_code}): {response.text}")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend gateway at `{api_endpoint}`.")
            except Exception as e:
                st.error(f"Processing Error: {e}")

# === TAB 2: ENTERPRISE ANALYTICS ===
with tab_analytics:
    st.subheader("Historical Field Diagnostics")
    
    if st.button("Refresh Telemetry Data"):
        headers = {"x-api-key": tenant_key}
        try:
            response = requests.get(f"{api_endpoint}/analytics", headers=headers)
            
            if response.status_code == 200:
                data = response.json().get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Diagnostic Events", len(df))
                    c2.metric("Average Severity", f"{df['severity'].mean():.1f}%")
                    
                    diseases_only = df[df['disease'] != "Healthy"]
                    top_issue = diseases_only['disease'].mode()[0] if not diseases_only.empty else "None"
                    c3.metric("Primary Field Threat", top_issue)
                    
                    st.markdown("**Disease Severity Trend (Over Time)**")
                    chart_data = df[['timestamp', 'severity']].set_index('timestamp')
                    st.line_chart(chart_data)
                    
                    st.markdown("**Raw Audit Logs**")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No diagnostic events recorded for this tenant yet. Run a scan in the Diagnostic Terminal.")
            else:
                st.error("Failed to retrieve analytics data from Gateway.")
        except Exception as e:
             st.error(f"Connection Error: {e}")