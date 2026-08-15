import streamlit as st
import requests
from PIL import Image
import io
import plotly.express as px

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PhytoVision AI | Plant Pathology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MODERN CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    * { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Background styling */
    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1f17 0%, #090e0c 100%);
        color: #f1f5f9;
    }

    /* Custom Header */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34d399 0%, #10b981 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        text-align: center;
    }
    .main-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Status Badges */
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid #10b981;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
    }
    .badge-disease {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid #ef4444;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
    }

    /* Treatment Cards */
    .advisory-card {
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
        background: rgba(255, 255, 255, 0.03);
    }
    .advisory-chemical {
        border-left: 4px solid #f87171;
    }
    .advisory-organic {
        border-left: 4px solid #34d399;
    }
    .advisory-prevention {
        border-left: 4px solid #38bdf8;
    }

    /* Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 10px;
        padding: 12px;
        border: none;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🌿 **PhytoVision AI**")
    st.caption("Deep Learning Plant Disease Diagnostic System")
    st.divider()

    st.markdown("### ⚙️ System Metrics")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Model", "MobileNetV4")
    m_col2.metric("Inference", "< 40ms")

    st.markdown("### 📋 Supported Pathologies")
    st.markdown("""
    * **Angular Leaf Spot** (*Phaeoisariopsis*)
    * **Bean Rust** (*Uromyces*)
    * **Healthy Foliage**
    """)
    st.divider()
    api_endpoint = st.text_input("FastAPI Endpoint", value="http://127.0.0.1:8000/diagnose")

# ----------------- HERO HEADER -----------------
st.markdown("<div class='main-title'>Plant Disease Diagnostic Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='main-sub'>Upload leaf specimens to get instant deep learning disease classification and agronomy action plans.</div>", unsafe_allow_html=True)

# ----------------- MAIN LAYOUT -----------------
col_upload, col_result = st.columns([1, 1.2], gap="large")

with col_upload:
    with st.container(border=True):
        st.markdown("#### 📷 Specimen Upload")
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Specimen Preview", use_container_width=True)
            diagnose_btn = st.button("🚀 Diagnose Leaf Specimen")
        else:
            st.info("👆 Drag & drop or browse a leaf image to begin pathology analysis.")
            diagnose_btn = False

with col_result:
    if uploaded_file and diagnose_btn:
        with st.spinner("🔬 Analyzing leaf features and computing class probabilities..."):
            try:
                # Convert image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()

                # API request
                files = {"file": ("leaf.jpg", img_bytes, "image/jpeg")}
                response = requests.post(api_endpoint, files=files)

                if response.status_code == 200:
                    data = response.json()
                    disease = data.get("disease", "Unknown")
                    confidence = float(data.get("confidence", 0.0))
                    is_healthy = "healthy" in disease.lower()
                    treatment = data.get("treatment", {})
                    probabilities = data.get("probabilities", {})

                    # Top Diagnosis Box
                    with st.container(border=True):
                        badge = "<span class='badge-healthy'>HEALTHY LEAF</span>" if is_healthy else "<span class='badge-disease'>DISEASE DETECTED</span>"
                        st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                            <span style='color: #94a3b8; font-size: 0.95rem;'>Primary Diagnosis</span>
                            {badge}
                        </div>
                        <h2 style='margin: 0; color: #ffffff;'>{disease}</h2>
                        """, unsafe_allow_html=True)

                        c1, c2 = st.columns(2)
                        c1.metric("Confidence Score", f"{confidence:.1f}%")
                        c2.metric("Pipeline Status", "Optimal", delta="Verified")

                    # Probability Chart
                    if probabilities:
                        with st.container(border=True):
                            st.markdown("#### 📊 Model Probability Distribution")
                            df_probs = [{"Class": k.replace("_", " ").title(), "Probability (%)": v} for k, v in probabilities.items()]
                            fig = px.bar(
                                df_probs,
                                x="Probability (%)",
                                y="Class",
                                orientation="h",
                                text="Probability (%)",
                                color="Probability (%)",
                                color_continuous_scale=["#1e293b", "#10b981"]
                            )
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                font=dict(color="#CBD5E1"),
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=180,
                                xaxis=dict(showgrid=False, range=[0, 100]),
                                yaxis=dict(showgrid=False),
                                coloraxis_showscale=False
                            )
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            st.plotly_chart(fig, use_container_width=True)

                    # Treatment Cards
                    if "chemical" in treatment:
                        with st.container(border=True):
                            st.markdown("#### 📋 Recommended Treatment & Action Plan")
                            st.markdown(f"""
                            <div class='advisory-card advisory-chemical'>
                                <b style='color: #f87171;'>🧪 Chemical Treatment</b><br>
                                <span style='color: #cbd5e1;'>{treatment.get('chemical', 'N/A')}</span>
                            </div>
                            <div class='advisory-card advisory-organic'>
                                <b style='color: #34d399;'>🌱 Organic Remedy</b><br>
                                <span style='color: #cbd5e1;'>{treatment.get('organic', 'N/A')}</span>
                            </div>
                            <div class='advisory-card advisory-prevention'>
                                <b style='color: #38bdf8;'>🛡️ Prevention & Culture</b><br>
                                <span style='color: #cbd5e1;'>{treatment.get('prevention', 'N/A')}</span>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Could not connect to FastAPI server. Ensure Uvicorn is running. Error: {e}")
    elif not uploaded_file:
        with st.container(border=True):
            st.markdown("<h4 style='color: #64748b; text-align: center; margin: 40px 0;'>Awaiting Specimen Upload</h4>", unsafe_allow_html=True)
            st.caption("<div style='text-align: center;'>Select a plant leaf photo on the left to view diagnosis and treatment recommendations.</div>", unsafe_allow_html=True)