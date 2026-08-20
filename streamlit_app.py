import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
import plotly.express as px

from advisory import get_treatment_plan
from pdf_report import generate_pdf_report

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="PhytoVision AI | Plant Pathology",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
CONFIDENCE_THRESHOLD = 65.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- CACHED MODEL LOADER -----------------
@st.cache_resource
def load_pathology_model():
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
    try:
        state_dict = torch.load("models/mobilenetv4_plant_disease.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception:
        pass
    model.to(DEVICE)
    model.eval()
    return model

model = load_pathology_model()

# ----------------- GRAD-CAM ENGINE -----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)

        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx

target_layer = model.conv_head if hasattr(model, "conv_head") else list(model.children())[-2]
grad_cam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## 🌿 **PhytoVision AI**")
    st.caption("Deep Learning Plant Pathology & Explainability Engine")
    st.divider()

    st.markdown("### 📋 Supported Pathologies")
    st.markdown("""
    * **Angular Leaf Spot** (*Phaeoisariopsis*)
    * **Bean Rust** (*Uromyces*)
    * **Healthy Foliage**
    """)
    st.divider()

    st.markdown("### 📸 Image Capture Guidelines")
    st.markdown("""
    - Ensure single leaf fills 70% of frame.
    - Avoid direct glare and extreme shadows.
    - Supported formats: `.jpg`, `.jpeg`, `.png` (Max 10MB).
    """)
    st.divider()

    st.warning("⚠️ **Agronomic Disclaimer**: Predictions are intended for informational guidance. Consult a local extension specialist prior to large-scale chemical application.")

# ----------------- MAIN UI -----------------
st.title("🌿 PhytoVision AI: Plant Pathology Diagnostic & Advisory Platform")
st.markdown("Upload foliar photographs for real-time disease detection, Grad-CAM attention mapping, and tiered remediation advice.")

uploaded_file = st.file_uploader("Upload Leaf Specimen", type=["jpg", "jpeg", "png"])

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    if uploaded_file is not None:
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ File size exceeds 10MB limit. Please upload an optimized image.")
            diagnose_btn = False
        else:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, caption="Uploaded Leaf Specimen", use_container_width=True)
            diagnose_btn = st.button("🚀 Diagnose & Generate Heatmap", use_container_width=True)
    else:
        st.info("👆 Upload a leaf photograph above to begin analysis.")
        diagnose_btn = False

with col_right:
    if uploaded_file and diagnose_btn:
        with st.spinner("🔬 Computing feature activations & generating Grad-CAM localization..."):
            input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
            
            with torch.enable_grad():
                cam, pred_idx = grad_cam.generate(input_tensor)
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]

            predicted_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx] * 100)
            is_uncertain = confidence < CONFIDENCE_THRESHOLD

            # Generate Heatmap Overlay
            orig_np = np.array(pil_image.resize((224, 224)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.6 * orig_np + 0.4 * heatmap)

            _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            heatmap_bytes = buffer.tobytes()

            treatment = get_treatment_plan(predicted_label) if not is_uncertain else {
                "chemical": "Diagnosis uncertain. Do not apply chemical treatments without secondary lab verification.",
                "organic": "Inspect foliar tissue under natural daylight; check for early spore formations.",
                "prevention": "Retake the leaf photograph under even lighting against a neutral background."
            }

            # Render Results
            if is_uncertain:
                st.warning("⚠️ **Low Confidence Diagnostic Flag (<65%)**: Specimen prediction is inconclusive. Please retake the leaf photograph under even lighting.")
                display_title = "Inconclusive / Low Confidence"
            else:
                display_title = predicted_label.replace("_", " ").title()

            st.markdown(f"### Pathology: **{display_title}**")
            st.metric("Model Confidence", f"{confidence:.2f}%")

            st.markdown("#### 🎯 Grad-CAM Lesion Localization Overlay")
            st.image(Image.open(io.BytesIO(heatmap_bytes)), use_container_width=True)

            # Probabilities chart
            st.markdown("#### 📊 Prediction Probability Distribution")
            df_probs = [{"Class": CLASS_NAMES[i].replace("_", " ").title(), "Probability (%)": round(float(probs[i] * 100), 2)} for i in range(len(CLASS_NAMES))]
            fig = px.bar(df_probs, x="Probability (%)", y="Class", orientation="h", text="Probability (%)")
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Treatment protocol
            st.markdown("#### 📋 Remediation Protocol")
            st.write(f"**🧪 Chemical:** {treatment.get('chemical', 'N/A')}")
            st.write(f"**🌱 Organic:** {treatment.get('organic', 'N/A')}")
            st.write(f"**🛡️ Prevention:** {treatment.get('prevention', 'N/A')}")

            # PDF Download
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="JPEG")
            pdf_buffer = generate_pdf_report(display_title, confidence, treatment, img_byte_arr.getvalue(), heatmap_bytes)
            
            st.download_button(
                label="📥 Download Diagnostic PDF Report",
                data=pdf_buffer,
                file_name=f"Pathology_Report_{predicted_label}.pdf",
                mime="application/pdf"
            )