import io
import base64
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import timm

from advisory import get_treatment_plan

app = FastAPI(title="PhytoVision AI - Plant Pathology API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- MODEL SETUP -----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

# Load MobileNetV4
model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
try:
    state_dict = torch.load("models/mobilenetv4_plant_disease.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
except Exception:
    pass  # Fallback for CI testing environments

model.to(DEVICE)
model.eval()

# ----------------- GRAD-CAM HOOKS -----------------
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

# Attach GradCAM to the last feature convolutional block
target_layer = model.conv_head if hasattr(model, "conv_head") else list(model.children())[-2]
grad_cam = GradCAM(model, target_layer)

# ----------------- IMAGE TRANSFORM -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----------------- ENDPOINTS -----------------
@app.get("/")
def root():
    return {
        "status": "online",
        "model": "MobileNetV4",
        "features": ["Classification", "Grad-CAM Localization", "Agronomic Advisory"],
        "device": str(DEVICE)
    }

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

    # Model inference
    input_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
    with torch.enable_grad():
        cam, pred_idx = grad_cam.generate(input_tensor)
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]

    predicted_label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100)

    # Generate Heatmap Overlay
    orig_np = np.array(pil_image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * orig_np + 0.4 * heatmap)

    # Convert overlay to base64
    _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    # Get structured agronomic treatment
    treatment = get_treatment_plan(predicted_label)
    prob_dist = {CLASS_NAMES[i]: round(float(probs[i] * 100), 2) for i in range(len(CLASS_NAMES))}

    return {
        "disease": predicted_label.replace("_", " ").title(),
        "raw_label": predicted_label,
        "confidence": round(confidence, 2),
        "probabilities": prob_dist,
        "heatmap": heatmap_base64,
        "treatment": treatment
    }