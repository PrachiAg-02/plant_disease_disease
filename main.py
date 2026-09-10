from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import time
import cv2
import base64

app = FastAPI(title="PhytoVision AI Enterprise API", version="2.0.0")

# 1. Load ONNX Model
MODEL_PATH = "models/mobilenetv4_plant_disease.onnx"
print(f"Loading ONNX Engine from {MODEL_PATH}...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("Engine Online.")

CLASS_NAMES = ["Angular Leaf Spot", "Bean Rust", "Healthy"]

VALID_API_KEYS = {
    "key_drone_corp_2026": "Pay-Per-Call Volume Tier",
    "key_insurance_mnc_001": "Enterprise Unlimited Tier"
}

def verify_api_key(x_api_key: str = Header(..., description="Client API Key required for billing")):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Unauthorized. Invalid or expired B2B API Key.")
    return x_api_key

# 2. Vision Preprocessing Pipeline
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    img_array = np.transpose(img_array, (2, 0, 1))
    return np.expand_dims(img_array, axis=0)

# 3. Agronomic Severity & Heatmap Generator (Computer Vision)
def analyze_severity_and_heatmap(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask for plant leaves
    lower_leaf = np.array([10, 20, 20])
    upper_leaf = np.array([100, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    
    # Mask for diseased lesions
    lower_disease = np.array([10, 50, 20])
    upper_disease = np.array([30, 255, 200])
    disease_mask = cv2.inRange(hsv, lower_disease, upper_disease)
    
    leaf_pixels = cv2.countNonZero(leaf_mask)
    disease_pixels = cv2.countNonZero(disease_mask)
    
    if leaf_pixels == 0:
        severity_pct = 0.0
    else:
        severity_pct = round((disease_pixels / leaf_pixels) * 100, 1)
        
    heatmap = cv2.applyColorMap(disease_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    _, buffer = cv2.imencode('.jpg', overlay)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return severity_pct, heatmap_b64

# 4. Core Inference Endpoint
@app.post("/v2/diagnostics/analyze")
async def run_diagnostic_inference(
    image: UploadFile = File(...),
    temperature_c: float = Form(...),
    humidity_pct: float = Form(...),
    api_key: str = Depends(verify_api_key)
):
    start_time = time.time()
    
    image_bytes = await image.read()
    input_tensor = preprocess_image(image_bytes)
    
    # ONNX Model Inference
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probabilities = exp_logits / exp_logits.sum()
    
    top_index = int(np.argmax(probabilities))
    top_disease = CLASS_NAMES[top_index]
    confidence = round(float(probabilities[top_index]) * 100, 1)
    
    # Computer Vision Severity Calculation & Heatmap
    severity_pct, heatmap_b64 = analyze_severity_and_heatmap(image_bytes)
    
    if top_disease == "Healthy Baseline":
        severity_pct = 0.0
        severity_tier = "Optimal (No Action Needed)"
    else:
        if severity_pct < 10: 
            severity_tier = "Mild (Stage I)"
        elif severity_pct < 30: 
            severity_tier = "Moderate (Stage II)"
        else: 
            severity_tier = "Severe (Stage III)"
        
    inference_time = round((time.time() - start_time) * 1000, 2)
    
    response_payload = {
        "status": "success",
        "client_tier": VALID_API_KEYS[api_key],
        "inference_time_ms": inference_time,
        "multimodal_context": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct
        },
        "results": {
            "primary_diagnosis": top_disease,
            "confidence_score": confidence,
            "severity_index": {
                "affected_leaf_pct": severity_pct,
                "tier": severity_tier
            },
            "decision_gate": "ACTIONABLE" if confidence >= 65 else "INCONCLUSIVE"
        },
        "visual_heatmap_b64": heatmap_b64
    }
    
    return JSONResponse(content=response_payload)