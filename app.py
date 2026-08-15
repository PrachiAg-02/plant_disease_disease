import os
import io
import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from advisory import get_treatment_recommendation

app = FastAPI(
    title="Plant Disease Diagnostic & Advisory Engine",
    version="2.0.0"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/mobilenetv4_plant_disease.pth"
CLASSES_PATH = "models/classes.json"

if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

# Load MobileNetV4 Model
model = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=False, num_classes=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"Loaded weights from {MODEL_PATH}")
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def health():
    return {"status": "online", "model": "MobileNetV4", "classes": CLASS_NAMES}

@app.post("/diagnose")
async def diagnose(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, class_idx = torch.max(probabilities, dim=0)

    predicted_label = CLASS_NAMES[class_idx.item()]
    
    # Structure all class probabilities for charts
    class_probs = {
        CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
        for i in range(len(CLASS_NAMES))
    }

    # Fetch treatment advisory
    result = get_treatment_recommendation(predicted_label, float(confidence.item()))
    result["probabilities"] = class_probs
    return JSONResponse(content=result)