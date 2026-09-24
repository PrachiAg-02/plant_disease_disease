import os
import io
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, Depends, HTTPException, Security, UploadFile, File
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Local module imports (Make sure vision_engine.py and database.py exist in the same folder)
from vision_engine import process_specimen_severity
from database import SessionLocal, Organization, Farm, DiagnosticEvent

# Load secure configuration from .env file
load_dotenv()

# Initialize Enterprise API
app = FastAPI(
    title="PhytoVision AI Enterprise API",
    version="1.2.0",
    description="Enterprise Agritech Diagnostic & Severity Intelligence Gateway with Data Persistence"
)

# ----------------- SECURITY: API KEY VALIDATION -----------------
API_KEY_HEADER = APIKeyHeader(name="x-api-key", auto_error=True)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    expected_key = os.getenv("PHYTOVISION_API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized Enterprise Client")
    return api_key


# ----------------- DATABASE INJECTION -----------------
def get_db():
    """Yields a database session and closes it after the request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_tenant(db: Session, api_key: str):
    """Auto-provisions a mock B2B tenant for demonstration purposes"""
    org = db.query(Organization).filter(Organization.api_key == api_key).first()
    if not org:
        org = Organization(name="Enterprise Client", api_key=api_key)
        farm = Farm(name="Sector 7 Autonomous Field", organization=org)
        db.add(org)
        db.add(farm)
        db.commit()
        db.refresh(org)
    return db.query(Farm).filter(Farm.organization_id == org.id).first()


# ----------------- INFERENCE ENGINE SETUP -----------------
MODEL_PATH = "models/mobilenetv4_combined.onnx"
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"ONNX Engine Online. Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Warning: ONNX model not found at {MODEL_PATH}. Ensure the path is correct. Error: {e}")

# Define actual class mappings based on your model
CLASS_NAMES = [
    'Apple__Healthy', 
    'Apple__Rotten', 
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Banana__Healthy', 
    'Banana__Rotten', 
    'Orange__Healthy', 
    'Orange__Rotten', 
    'Tomato__Healthy', 
    'Tomato__Rotten', 
    'Tomato___Bacterial_spot', 
    'Tomato___Early_blight', 
    'Tomato___healthy'

]

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Prepares image for MobileNetV4 ONNX inference"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    
    # Normalize to [0, 1] then apply standard ImageNet mean/std
    img_array = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # ONNX expects NCHW format: (Batch, Channels, Height, Width)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------------- API ENDPOINTS -----------------
@app.post("/analyze")
async def analyze_crop(
    file: UploadFile = File(...), 
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    try:
        # Read image
        image_bytes = await file.read()
        
        # 1. Run MobileNetV4 Pathology Classification
        input_tensor = preprocess_image(image_bytes)
        logits = session.run(None, {input_name: input_tensor})[0]
        
        # Apply softmax to get confidence scores
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Get top prediction
        class_idx = int(np.argmax(probabilities[0]))
        confidence_score = float(probabilities[0][class_idx] * 100)
        top_disease = CLASS_NAMES[class_idx]
        
        # 2. Handle Healthy vs Diseased logic and Visuals
        if top_disease == "Healthy":
            severity_pct = 0.0
            sev_class = "None"
            risk = "Low"
            heatmap_b64 = None
            action = "Tissue integrity intact. Maintain scheduled preventative irrigation and nutrient management."
        else:
            # Real Pathology Severity & Heatmap Extraction
            severity_pct, heatmap_b64 = process_specimen_severity(image_bytes)
            
            # Define severity classification
            if severity_pct < 10.0:
                sev_class, risk = "Mild", "Observation Advisory"
            elif severity_pct < 25.0:
                sev_class, risk = "Moderate", "Action Required (Targeted Treatment)"
            else:
                sev_class, risk = "Severe", "Critical Outbreak Alert"
                
            action = f"Confirmed {top_disease}. Initiate targeted fungicide application based on regional canopy density."

        # 3. Database Persistence (Save the record)
        farm = get_or_create_tenant(db, api_key)
        db_record = DiagnosticEvent(
            farm_id=farm.id,
            disease_predicted=top_disease,
            confidence_score=confidence_score,
            severity_percentage=severity_pct,
            risk_level=risk
        )
        db.add(db_record)
        db.commit()

        # 4. Return Payload
        return {
            "diagnosis": top_disease,
            "confidence": confidence_score,
            "severity": severity_pct,
            "severity_class": sev_class,
            "risk_level": risk,
            "heatmap_base64": heatmap_b64,
            "action": action
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic Pipeline Error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker & system monitoring"""
    return {"status": "online", "engine": "ONNX Runtime", "persistence": "SQLite Database", "pipeline": "v1.2"}

# Add this to the bottom of main.py
@app.get("/analytics")
async def get_tenant_analytics(api_key: str = Depends(verify_api_key), db: Session = Depends(get_db)):
    """Retrieves all historical diagnostic events for the authenticated tenant."""
    try:
        farm = get_or_create_tenant(db, api_key)
        
        # Query all events for this tenant, ordered by newest first
        events = db.query(DiagnosticEvent).filter(
            DiagnosticEvent.farm_id == farm.id
        ).order_by(DiagnosticEvent.timestamp.desc()).all()
        
        # Format the data for the frontend DataFrame
        data = []
        for event in events:
            data.append({
                "timestamp": event.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "disease": event.disease_predicted,
                "confidence": round(event.confidence_score, 2),
                "severity": round(event.severity_percentage, 2),
                "risk": event.risk_level
            })
            
        return {"status": "success", "total_records": len(data), "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics Engine Error: {str(e)}")

    # Add this to the bottom of main.py
import shutil
from fastapi import Form

# Ensure the MLOps storage directory exists
os.makedirs("flagged_samples", exist_ok=True)

@app.post("/flag")
async def flag_sample(
    file: UploadFile = File(...),
    actual_diagnosis: str = Form(...),
    api_key: str = Depends(verify_api_key)
):
    """Saves hard samples to an object queue for continuous ML retraining."""
    try:
        # In a real cloud environment, this would push to an AWS S3 or GCP Cloud Storage bucket
        safe_filename = file.filename.replace(" ", "_")
        file_location = f"flagged_samples/flagged_{actual_diagnosis}_{safe_filename}"
        
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        return {"status": "success", "message": "Sample successfully queued for MLOps review."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLOps Queue Error: {str(e)}")