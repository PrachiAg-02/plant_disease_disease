import io
import time
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def create_dummy_image(format="JPEG", mode="RGB"):
    """Helper to generate in-memory mock leaf images."""
    img = Image.new(mode, (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=format)
    buf.seek(0)
    return buf

# REQ-AI-001: Health check availability
def test_health_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "MobileNetV4" in data["model"]

# REQ-AI-002: Successful diagnosis and latency under 1 second
def test_diagnose_endpoint_latency_and_schema():
    img_buf = create_dummy_image()
    files = {"file": ("test_leaf.jpg", img_buf, "image/jpeg")}
    
    start_time = time.time()
    response = client.post("/diagnose", files=files)
    latency = time.time() - start_time
    
    assert response.status_code == 200
    assert latency < 1.0  # Latency requirement
    
    data = response.json()
    assert "disease" in data
    assert "confidence" in data
    assert "treatment" in data
    assert "probabilities" in data

# REQ-AI-003: Non-image rejection test
def test_invalid_file_rejection():
    fake_file = io.BytesIO(b"not an image file")
    files = {"file": ("document.txt", fake_file, "text/plain")}
    response = client.post("/diagnose", files=files)
    assert response.status_code == 400