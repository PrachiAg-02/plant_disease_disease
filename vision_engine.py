import io
import base64
import cv2
import numpy as np
from PIL import Image

def process_specimen_severity(image_bytes: bytes):
    """
    Segments the leaf contour, detects lesion patches,
    calculates precise infected area percentage, and generates
    a visual diagnostic heatmap overlay.
    """
    # Decode image to OpenCV BGR
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img_bgr.shape
    
    # 1. Convert to HSV for adaptive agronomic color analysis
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # 2. Leaf canopy segmentation (isolate green/yellow biological matter from background)
    lower_plant = np.array([20, 35, 35])
    upper_plant = np.array([90, 255, 255])
    plant_mask = cv2.inRange(hsv, lower_plant, upper_plant)
    
    # Clean noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
    total_leaf_pixels = cv2.countNonZero(plant_mask)
    
    # Fallback if background or illumination is atypical
    if total_leaf_pixels == 0:
        total_leaf_pixels = h * w
        plant_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 3. Detect lesion / necrotic tissue (rust, spots, blight, chlorosis)
    lower_lesion_1 = np.array([10, 50, 50])   # Brown / Rust tones
    upper_lesion_1 = np.array([25, 255, 200])
    lower_lesion_2 = np.array([0, 0, 0])      # Necrotic dark / black spots
    upper_lesion_2 = np.array([180, 255, 70])
    
    mask_lesion_1 = cv2.inRange(hsv, lower_lesion_1, upper_lesion_1)
    mask_lesion_2 = cv2.inRange(hsv, lower_lesion_2, upper_lesion_2)
    lesion_mask_raw = cv2.bitwise_or(mask_lesion_1, mask_lesion_2)
    
    # Constrain lesions strictly within the detected leaf boundaries
    lesion_mask = cv2.bitwise_and(lesion_mask_raw, lesion_mask_raw, mask=plant_mask)
    lesion_pixels = cv2.countNonZero(lesion_mask)
    
    # Calculate Severity Index
    severity_pct = round(float((lesion_pixels / total_leaf_pixels) * 100), 2)
    severity_pct = min(severity_pct, 100.0)

    # 4. Generate Explainability Heatmap Overlay
    # Blur the lesion mask to simulate activation density
    heatmap_density = cv2.GaussianBlur(lesion_mask, (21, 21), 0)
    norm_heatmap = cv2.normalize(heatmap_density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply JET colormap for high-contrast thermal-style explainability
    heatmap_color = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    
    # Blend overlay with the original image
    diagnostic_overlay = cv2.addWeighted(img_bgr, 0.65, heatmap_color, 0.35, 0)
    
    # Encode overlay to PNG Base64 string for JSON transit
    _, buffer = cv2.imencode('.png', diagnostic_overlay)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return severity_pct, heatmap_base64