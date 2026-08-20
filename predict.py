import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import timm

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def predict(image_path: str, model_path: str = "models/mobilenetv4_plant_disease.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model architecture
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # Validate file existence or fallback to synthetic sample
    if not os.path.exists(image_path):
        print(f"⚠️ Image '{image_path}' not found. Generating a synthetic test specimen...")
        image = Image.fromarray(np.random.randint(40, 180, (224, 224, 3), dtype=np.uint8))
    else:
        image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    confidence = probs[pred_idx] * 100
    label = CLASS_NAMES[pred_idx]

    print(f"\n==========================================")
    print(f"🌿 PHYTOMOBILE INFERENCE DIAGNOSIS")
    print(f"==========================================")
    print(f"Specimen Path    : {image_path}")
    print(f"Predicted Disease: {label.replace('_', ' ').title()}")
    print(f"Model Confidence : {confidence:.2f}%")
    print(f"Softmax Breakdown: {dict(zip(CLASS_NAMES, [round(float(p)*100, 2) for p in probs]))}")
    print(f"==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhytoVision AI Specimen Inference")
    parser.add_argument("--image", type=str, default="sample_leaf.jpg", help="Path to input leaf image")
    args = parser.parse_args()
    predict(args.image)