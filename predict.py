import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def predict(image_path: str, model_path: str = "models/mobilenetv4_plant_disease.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load architecture
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Loading unweighted benchmark: {e}")
    model.to(device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()

    confidence = probs[pred_idx] * 100
    label = CLASS_NAMES[pred_idx]

    print(f"\n--- Diagnostic Prediction ---")
    print(f"Specimen: {image_path}")
    print(f"Pathology: {label.replace('_', ' ').title()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Class Probabilities: {dict(zip(CLASS_NAMES, [round(float(p)*100, 2) for p in probs]))}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhytoVision AI Specimen Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input leaf image")
    args = parser.parse_args()
    predict(args.image)