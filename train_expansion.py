import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os

# --- 1. CONFIGURATION ---
DATA_DIR = "combined_dataset"
BATCH_SIZE = 32
EPOCHS = 5
MODEL_NAME = 'mobilenetv4_conv_small.e2400_r224_in1k'

def main():
    # --- 2. DATA PREPARATION ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading dataset...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    # num_workers=0 ensures seamless execution on Windows
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    num_classes = len(dataset.classes)
    print(f"Found {num_classes} classes: {dataset.classes}")

    # --- 3. MODEL SETUP ---
    print("Initializing MobileNetV4 backbone...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. RAPID TRAINING LOOP ---
    print(f"Starting training on {device} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")

    # --- 5. EXPORT TO ONNX ---
    print("Training complete. Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    onnx_path = "models/mobilenetv4_combined.onnx"

    os.makedirs("models", exist_ok=True)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Success! New model saved to {onnx_path}")

if __name__ == '__main__':
    main()