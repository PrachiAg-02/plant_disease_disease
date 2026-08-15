import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

def train_model(
    data_dir: str = "dataset",
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    save_path: str = "models/mobilenetv4_plant_disease.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Standard augmentations & normalizations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, "train")
    if not os.path.exists(train_path) or len(os.listdir(train_path)) == 0:
        raise FileNotFoundError(f"Training folder '{train_path}' not found.")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    classes = train_dataset.classes
    print(f"Training on {len(classes)} classes: {classes}")

    # Save class list for inference matching
    os.makedirs("models", exist_ok=True)
    with open("models/classes.json", "w") as f:
        json.dump(classes, f)

    # Initialize MobileNetV4 backbone
    model = timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, num_classes=len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar per epoch
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            # Live batch update
            loop.set_postfix(loss=loss.item(), acc=f"{(correct/total)*100:.1f}%")

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1} Complete | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel training finished! Checkpoint saved to: {save_path}")

if __name__ == "__main__":
    train_model()