import os
import shutil
import numpy as np
from PIL import Image

classes = ["Tomato___Early_blight", "Tomato___healthy", "Potato___Late_blight"]
splits = ["train", "val"]

# Clean old dataset folder if exists
if os.path.exists("dataset"):
    shutil.rmtree("dataset")

for split in splits:
    count = 15 if split == "train" else 5
    for cls in classes:
        folder_path = os.path.join("dataset", split, cls)
        os.makedirs(folder_path, exist_ok=True)
        
        for i in range(count):
            # Create synthetic 224x224 RGB image
            random_pixels = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(random_pixels)
            img.save(os.path.join(folder_path, f"sample_{i}.jpg"))

print("Sample dataset created successfully inside dataset/train and dataset/val!")