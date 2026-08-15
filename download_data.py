import os
import shutil
from datasets import load_dataset

print("Downloading real plant leaf disease dataset...")

# Use the full namespace for the dataset
ds = load_dataset("AI-Lab-Makerere/beans")

labels_map = ds["train"].features["labels"].names
print(f"Target Disease Classes: {labels_map}")

# Reset local dataset folder
if os.path.exists("dataset"):
    shutil.rmtree("dataset")

os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/val", exist_ok=True)

# Save real train images
print("Saving training images...")
for i, item in enumerate(ds["train"]):
    label_name = labels_map[item["labels"]]
    folder = os.path.join("dataset", "train", label_name)
    os.makedirs(folder, exist_ok=True)
    item["image"].save(os.path.join(folder, f"leaf_train_{i}.jpg"))

# Save real validation images
print("Saving validation images...")
for i, item in enumerate(ds["validation"]):
    label_name = labels_map[item["labels"]]
    folder = os.path.join("dataset", "val", label_name)
    os.makedirs(folder, exist_ok=True)
    item["image"].save(os.path.join(folder, f"leaf_val_{i}.jpg"))

print("Done! Real leaf images saved to dataset/train and dataset/val.")