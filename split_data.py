import os
import shutil
import random

# Paths
DATASET_DIR = "C:\plant_disease_identification\dataset\plantvillage\color"

OUTPUT_DIR = "dataset_split"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

# Train/Val ratio
train_ratio = 0.8

# Create output folders
for folder in [TRAIN_DIR, VAL_DIR]:
    os.makedirs(folder, exist_ok=True)

# Loop over each class
for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue  # Skip files or nested folders

    # Make class subfolders
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    # List all images
    images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
    random.shuffle(images)

    # Split into train/val
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Copy train images
    for img in train_images:
        shutil.copy(os.path.join(class_dir, img),
                    os.path.join(TRAIN_DIR, class_name, img))

    # Copy val images
    for img in val_images:
        shutil.copy(os.path.join(class_dir, img),
                    os.path.join(VAL_DIR, class_name, img))

print("✅ Dataset successfully split into train and val!")
