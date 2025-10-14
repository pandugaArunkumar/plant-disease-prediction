import tensorflow as tf
import numpy as np
import os

MODEL_PATH = "model/plant_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.txt"
TRAIN_DIR = "dataset_split/train"  # <-- path you used for training

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded:", MODEL_PATH)

# Step 1: TensorFlow order (how dataset_from_directory sorted folders)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)
tf_class_names = train_ds.class_names

print("\n📂 TensorFlow detected class order:")
for i, name in enumerate(tf_class_names):
    print(f"{i:2d}: {name}")

# Step 2: Your saved label file
print("\n📄 Your saved class_names.txt order:")
with open(CLASS_NAMES_PATH) as f:
    saved = [l.strip() for l in f if l.strip()]
for i, name in enumerate(saved):
    print(f"{i:2d}: {name}")

# Step 3: Check if they match
if tf_class_names == saved:
    print("\n✅ MATCH: class_names.txt is correct.")
else:
    print("\n❌ MISMATCH: class_names.txt order does not match training dataset order!")
    print("You can fix this by re-creating class_names.txt in the correct order.")