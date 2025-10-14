import tensorflow as tf

# Paths to dataset
TRAIN_DIR = "dataset_split/train"

# Load dataset (only to get class names)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(128, 128),
    batch_size=32
)

# Extract class names
class_names = train_ds.class_names
print("✅ Classes found:", class_names)

# Save to file
with open("model/class_names.txt", "w") as f:
    for c in class_names:
        f.write(c + "\n")

print("✅ class_names.txt regenerated successfully!")
