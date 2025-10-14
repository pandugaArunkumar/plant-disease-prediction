# check_accuracy_fixed.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

val_dir = "dataset_split/val"

datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model("model/plant_disease_model.keras")

# Predict
preds = model.predict(val_data)
pred_labels = np.argmax(preds, axis=1)
true_labels = val_data.classes

class_names = list(val_data.class_indices.keys())

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(true_labels, pred_labels, target_names=class_names))
