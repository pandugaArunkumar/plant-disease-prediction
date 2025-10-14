# quick_check.py
import tensorflow as tf, numpy as np
from tensorflow.keras.models import load_model

IMG = 128
VAL_DIR = "dataset_split/val"
m = load_model("model/plant_disease_model.h5")
val = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR, image_size=(IMG, IMG), shuffle=False, batch_size=32)

probs = m.predict(val)
pred  = np.argmax(probs, axis=1)
true  = np.concatenate([y.numpy() for x,y in val])
acc = (pred == true).mean()
print("Val accuracy:", acc)
