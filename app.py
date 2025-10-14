from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("model/plant_disease_model.keras")

# Load class names
with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# Folder for uploads
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# Home Page (main landing)
# -------------------------
@app.route("/")
def home():
    return render_template("home.html")  # lowercase filename (ensure template name matches)

# -------------------------
# Predict Page (GET + POST)
# -------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        # Show the upload page (index.html)
        return render_template("index.html")

    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load and preprocess image (128x128)
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = round(100 * np.max(predictions), 2)

    return render_template(
        "result.html",
        filename=file.filename,
        predicted_class=predicted_class,
        confidence=confidence
    )

# -------------------------
# Optional: To display image directly
# -------------------------
@app.route("/display/<filename>")
def display_image(filename):
    return f"<img src='/static/uploads/{filename}' width='400'>"

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
