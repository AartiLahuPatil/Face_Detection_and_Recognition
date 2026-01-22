import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
from flask import Flask, render_template, request

# 🔹 IMPORT MODEL & DATABASE
from model.face_recognition_model import FaceRecognitionModel
from database.db import load_embeddings

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# ---------------- LOAD MODEL ONCE ----------------
model = FaceRecognitionModel()

# ---------------- LOAD DATABASE ONCE ----------------
db_names, db_embeddings = load_embeddings()

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or file.filename == "":
            result = "No file selected"
            return render_template("index.html", result=result)

        # Save uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(image_path)

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            result = "Invalid image file"
            return render_template("index.html", result=result)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        faces = model.detect_faces(rgb)

        if len(faces) == 0:
            result = "No face detected"
        else:
            for f in faces:
                x, y, w, h = f["box"]
                x, y = max(0, x), max(0, y)
                face = rgb[y:y+h, x:x+w]

                if face.size == 0:
                    continue

                # 🔹 EMBEDDING + PREDICTION
                embedding = model.get_embedding(face)
                result = model.predict(
                    embedding,
                    db_embeddings,
                    db_names
                )

        image_path = image_path.replace("static/", "")

    return render_template(
        "index.html",
        result=result,
        image_path=image_path
    )

# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
