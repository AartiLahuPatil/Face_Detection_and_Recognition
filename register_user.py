import os
import cv2
from model.face_recognition_model import FaceRecognitionModel
from database.db import save_embedding

DATASET = "dataset"

model = FaceRecognitionModel()

for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)

    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name))
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = model.detect_faces(rgb)

        for f in faces:
            x, y, w, h = f["box"]
            x, y = max(0, x), max(0, y)
            face = rgb[y:y+h, x:x+w]

            if face.size == 0:
                continue

            emb = model.get_embedding(face)
            save_embedding(person, emb)

    print(f"Registered: {person}")

print("All users saved to database")
