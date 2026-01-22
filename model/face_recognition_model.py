import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity


class FaceRecognitionModel:

    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        print("Loading face models...")
        detector = MTCNN()
        embedder = FaceNet()
        print("Models loaded successfully")

        return {
            "detector": detector,
            "embedder": embedder
        }

    def detect_faces(self, rgb_img):
        return self.model["detector"].detect_faces(rgb_img)

    def get_embedding(self, face):
        face = cv2.resize(face, (160, 160))
        face = face.astype("float32")
        return self.model["embedder"].embeddings([face])[0]

    def predict(self, emb, db_embeddings, db_names, threshold=0.8):
        if len(db_embeddings) == 0:
            return "Unknown"

        scores = cosine_similarity([emb], db_embeddings)[0]
        idx = np.argmax(scores)

        if scores[idx] >= threshold:
            return f"{db_names[idx]} ({scores[idx]:.2f})"
        return "Unknown"
