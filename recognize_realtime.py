import cv2
from model.face_recognition_model import FaceRecognitionModel
from database.db import load_embeddings

model = FaceRecognitionModel()
db_names, db_embeddings = load_embeddings()

cap = cv2.VideoCapture(0)
print("Camera started | Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = model.detect_faces(rgb)

    for f in faces:
        x, y, w, h = f["box"]
        face = rgb[y:y+h, x:x+w]

        emb = model.get_embedding(face)
        name = model.predict(emb, db_embeddings, db_names)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Real-Time Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
