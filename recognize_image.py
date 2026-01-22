import cv2
from tkinter import Tk, filedialog
from model.face_recognition_model import FaceRecognitionModel
from database.db import load_embeddings

model = FaceRecognitionModel()
db_names, db_embeddings = load_embeddings()

Tk().withdraw()
img_path = filedialog.askopenfilename(
    title="Select Image",
    filetypes=[("Images", "*.jpg *.jpeg *.png")]
)

img = cv2.imread(img_path)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

faces = model.detect_faces(rgb)

for f in faces:
    x, y, w, h = f["box"]
    face = rgb[y:y+h, x:x+w]

    emb = model.get_embedding(face)
    name = model.predict(emb, db_embeddings, db_names)

    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(img, name, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

cv2.imshow("Face Recognition", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
