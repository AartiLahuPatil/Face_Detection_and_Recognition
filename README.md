# Face Recognition System using MTCNN & FaceNet

## Project Overview

This project implements a Face Recognition System using MTCNN for face
detection and FaceNet for face embeddings. It supports face
registration, recognition, and accuracy evaluation.

## Technologies Used

-   Python
-   OpenCV
-   TensorFlow / Keras
-   MTCNN
-   keras-facenet
-   SQLite
-   NumPy
-   Scikit-learn

## Project Structure

Face/ 
├── register_user.py 
├── recognize_face.py
├── create_db.py.py 
├── README.md
├── dataset/ 
│ └── Manu 
    └── img1.jpeg
    └── img2.jpeg
  └── Rutuja 
    └── img1.jpeg
    └── img2.jpeg
  └── Samruddhi 
    └── img1.jpeg
    └── img2.jpeg
├── database/ 
│ └── face_recognition.db
├── test_images/ 
  └── test1.jpeg
  └── test2.jpeg
  └── test3.jpeg
  └── test4.jpeg
     
## Face Recognition System –

Face Recognition is a biometric technology used to identify or verify a person using facial features from images or videos.
This project uses deep learning techniques to detect and recognize human faces accurately.

## Face Detection -

Face detection is the first step where the system locates human faces in an image.
In this project, MTCNN (Multi-Task Cascaded Convolutional Neural Network) is used.
It detects faces by identifying facial landmarks like eyes, nose, and mouth.

## Face Embedding -

After detection, the face image is converted into a numerical representation called an embedding.
FaceNet is used to generate a 128-dimensional vector that uniquely represents a person’s face.

## Database Storage -

The generated face embeddings are stored in an SQLite database along with the person’s name.
These embeddings act as reference data for future recognition.

## Face Recognition -

For recognition, embeddings from a test image are compared with stored embeddings using
Cosine Similarity.
If similarity score is above a threshold, the face is recognized; otherwise, it is labeled Unknown.

### Conclusion:-

This system provides an efficient and accurate method for face recognition using deep learning.
It can be used in applications such as attendance systems, security, and access control.