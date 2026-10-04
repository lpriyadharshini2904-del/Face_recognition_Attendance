import cv2
import os

dataset_path = "dataset"
model_path = "model"

os.makedirs(model_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

faces = []
ids = []

for file_name in os.listdir(dataset_path):

    if file_name.endswith(".jpg"):

        image_path = os.path.join(dataset_path, file_name)

        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        face_id = int(file_name.split(".")[1])

        faces.append(gray_image)
        ids.append(face_id)

recognizer.train(faces, __import__("numpy").array(ids))

recognizer.write("model/trainer.yml")

print("Training completed successfully.")
print("Model saved as: model/trainer.yml")