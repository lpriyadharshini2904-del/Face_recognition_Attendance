import cv2
import csv

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model/trainer.yml")

# Load face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Face detector file not found")
    exit()

# Load student details
students = {}

with open("students.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        students[int(row["FaceID"])] = row

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Face Recognition Started...")
print("Look at the camera.")
print("Press Q to stop.")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read camera frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face_id, confidence = recognizer.predict(face)

        if confidence < 70 and face_id in students:

            student = students[face_id]

            text = (
                f"{student['Name']} | "
                f"{student['RollNo']}"
            )

        else:
            text = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()