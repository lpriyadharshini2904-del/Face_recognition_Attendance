import cv2
import csv
from datetime import datetime

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

# Attendance file
attendance_file = "attendance.csv"

# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("Attendance System Started...")
print("Look at the camera.")
print("Press Q to stop.")

marked_ids = set()

# Check today's attendance already marked
today = datetime.now().strftime("%Y-%m-%d")

try:
    with open(attendance_file, "r") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["Date"] == today:
                marked_ids.add(int(row["FaceID"]))

except FileNotFoundError:
    pass

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

            name = student["Name"]
            roll_no = student["RollNo"]
            department = student["Department"]
            year = student["Year"]

            # Mark attendance once
            if face_id not in marked_ids:

                now = datetime.now()

                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        date,
                        time,
                        face_id,
                        name,
                        roll_no,
                        department,
                        year,
                        "Present"
                    ])

                marked_ids.add(face_id)

                print(
                    f"Attendance marked: "
                    f"{name} - {roll_no}"
                )

            # Display full student details
            cv2.putText(
                frame,
                f"Name: {name}",
                (x, y - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Roll No: {roll_no}",
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Dept: {department}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Year: {year}",
                (x, y + h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
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