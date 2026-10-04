import cv2
import os

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# Student details
face_id = input("Enter Face ID: ")
name = input("Enter Student Name: ")
roll_no = input("Enter Roll No: ")
department = input("Enter Department: ")
year = input("Enter Year: ")

# Face detector
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Face detector file not found")
    exit()

# Open camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera could not be opened")
    exit()

print("\nCamera started.")
print("Look at the camera.")
print("30 samples will be captured.")
print("Press Q to stop.")

sample_count = 0

while sample_count < 30:

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

        sample_count += 1

        face_image = gray[y:y+h, x:x+w]

        file_name = f"dataset/User.{face_id}.{sample_count}.jpg"

        cv2.imwrite(file_name, face_image)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        if sample_count >= 30:
            break

    cv2.imshow("Student Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

if sample_count == 30:
    print("\nFace samples collected successfully.")
else:
    print(f"\nRegistration stopped. {sample_count} samples collected.")

print("Student:", name)
print("Roll No:", roll_no)
print("Department:", department)
print("Year:", year)