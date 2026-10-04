from flask import Flask, render_template, Response, jsonify, request, redirect, session
import cv2
import csv
import os
from datetime import datetime


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder="frontend",
    static_folder="frontend",
    static_url_path="/static"
)

# Required for Flask session
app.secret_key = "face_attendance_secret_2026"


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

STUDENTS_FILE = os.path.join(BASE_DIR, "students.csv")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")

CASCADE_FILE = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

TRAINER_FILE = os.path.join(
    MODEL_DIR,
    "trainer.yml"
)


# Create folders if needed
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("WARNING: Camera could not be opened.")


# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(CASCADE_FILE)

if face_detector.empty():
    print("ERROR: haarcascade_frontalface_default.xml not found.")


# ============================================================
# FACE RECOGNIZER
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists(TRAINER_FILE):

    try:
        recognizer.read(TRAINER_FILE)
        print("Face recognition model loaded.")
    except Exception as e:
        print("ERROR loading trainer.yml:", e)

else:
    print("WARNING: model/trainer.yml not found.")


# ============================================================
# STUDENT DATA
# ============================================================

students = {}


def load_students():

    global students

    students = {}

    if not os.path.exists(STUDENTS_FILE):
        return

    try:

        with open(
            STUDENTS_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                try:

                    face_id = int(row["FaceID"])

                    students[face_id] = {
                        "FaceID": str(face_id),
                        "Name": row.get("Name", ""),
                        "RollNo": row.get("RollNo", ""),
                        "Department": row.get("Department", ""),
                        "Year": row.get("Year", "")
                    }

                except (ValueError, KeyError):
                    continue

    except Exception as e:
        print("Error loading students.csv:", e)


load_students()


# ============================================================
# ATTENDANCE FILE
# ============================================================

def create_attendance_file():

    if not os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Date",
                "Time",
                "FaceID",
                "Name",
                "RollNo",
                "Department",
                "Year",
                "Status"
            ])


create_attendance_file()


# ============================================================
# CURRENT RECOGNIZED STUDENT
# ============================================================

current_face_id = None
current_status = "Waiting"


# ============================================================
# CHECK TODAY ATTENDANCE
# ============================================================

def attendance_already_marked(face_id):

    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(ATTENDANCE_FILE):
        return False

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if (
                    row.get("Date") == today
                    and row.get("FaceID") == str(face_id)
                ):
                    return True

    except Exception as e:

        print("Attendance checking error:", e)

    return False


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(face_id):

    global current_status

    if face_id not in students:
        return False

    # Already marked today
    if attendance_already_marked(face_id):

        current_status = "Attendance Marked"

        return False

    student = students[face_id]

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    try:

        with open(
            ATTENDANCE_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                date,
                time,
                face_id,
                student["Name"],
                student["RollNo"],
                student["Department"],
                student["Year"],
                "Present"
            ])

        current_status = "Attendance Marked"

        print(
            "Attendance marked:",
            student["Name"],
            "(" + student["RollNo"] + ")"
        )

        return True

    except Exception as e:

        print("Attendance marking error:", e)

        current_status = "Error"

        return False


# ============================================================
# CAMERA FRAME GENERATOR
# ============================================================

def generate_frames():

    global current_face_id
    global current_status

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame could not be read.")
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(100, 100)
        )

        # No face
        if len(faces) == 0:

            current_face_id = None
            current_status = "Waiting"

        # Face detected
        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            try:

                face_id, confidence = recognizer.predict(face)

            except Exception:

                face_id = -1
                confidence = 999

            # LBPH:
            # lower confidence = better match

            if (
                confidence < 70
                and face_id in students
            ):

                current_face_id = face_id

                student = students[face_id]

                name = student["Name"]

                # Mark attendance
                mark_attendance(face_id)

                display_text = name

                # Green box
                box_color = (0, 255, 0)

            else:

                current_face_id = None
                current_status = "Unknown"

                display_text = "Unknown"

                # Red box
                box_color = (0, 0, 255)

            # Draw rectangle

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                box_color,
                2
            )

            # Display name

            cv2.putText(
                frame,
                display_text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2
            )

        # Convert to JPEG

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template("index.html")


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        # Admin credentials
        if username == "admin" and password == "admin123":

            session["admin_logged_in"] = True

            return redirect("/students")

        return render_template(
            "login.html",
            error="Invalid username or password"
        )

    return render_template("login.html")


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect("/login")


# ============================================================
# ADMIN CHECK
# ============================================================

def admin_required():

    return session.get("admin_logged_in", False)


# ============================================================
# STUDENT MANAGEMENT PAGE
# ============================================================

@app.route("/students")
def students_page():

    if not admin_required():

        return redirect("/login")

    load_students()

    student_list = list(students.values())

    return render_template(
        "students.html",
        students=student_list
    )


# ============================================================
# ADD STUDENT PAGE
# ============================================================

@app.route("/add-student", methods=["GET", "POST"])
def add_student():

    if not admin_required():

        return redirect("/login")

    if request.method == "POST":

        face_id = request.form.get("face_id", "").strip()
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        department = request.form.get("department", "").strip()
        year = request.form.get("year", "").strip()

        if not face_id or not name or not roll_no:

            return render_template(
                "add_student.html",
                error="Face ID, Name and Roll Number are required."
            )

        try:

            face_id_int = int(face_id)

        except ValueError:

            return render_template(
                "add_student.html",
                error="Face ID must be a number."
            )

        load_students()

        # Check duplicate Face ID

        if face_id_int in students:

            return render_template(
                "add_student.html",
                error="This Face ID already exists."
            )

        # Check duplicate Roll Number

        for student in students.values():

            if student["RollNo"] == roll_no:

                return render_template(
                    "add_student.html",
                    error="This Roll Number already exists."
                )

        # Add student to CSV

        file_exists = os.path.exists(STUDENTS_FILE)

        try:

            with open(
                STUDENTS_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.writer(file)

                if not file_exists:

                    writer.writerow([
                        "FaceID",
                        "Name",
                        "RollNo",
                        "Department",
                        "Year"
                    ])

                writer.writerow([
                    face_id_int,
                    name,
                    roll_no,
                    department,
                    year
                ])

            load_students()

            return redirect("/students")

        except Exception as e:

            return render_template(
                "add_student.html",
                error="Could not add student: " + str(e)
            )

    return render_template("add_student.html")


# ============================================================
# DELETE STUDENT
# ============================================================

@app.route("/delete-student/<int:face_id>")
def delete_student(face_id):

    if not admin_required():

        return redirect("/login")

    load_students()

    if face_id not in students:

        return redirect("/students")

    # Remove student from CSV

    remaining_students = []

    try:

        with open(
            STUDENTS_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                try:

                    if int(row["FaceID"]) != face_id:
                        remaining_students.append(row)

                except (ValueError, KeyError):

                    continue

        with open(
            STUDENTS_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "FaceID",
                "Name",
                "RollNo",
                "Department",
                "Year"
            ])

            for row in remaining_students:

                writer.writerow([
                    row["FaceID"],
                    row["Name"],
                    row["RollNo"],
                    row["Department"],
                    row["Year"]
                ])

        load_students()

    except Exception as e:

        print("Delete student error:", e)

    return redirect("/students")


# ============================================================
# LIVE VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# STUDENT DATA FOR WEBPAGE
# ============================================================

@app.route("/student-data")
def student_data():

    if (
        current_face_id is not None
        and current_face_id in students
    ):

        student = students[current_face_id]

        return jsonify({

            "name": student["Name"],

            "rollno": student["RollNo"],

            "department": student["Department"],

            "year": student["Year"],

            "status": "Attendance Marked"

        })

    return jsonify({

        "name": "---",

        "rollno": "---",

        "department": "---",

        "year": "---",

        "status": "Waiting"

    })


# ============================================================
# ATTENDANCE TABLE DATA
# ============================================================

@app.route("/attendance-data")
def attendance_data():

    records = []

    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(ATTENDANCE_FILE):

        return jsonify(records)

    try:

        with open(
            ATTENDANCE_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                if row.get("Date") == today:

                    records.append({

                        "time": row.get("Time", ""),

                        "rollno": row.get("RollNo", ""),

                        "name": row.get("Name", ""),

                        "department": row.get("Department", ""),

                        "year": row.get("Year", ""),

                        "status": row.get("Status", "")

                    })

    except Exception as e:

        print("Attendance data error:", e)

    return jsonify(records)


# ============================================================
# SUMMARY
# ============================================================

@app.route("/summary")
def summary():

    load_students()

    total_students = len(students)

    present_today = 0

    today = datetime.now().strftime("%Y-%m-%d")

    unique_students = set()

    if os.path.exists(ATTENDANCE_FILE):

        try:

            with open(
                ATTENDANCE_FILE,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    if row.get("Date") == today:

                        face_id = row.get("FaceID")

                        if face_id:
                            unique_students.add(face_id)

        except Exception as e:

            print("Summary error:", e)

    present_today = len(unique_students)

    return jsonify({

        "total_students": total_students,

        "present_today": present_today,

        "date": today

    })


# ============================================================
# RELOAD STUDENTS
# ============================================================

@app.route("/reload-students")
def reload_students():

    if not admin_required():

        return redirect("/login")

    load_students()

    return redirect("/students")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("==========================================")
    print(" Face Recognition Attendance System")
    print("==========================================")
    print("Admin Username : admin")
    print("Admin Password : admin123")
    print("------------------------------------------")
    print("Web server:")
    print("http://127.0.0.1:5000")
    print("==========================================")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )