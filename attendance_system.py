"""
attendance_system.py
---------------------
STEP 3 of the pipeline.

Opens the webcam, detects and recognizes faces using the trained LBPH
model, and marks attendance (once per person per day) in a CSV file
under Attendance/Attendance_YYYY-MM-DD.csv.

Press 'q' to quit.

Usage:
    python attendance_system.py
"""

import cv2
import os
import csv
from datetime import datetime

TRAINER_FILE = os.path.join("trainer", "trainer.yml")
STUDENTS_CSV = "students.csv"
ATTENDANCE_DIR = "Attendance"

# Lower distance = more confident match. Tune this if you get false
# matches (lower it) or too many "Unknown" results (raise it slightly).
CONFIDENCE_THRESHOLD = 65


def load_names():
    names = {}
    if not os.path.exists(STUDENTS_CSV):
        print("WARNING: students.csv not found. Did you run register_faces.py?")
        return names
    with open(STUDENTS_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row:
                names[int(row[0])] = row[1]
    return names


def get_today_file():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(ATTENDANCE_DIR, f"Attendance_{today}.csv")
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Time"])
    return path


def already_marked(path, student_id):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and int(row[0]) == student_id:
                return True
    return False


def mark_attendance(path, student_id, name):
    if already_marked(path, student_id):
        return False
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([student_id, name, datetime.now().strftime("%H:%M:%S")])
    return True


def main():
    if not os.path.exists(TRAINER_FILE):
        print("ERROR: trainer/trainer.yml not found. Run train_model.py first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    names = load_names()
    attendance_file = get_today_file()
    marked_today = set()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    print("Attendance system running. Press 'q' to quit.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                 minSize=(80, 80))

        for (x, y, w, h) in faces:
            student_id, distance = recognizer.predict(gray[y:y + h, x:x + w])

            if distance < CONFIDENCE_THRESHOLD:
                name = names.get(student_id, "Unknown")
                confidence_pct = round(100 - distance)
                if student_id not in marked_today:
                    if mark_attendance(attendance_file, student_id, name):
                        marked_today.add(student_id)
                        print(f"Marked present: {name} (ID {student_id})")
                label = f"{name} ({confidence_pct}%)"
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        cv2.imshow("Attendance System - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Attendance saved to: {attendance_file}")


if __name__ == "__main__":
    main()
