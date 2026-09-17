"""
register_faces.py
------------------
STEP 1 of the pipeline.

Captures face images from your webcam for a new person and stores them in
the dataset/ folder. Also keeps a students.csv file mapping numeric IDs to
names (the recognizer only works with numeric labels internally).

Run this once per person you want the system to recognize.

Usage:
    python register_faces.py
"""

import cv2
import os
import csv

DATASET_DIR = "dataset"
STUDENTS_CSV = "students.csv"
SAMPLES_TO_CAPTURE = 60  # number of face images to collect per person


def get_next_id():
    """Look at students.csv to figure out the next free numeric ID."""
    if not os.path.exists(STUDENTS_CSV):
        return 1
    with open(STUDENTS_CSV, "r", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if not rows:
        return 1
    return max(int(row[0]) for row in rows) + 1


def save_student(student_id, name):
    file_exists = os.path.exists(STUDENTS_CSV)
    with open(STUDENTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "name"])
        writer.writerow([student_id, name])


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    name = input("Enter the person's full name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    student_id = get_next_id()
    print(f"Assigned ID: {student_id} for '{name}'")

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions/index.")
        return

    cam.set(3, 640)
    cam.set(4, 480)

    count = 0
    print("\nLook at the camera. Capturing images... Press 'q' to stop early.\n")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            filename = f"{DATASET_DIR}/User.{student_id}.{count}.jpg"
            cv2.imwrite(filename, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{SAMPLES_TO_CAPTURE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Registering Face - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= SAMPLES_TO_CAPTURE:
            break

    cam.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No face samples captured. Nothing was saved.")
        return

    save_student(student_id, name)
    print(f"\nDone. Captured {count} images for '{name}' (ID {student_id}).")
    print("Next step: run `python train_model.py` to train the recognizer.")


if __name__ == "__main__":
    main()
