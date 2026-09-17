"""
train_model.py
---------------
STEP 2 of the pipeline.

Reads all images in dataset/ (created by register_faces.py), trains an
LBPH (Local Binary Patterns Histograms) face recognizer, and saves the
trained model to trainer/trainer.yml.

Run this every time you add a new person (or new samples) to the dataset.

Usage:
    python train_model.py
"""

import cv2
import numpy as np
import os
from PIL import Image

DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = os.path.join(TRAINER_DIR, "trainer.yml")


def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    face_samples = []
    ids = []

    for image_path in image_paths:
        pil_img = Image.open(image_path).convert("L")  # grayscale
        img_numpy = np.array(pil_img, "uint8")

        try:
            student_id = int(os.path.split(image_path)[-1].split(".")[1])
        except (IndexError, ValueError):
            continue

        faces = face_detector.detectMultiScale(img_numpy)
        if len(faces) == 0:
            # image was already a cropped face from register_faces.py
            face_samples.append(img_numpy)
            ids.append(student_id)
        else:
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])
                ids.append(student_id)

    return face_samples, ids


def main():
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("ERROR: dataset/ is empty. Run register_faces.py first.")
        return

    os.makedirs(TRAINER_DIR, exist_ok=True)

    print("Training model, this may take a moment...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels(DATASET_DIR)

    if len(faces) == 0:
        print("ERROR: No valid face samples found in dataset/.")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)

    print(f"Training complete. {len(set(ids))} unique person(s), {len(faces)} images used.")
    print(f"Model saved to {TRAINER_FILE}")
    print("Next step: run `python attendance_system.py` to start marking attendance.")


if __name__ == "__main__":
    main()
