"""
liveness.py
------------
Basic anti-spoofing / liveness detection, used by attendance_system.py to
stop the most common cheap spoofing attempts before marking attendance:

  1. Blink detection - requires a real open -> closed -> open eye cycle to
     be observed. A static printed photo, or a still image displayed on a
     phone/tablet, will never blink, so it can never pass this check.

  2. Image "flatness" check - uses the variance of the Laplacian (a
     standard blur/detail measure) on the face region. Printed photos and
     phone/tablet screens generally produce a flatter, lower-detail image
     than a real face under normal lighting, so unusually flat faces are
     rejected.

  3. A rolling timeout - if liveness isn't proven within a few seconds the
     challenge resets, so someone can't just hold a photo in front of the
     camera indefinitely and wait it out.

HONEST LIMITATIONS (please read):
This is a lightweight, classical-CV liveness check, not a commercial-grade
anti-spoofing system. It meaningfully raises the bar against the most
common cheap attacks (printed photo, static image on a screen), but it is
NOT guaranteed to stop every attack. A determined attacker with a *video
recording of the real person's face, including natural blinking*, could
potentially still defeat it. True robust liveness detection typically
requires infrared/depth cameras or a trained deep-learning model, which is
beyond the scope of a basic local webcam project.
"""

import time
from collections import deque
import cv2


class LivenessDetector:
    def __init__(self, required_blinks=1, challenge_timeout=6.0,
                 flatness_threshold=25.0):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self.required_blinks = required_blinks
        self.challenge_timeout = challenge_timeout
        self.flatness_threshold = flatness_threshold
        self.states = {}

    def _get_state(self, student_id):
        if student_id not in self.states:
            self.states[student_id] = {
                "start_time": time.time(),
                "blinks": 0,
                "eyes_open_prev": True,
                "closing": False,
                "recent_sharpness": deque(maxlen=10),
                "verified": False,
            }
        return self.states[student_id]

    def reset(self, student_id):
        if student_id in self.states:
            del self.states[student_id]

    @staticmethod
    def _sharpness(face_gray):
        return cv2.Laplacian(face_gray, cv2.CV_64F).var()

    def check(self, student_id, gray_frame, face_rect):
        """
        Call this once per frame for each recognized face.
        Returns (verified: bool, status_text: str).
        Once verified, it stays verified for that student_id until reset().
        """
        x, y, w, h = face_rect
        state = self._get_state(student_id)

        if state["verified"]:
            return True, "Live - Verified"

        # Reset the challenge if it's taking too long (prevents someone
        # from just holding up a photo indefinitely).
        if time.time() - state["start_time"] > self.challenge_timeout:
            state["start_time"] = time.time()
            state["blinks"] = 0
            state["closing"] = False

        face_roi = gray_frame[y:y + h, x:x + w]
        if face_roi.size == 0:
            return False, "Positioning..."

        # --- Flatness / texture check ---
        sharpness = self._sharpness(face_roi)
        state["recent_sharpness"].append(sharpness)
        avg_sharpness = sum(state["recent_sharpness"]) / len(state["recent_sharpness"])
        is_flat = avg_sharpness < self.flatness_threshold

        # --- Blink detection ---
        eyes = self.eye_cascade.detectMultiScale(
            face_roi, scaleFactor=1.1, minNeighbors=6, minSize=(15, 15)
        )
        eyes_open_now = len(eyes) >= 2

        if state["eyes_open_prev"] and not eyes_open_now:
            state["closing"] = True
        if state["closing"] and eyes_open_now:
            state["blinks"] += 1
            state["closing"] = False
        state["eyes_open_prev"] = eyes_open_now

        blinked_enough = state["blinks"] >= self.required_blinks

        if blinked_enough and not is_flat:
            state["verified"] = True
            return True, "Live - Verified"

        remaining = max(0, int(self.challenge_timeout - (time.time() - state["start_time"])))
        if is_flat:
            return False, f"Move closer / check lighting ({remaining}s)"
        return False, f"Please blink ({state['blinks']}/{self.required_blinks}) - {remaining}s"
