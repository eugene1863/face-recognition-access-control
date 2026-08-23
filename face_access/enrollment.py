"""Enrollment: captures a sequence of face samples for a new user and (re)trains the model.

Designed to be driven frame-by-frame from the GUI's live video loop (not a blocking
capture loop), so the same camera feed is reused for both enrollment and recognition.
"""
import re
import time

import cv2

import config
import database
from detector import FaceDetector
from recognizer import FaceRecognizer


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    return slug.strip("_") or "user"


class EnrollmentSession:
    def __init__(self, name: str, samples_needed: int = config.SAMPLES_PER_USER):
        self.name = name.strip()
        if not self.name:
            raise ValueError("Name cannot be empty.")
        self.slug = _slugify(self.name)
        self.samples_needed = samples_needed
        self.samples_captured = 0
        self._last_capture_time = 0.0

        self.person_dir = config.FACES_DIR / self.slug
        self.person_dir.mkdir(parents=True, exist_ok=True)
        existing = list(self.person_dir.glob("*.jpg"))
        self._start_index = len(existing)

    @property
    def is_done(self) -> bool:
        return self.samples_captured >= self.samples_needed

    @property
    def progress(self) -> str:
        return f"{self.samples_captured}/{self.samples_needed}"

    def offer_frame(self, gray_frame, box) -> bool:
        """Call once per video frame with the largest detected face box.
        Saves a sample if enough time has passed since the last one.
        Returns True if a sample was captured this call."""
        if self.is_done:
            return False

        now = time.time()
        if (now - self._last_capture_time) * 1000 < config.CAPTURE_INTERVAL_MS:
            return False

        face = FaceDetector.crop_and_normalize(gray_frame, box)
        index = self._start_index + self.samples_captured
        out_path = self.person_dir / f"img_{index:04d}.jpg"
        cv2.imwrite(str(out_path), face)

        self.samples_captured += 1
        self._last_capture_time = now
        return True

    def finalize(self) -> FaceRecognizer:
        """Register the user (idempotent) and retrain the recognizer from all data on disk."""
        database.get_or_create_user(self.slug)
        recognizer = FaceRecognizer()
        recognizer.train_from_disk()
        return recognizer
