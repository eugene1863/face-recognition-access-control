"""Face detection using OpenCV's bundled Haar cascade (no external model download)."""
import cv2

import config


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def detect(self, gray_frame):
        """Return a list of (x, y, w, h) bounding boxes for detected faces."""
        faces = self._cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=config.MIN_FACE_SIZE,
        )
        return list(faces)

    @staticmethod
    def crop_and_normalize(gray_frame, box):
        x, y, w, h = box
        face = gray_frame[y : y + h, x : x + w]
        face = cv2.resize(face, config.FACE_SIZE)
        face = cv2.equalizeHist(face)
        return face
