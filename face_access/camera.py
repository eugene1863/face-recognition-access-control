"""Thin wrapper around cv2.VideoCapture."""
import cv2

import config


class Camera:
    def __init__(self, index=config.CAMERA_INDEX):
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {index}. "
                "Check that a webcam is connected and not in use by another app."
            )

    def read(self):
        """Return a BGR frame, or None if the read failed."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
