"""LBPH-based face recognizer: trains on enrolled face images and predicts identity."""
import json

import cv2
import numpy as np

import config


class FaceRecognizer:
    def __init__(self):
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._label_to_name = {}
        self._trained = False
        self._load_if_available()

    def _load_if_available(self):
        if config.MODEL_PATH.exists() and config.LABELS_PATH.exists():
            self._model.read(str(config.MODEL_PATH))
            with open(config.LABELS_PATH, "r", encoding="utf-8") as f:
                self._label_to_name = {int(k): v for k, v in json.load(f).items()}
            self._trained = True

    @property
    def is_trained(self):
        return self._trained

    def train_from_disk(self):
        """(Re)train the model from every image under config.FACES_DIR/<name>/*.jpg"""
        faces, labels = [], []
        label_to_name = {}
        next_label = 0

        for person_dir in sorted(config.FACES_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            image_paths = list(person_dir.glob("*.jpg"))
            if not image_paths:
                continue
            label_to_name[next_label] = person_dir.name
            for img_path in image_paths:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces.append(img)
                labels.append(next_label)
            next_label += 1

        if not faces:
            raise ValueError("No enrolled face images found — enroll a user first.")

        self._model.train(faces, np.array(labels))
        self._model.save(str(config.MODEL_PATH))
        with open(config.LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(label_to_name, f)

        self._label_to_name = label_to_name
        self._trained = True

    def predict(self, face_gray):
        """Return (name, confidence). confidence is an LBPH distance — lower is better.
        Returns (config.UNKNOWN_LABEL, None) if the model isn't trained yet."""
        if not self._trained:
            return config.UNKNOWN_LABEL, None

        label, confidence = self._model.predict(face_gray)
        name = self._label_to_name.get(label, config.UNKNOWN_LABEL)
        if confidence > config.CONFIDENCE_THRESHOLD:
            return config.UNKNOWN_LABEL, confidence
        return name, confidence
