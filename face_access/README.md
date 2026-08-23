# Smart Face Access Control

A desktop app that recognizes enrolled faces via your webcam and grants/denies
access, logging every attempt.

## How it works

- **Detection**: OpenCV Haar cascade (bundled with OpenCV, no extra download).
- **Recognition**: OpenCV's LBPH face recognizer, trained on images you enroll.
- **Storage**: SQLite (`data/access.db`) for users and the access log; enrolled
  face images live under `data/faces/<name>/`.
- **UI**: Tkinter desktop window with a live camera feed, bounding boxes, and
  a grant/deny overlay.

No `dlib`/`face_recognition` dependency — those need a C++ build toolchain
and don't have prebuilt wheels for newer Python versions on Windows. This
setup installs with plain `pip`.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Usage

1. Click **Enroll New User**, type a name, and hold your face steady in
   front of the camera until it captures 25 samples (a few seconds). The
   model retrains automatically when enrollment finishes.
2. Once enrolled, step back in front of the camera — you should see a green
   box with your name and **ACCESS GRANTED**. Unrecognized faces show a red
   box and **ACCESS DENIED**.
3. Click **View Access Log** to see the timestamped history of every grant
   and denial.

## Tuning

Edit `config.py`:
- `CONFIDENCE_THRESHOLD` — lower = stricter matching (fewer false accepts,
  more false rejects). Default `65.0`. If real users are getting denied,
  raise it a bit; if strangers are getting granted, lower it.
- `SAMPLES_PER_USER` — more samples generally improve accuracy, at the cost
  of longer enrollment.
- `RECOGNITION_COOLDOWN_SEC` — how often the same person can be re-logged.

## Extending toward real physical access control

`gui.py`'s `_maybe_log` method is the single place a "GRANTED" decision is
made. To drive real hardware:
- **USB relay / Arduino**: send a serial command there on grant.
- **Raspberry Pi GPIO**: pulse a GPIO pin to energize a door strike.
- **Smart lock API**: fire an HTTP request to the lock's API.

## Known limitations

LBPH is a classical (non-deep-learning) method — it's lightweight and needs
no GPU or large model downloads, but is more sensitive to lighting and pose
than modern embedding-based recognizers. For higher accuracy later, swap
`recognizer.py`'s LBPH model for a deep embedding model (e.g. `insightface`
via ONNX Runtime) without touching the rest of the app.
