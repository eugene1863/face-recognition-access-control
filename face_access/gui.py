"""Tkinter desktop UI: live camera feed, face recognition overlay, enrollment, and access log."""
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

import config
import database
from camera import Camera
from detector import FaceDetector
from enrollment import EnrollmentSession
from recognizer import FaceRecognizer

GRANTED_COLOR = (0, 200, 0)   # BGR
DENIED_COLOR = (0, 0, 220)
SCANNING_COLOR = (0, 165, 255)


def _largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Face Access Control")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        database.init_db()
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.camera = Camera()

        self.mode = "recognize"          # "recognize" | "enroll"
        self.enrollment_session = None
        self._last_logged = {}           # name -> last logged timestamp (cooldown)
        self._photo = None                # keep a reference so Tk doesn't GC it

        self._build_widgets()
        self._tick()

    # ---------- UI layout ----------

    def _build_widgets(self):
        self.video_label = tk.Label(self)
        self.video_label.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

        self.status_var = tk.StringVar(value="Starting camera...")
        status_label = tk.Label(
            self, textvariable=self.status_var, font=("Segoe UI", 14, "bold")
        )
        status_label.grid(row=1, column=0, columnspan=3, pady=(0, 8))

        enroll_btn = tk.Button(self, text="Enroll New User", command=self._start_enrollment)
        enroll_btn.grid(row=2, column=0, padx=8, pady=(0, 10), sticky="ew")

        log_btn = tk.Button(self, text="View Access Log", command=self._show_log)
        log_btn.grid(row=2, column=1, padx=8, pady=(0, 10), sticky="ew")

        quit_btn = tk.Button(self, text="Quit", command=self._on_close)
        quit_btn.grid(row=2, column=2, padx=8, pady=(0, 10), sticky="ew")

    # ---------- main video loop ----------

    def _tick(self):
        frame = self.camera.read()
        if frame is None:
            self.status_var.set("Camera read failed")
            self.after(200, self._tick)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self.detector.detect(gray)

        if self.mode == "enroll":
            self._handle_enroll_frame(frame, gray, boxes)
        else:
            self._handle_recognize_frame(frame, gray, boxes)

        self._render(frame)
        self.after(30, self._tick)

    def _handle_recognize_frame(self, frame, gray, boxes):
        if not boxes:
            self.status_var.set("No face detected")
            return

        any_granted = False
        for box in boxes:
            x, y, w, h = box
            face = FaceDetector.crop_and_normalize(gray, box)
            name, confidence = self.recognizer.predict(face)
            granted = name != config.UNKNOWN_LABEL
            any_granted = any_granted or granted

            color = GRANTED_COLOR if granted else DENIED_COLOR
            label = f"{name}" + (f" ({confidence:.0f})" if confidence is not None else "")
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, max(y - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

            self._maybe_log(name, granted, confidence)

        self.status_var.set("ACCESS GRANTED" if any_granted else "ACCESS DENIED")

    def _maybe_log(self, name, granted, confidence):
        now = time.time()
        last = self._last_logged.get(name, 0)
        if now - last < config.RECOGNITION_COOLDOWN_SEC:
            return
        self._last_logged[name] = now

        user = database.get_user_by_name(name) if granted else None
        database.log_access(
            user_id=user["id"] if user else None,
            name=name,
            status="GRANTED" if granted else "DENIED",
            confidence=confidence,
        )

    def _handle_enroll_frame(self, frame, gray, boxes):
        session = self.enrollment_session
        box = _largest_box(boxes)

        if box is None:
            self.status_var.set(f"Enrolling {session.name}: show your face ({session.progress})")
            return

        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), SCANNING_COLOR, 2)

        captured = session.offer_frame(gray, box)
        if captured:
            self.status_var.set(f"Enrolling {session.name}: captured {session.progress}")
        else:
            self.status_var.set(f"Enrolling {session.name}: hold still ({session.progress})")

        if session.is_done:
            self._finish_enrollment()

    def _render(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self._photo)

    # ---------- enrollment flow ----------

    def _start_enrollment(self):
        if self.mode == "enroll":
            return
        name = simpledialog.askstring(
            "Enroll New User",
            "Enter the person's name:\n(will be saved as a filesystem-safe id)",
            parent=self,
        )
        if not name or not name.strip():
            return
        try:
            self.enrollment_session = EnrollmentSession(name)
        except ValueError as e:
            messagebox.showerror("Enrollment error", str(e))
            return
        self.mode = "enroll"

    def _finish_enrollment(self):
        session = self.enrollment_session
        try:
            self.recognizer = session.finalize()
            messagebox.showinfo(
                "Enrollment complete",
                f"Captured {session.samples_captured} samples for '{session.name}'.\n"
                "The recognition model has been retrained.",
            )
        except ValueError as e:
            messagebox.showerror("Training error", str(e))
        finally:
            self.enrollment_session = None
            self.mode = "recognize"

    # ---------- access log viewer ----------

    def _show_log(self):
        win = tk.Toplevel(self)
        win.title("Access Log")
        win.geometry("720x400")

        columns = ("timestamp", "name", "status", "confidence")
        tree = ttk.Treeview(win, columns=columns, show="headings")
        for col, width in zip(columns, (170, 200, 100, 100)):
            tree.heading(col, text=col.capitalize())
            tree.column(col, width=width, anchor="center")
        tree.pack(fill="both", expand=True, padx=8, pady=8)

        for row in database.recent_log():
            confidence = f"{row['confidence']:.1f}" if row["confidence"] is not None else "-"
            tag = "granted" if row["status"] == "GRANTED" else "denied"
            tree.insert(
                "", "end",
                values=(row["timestamp"], row["name"], row["status"], confidence),
                tags=(tag,),
            )
        tree.tag_configure("granted", foreground="#0a8a0a")
        tree.tag_configure("denied", foreground="#c0392b")

    # ---------- shutdown ----------

    def _on_close(self):
        self.camera.release()
        self.destroy()


def run():
    app = App()
    app.mainloop()
