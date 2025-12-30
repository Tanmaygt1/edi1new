import sys, os, time, json, csv, threading
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from playsound import playsound

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QProgressBar
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOUND_DIR = os.path.join(BASE_DIR, "sounds")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(LOG_DIR, "session_reports")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================= SOUND =================
def play(sound):
    threading.Thread(
        target=playsound,
        args=(os.path.join(SOUND_DIR, sound),),
        daemon=True
    ).start()

# ================= FACEMESH =================
mp_face = mp.solutions.face_mesh
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH     = [13,14,78,308]

def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))

def EAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    return (dist(pts[1],pts[5]) + dist(pts[2],pts[4])) / (2*dist(pts[0],pts[3]))

def MAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    return dist(pts[0],pts[1]) / dist(pts[2],pts[3])

# ================= UI =================
class DrowsinessUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDI Driver Drowsiness Detection System")
        self.setFixedSize(1100, 640)
        self.setStyleSheet("background:#121212;color:white;")

        # -------- UI Widgets --------
        self.camera = QLabel()
        self.camera.setFixedSize(640, 480)
        self.camera.setStyleSheet("border:2px solid #444;")

        self.status = QLabel("Status: Idle")
        self.status.setStyleSheet("font-size:16px;")

        self.start_btn = QPushButton("▶ Start Detection")
        self.stop_btn = QPushButton("■ Stop")
        self.export_btn = QPushButton("⬇ Export Session")

        for b in [self.start_btn, self.stop_btn, self.export_btn]:
            b.setStyleSheet(
                "padding:8px;font-size:14px;"
                "background:#1f1f1f;border:1px solid #555;"
            )

        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setMaximum(100)
        self.fatigue_bar.setStyleSheet(
            "QProgressBar {height:25px;text-align:center;}"
            "QProgressBar::chunk {background:#e53935;}"
        )

        side = QVBoxLayout()
        side.addWidget(self.status)
        side.addSpacing(10)
        side.addWidget(QLabel("Fatigue Risk Score"))
        side.addWidget(self.fatigue_bar)
        side.addSpacing(20)
        side.addWidget(self.start_btn)
        side.addWidget(self.stop_btn)
        side.addWidget(self.export_btn)
        side.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.camera)
        main.addLayout(side)
        self.setLayout(main)

        # -------- Signals --------
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.export_btn.clicked.connect(self.export_session)

        # -------- Detection State --------
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

        self.ear_buf = deque(maxlen=10)
        self.mar_buf = deque(maxlen=10)
        self.blink_times = deque(maxlen=60)

        self.blink_state = 0
        self.ms_frames = 0
        self.yawn_frames = 0

        self.microsleeps = 0
        self.yawns = 0
        self.fatigue = 0

        self.FPS = 20
        self.session_log = []

    # ================= START / STOP =================
    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer.start(30)
        self.status.setText("Status: Running")

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.status.setText("Status: Stopped")

    # ================= EXPORT =================
    def export_session(self):
        ts = int(time.time())
        csv_path = os.path.join(LOG_DIR, f"session_{ts}.csv")
        report_path = os.path.join(REPORT_DIR, f"session_{ts}_summary.txt")

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","EAR","MAR","blinks_per_min",
                "yawns","microsleeps","fatigue"
            ])
            writer.writerows(self.session_log)

        fatigue_vals = [row[6] for row in self.session_log]
        peak = max(fatigue_vals) if fatigue_vals else 0

        risk = "LOW" if peak < 30 else "MODERATE" if peak < 60 else "HIGH"

        with open(report_path, "w") as r:
            r.write("DRIVER DROWSINESS SESSION SUMMARY\n")
            r.write("="*40+"\n\n")
            r.write(f"Peak Fatigue Score : {peak}\n")
            r.write(f"Yawns Detected     : {self.yawns}\n")
            r.write(f"Microsleeps        : {self.microsleeps}\n")
            r.write(f"Risk Level         : {risk}\n")

        self.status.setText("Session exported with summary")

    # ================= MAIN LOOP =================
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        h,w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        ear = mar = 0
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            ear = (EAR(lm,LEFT_EYE,w,h)+EAR(lm,RIGHT_EYE,w,h))/2
            mar = MAR(lm,MOUTH,w,h)

        self.ear_buf.append(ear)
        self.mar_buf.append(mar)

        ear_avg = np.mean(self.ear_buf)
        mar_avg = np.mean(self.mar_buf)

        # -------- Blink --------
        if ear < 0.21 and self.blink_state == 0:
            self.blink_state = 1
        elif ear >= 0.21 and self.blink_state == 1:
            self.blink_times.append(time.time())
            self.blink_state = 0

        self.blink_times = deque(t for t in self.blink_times if time.time()-t <= 60)
        blinks_per_min = len(self.blink_times)

        # -------- Microsleep --------
        if ear < 0.175:
            self.ms_frames += 1
            if self.ms_frames == int(3*self.FPS):
                self.microsleeps += 1
                self.fatigue += 15
                play("alarm.wav")
        else:
            self.ms_frames = 0

        # -------- Yawn --------
        if mar_avg > 0.55:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= int(1.2*self.FPS):
                self.yawns += 1
                self.fatigue += 8
                play("beep.wav")
            self.yawn_frames = 0

        self.fatigue = min(max(self.fatigue - 0.02, 0), 100)
        self.fatigue_bar.setValue(int(self.fatigue))

        # -------- LOG --------
        self.session_log.append([
            time.time(),
            round(ear_avg,3),
            round(mar_avg,3),
            blinks_per_min,
            self.yawns,
            self.microsleeps,
            int(self.fatigue)
        ])

        # -------- OVERLAY ON CAMERA --------
        cv2.putText(frame,f"EAR: {ear_avg:.3f}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"MAR: {mar_avg:.3f}",(10,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"Blinks/min: {blinks_per_min}",(10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"Yawns: {self.yawns}",(10,130),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv2.putText(frame,f"Microsleeps: {self.microsleeps}",(10,160),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        img = QImage(frame.data, w, h, 3*w, QImage.Format_BGR888)
        self.camera.setPixmap(QPixmap.fromImage(img))

# ================= START =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = DrowsinessUI()
    ui.show()
    sys.exit(app.exec_()) 