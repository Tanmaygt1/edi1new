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
CALIB_PATH = os.path.join(BASE_DIR, "calibration.json")

os.makedirs(LOG_DIR, exist_ok=True)

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

# ================= LOAD CALIBRATION =================
DEFAULTS = {"ear_blink_t":0.21, "ear_microsleep_t":0.175, "mar_yawn_t":0.55}

if os.path.exists(CALIB_PATH):
    cal = json.load(open(CALIB_PATH))
    EAR_BLINK_T = cal["ear_blink_t"]
    EAR_MICROSLEEP_T = cal["ear_microsleep_t"]
    MAR_YAWN_T = cal["mar_yawn_t"]
else:
    EAR_BLINK_T, EAR_MICROSLEEP_T, MAR_YAWN_T = DEFAULTS.values()

# ================= SUMMARY GENERATOR =================
def generate_session_summary(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        return

    duration = int(float(rows[-1]["timestamp"]) - float(rows[0]["timestamp"]))
    avg_ear = sum(float(r["EAR"]) for r in rows) / len(rows)
    avg_mar = sum(float(r["MAR"]) for r in rows) / len(rows)

    yawns = int(rows[-1]["yawns"])
    microsleeps = int(rows[-1]["microsleeps"])
    fatigue = int(rows[-1]["fatigue"])
    blinks = max(int(r["blinks_per_min"]) for r in rows)

    risk = "LOW" if fatigue < 30 else "MODERATE" if fatigue < 65 else "HIGH"

    summary_path = csv_path.replace(".csv", "_summary.txt")

    with open(summary_path, "w") as f:
        f.write("DRIVER DROWSINESS SESSION SUMMARY\n")
        f.write("--------------------------------\n")
        f.write(f"Session Duration : {duration} seconds\n")
        f.write(f"Average EAR      : {avg_ear:.3f}\n")
        f.write(f"Average MAR      : {avg_mar:.3f}\n")
        f.write(f"Yawns            : {yawns}\n")
        f.write(f"Microsleeps      : {microsleeps}\n")
        f.write(f"Blink Rate       : {blinks} / min\n")
        f.write(f"Fatigue Score    : {fatigue}\n")
        f.write(f"Risk Level       : {risk}\n")

# ================= UI =================
class DrowsinessUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EDI Driver Drowsiness Detection System")
        self.setFixedSize(1100, 620)
        self.setStyleSheet("background-color:#121212;color:white;")

        self.camera = QLabel()
        self.camera.setFixedSize(640, 480)
        self.camera.setStyleSheet("border:2px solid #444;")

        self.status = QLabel("Status: Idle")

        self.start_btn = QPushButton("▶ Start Detection")
        self.stop_btn = QPushButton("■ Stop")
        self.export_btn = QPushButton("⬇ Export Session")

        for b in [self.start_btn, self.stop_btn, self.export_btn]:
            b.setStyleSheet("padding:8px;background:#1f1f1f;border:1px solid #555;")

        self.fatigue_bar = QProgressBar()
        self.fatigue_bar.setMaximum(100)

        side = QVBoxLayout()
        side.addWidget(self.status)
        side.addWidget(QLabel("Fatigue Level"))
        side.addWidget(self.fatigue_bar)
        side.addWidget(self.start_btn)
        side.addWidget(self.stop_btn)
        side.addWidget(self.export_btn)
        side.addStretch()

        main = QHBoxLayout()
        main.addWidget(self.camera)
        main.addLayout(side)
        self.setLayout(main)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.export_btn.clicked.connect(self.export_session)

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

        self.yawns = 0
        self.microsleeps = 0
        self.fatigue = 0

        self.session_log = []
        self.FPS = 20

    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer.start(30)
        self.status.setText("Status: Running")

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.status.setText("Status: Stopped")

    def export_session(self):
        fname = f"session_{int(time.time())}.csv"
        path = os.path.join(LOG_DIR, fname)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp","EAR","MAR","blinks_per_min",
                "yawns","microsleeps","fatigue"
            ])
            writer.writerows(self.session_log)

        generate_session_summary(path)
        self.status.setText("Session exported + summary generated")

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

        if ear < EAR_BLINK_T and self.blink_state == 0:
            self.blink_state = 1
        elif ear >= EAR_BLINK_T and self.blink_state == 1:
            self.blink_times.append(time.time())
            self.blink_state = 0

        self.blink_times = deque(t for t in self.blink_times if time.time()-t <= 60)
        blinks_per_min = len(self.blink_times)

        if ear < EAR_MICROSLEEP_T:
            self.ms_frames += 1
            if self.ms_frames == int(3*self.FPS):
                self.microsleeps += 1
                self.fatigue += 15
                play("alarm.wav")
        else:
            self.ms_frames = 0

        if mar_avg > MAR_YAWN_T:
            self.yawn_frames += 1
        else:
            if self.yawn_frames >= int(1.2*self.FPS):
                self.yawns += 1
                self.fatigue += 8
                play("beep.wav")
            self.yawn_frames = 0

        self.fatigue = min(max(self.fatigue - 0.02, 0), 100)
        self.fatigue_bar.setValue(int(self.fatigue))

        self.session_log.append([
            time.time(), round(ear_avg,3), round(mar_avg,3),
            blinks_per_min, self.yawns, self.microsleeps, int(self.fatigue)
        ])

        img = QImage(frame.data, w, h, 3*w, QImage.Format_BGR888)
        self.camera.setPixmap(QPixmap.fromImage(img))

# ================= START =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = DrowsinessUI()
    ui.show()
    sys.exit(app.exec_())
