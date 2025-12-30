# src/main_mvp.py
import cv2, mediapipe as mp, numpy as np, time, os, json, pickle, threading
from collections import deque
from playsound import playsound
from shared_state import state

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(BASE_DIR, "calibration.json")
SOUND_DIR = os.path.join(BASE_DIR, "sounds")

def play(sound):
    threading.Thread(
        target=playsound,
        args=(os.path.join(SOUND_DIR, sound),),
        daemon=True
    ).start()

# ---------------- FACEMESH ----------------
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

# ---------------- CALIBRATION ----------------
DEFAULTS = {"ear_blink_t":0.21,"ear_microsleep_t":0.175,"mar_yawn_t":0.55}
if os.path.exists(CALIB_PATH):
    cal = json.load(open(CALIB_PATH))
else:
    cal = DEFAULTS

EAR_BLINK_T = cal["ear_blink_t"]
EAR_MICROSLEEP_T = cal["ear_microsleep_t"]
MAR_YAWN_T = cal["mar_yawn_t"]

# ---------------- DETECTION THREAD ----------------
def run_detection():
    cap = cv2.VideoCapture(0)
    ear_buf = deque(maxlen=10)
    mar_buf = deque(maxlen=10)
    blink_times = deque(maxlen=60)

    blink_state = 0
    ms_frames = 0
    yawn_frames = 0
    microsleeps = 0
    yawns = 0
    fatigue = 0

    FPS = 20

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        while state["running"]:
            ret, frame = cap.read()
            if not ret:
                continue

            h,w = frame.shape[:2]
            res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            ear = mar = 0
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                ear = (EAR(lm,LEFT_EYE,w,h)+EAR(lm,RIGHT_EYE,w,h))/2
                mar = MAR(lm,MOUTH,w,h)

            ear_buf.append(ear)
            mar_buf.append(mar)
            ear_avg = np.mean(ear_buf)
            mar_avg = np.mean(mar_buf)

            # Blink
            if ear < EAR_BLINK_T and blink_state == 0:
                blink_state = 1
            elif ear >= EAR_BLINK_T and blink_state == 1:
                blink_times.append(time.time())
                blink_state = 0

            blink_times = deque(t for t in blink_times if time.time()-t <= 60)
            blinks_per_min = len(blink_times)

            # Yawn
            if mar_avg > MAR_YAWN_T:
                yawn_frames += 1
            else:
                if yawn_frames > int(1.2*FPS):
                    yawns += 1
                    fatigue += 8
                    play("beep.wav")
                yawn_frames = 0

            # Microsleep (FIXED)
            if ear < EAR_MICROSLEEP_T:
                ms_frames += 1
                if ms_frames == int(3*FPS):
                    microsleeps += 1
                    fatigue += 15
                    play("alarm.wav")
            else:
                ms_frames = 0

            fatigue = min(max(fatigue - 0.02, 0), 100)

            # UPDATE SHARED STATE
            state.update({
                "ear": round(ear_avg,3),
                "mar": round(mar_avg,3),
                "blinks_per_min": blinks_per_min,
                "yawns": yawns,
                "microsleeps": microsleeps,
                "fatigue": int(fatigue),
                "ml_label": "DROWSY" if fatigue > 60 else "ALERT"
            })

    cap.release()
