import cv2, os, csv, datetime
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs", "sessions")
SOUND_DIR = os.path.join(BASE_DIR, "sounds")
os.makedirs(LOG_DIR, exist_ok=True)

def play(sound):
    threading.Thread(
        target=playsound,
        args=(os.path.join(SOUND_DIR, sound),),
        daemon=True
    ).start()

LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH     = [13,14,78,308]

def dist(a,b): return np.linalg.norm(np.array(a)-np.array(b))

def EAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    C = dist(pts[0], pts[3])
    return 0 if C==0 else (dist(pts[1],pts[5])+dist(pts[2],pts[4]))/(2*C)

def MAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    W = dist(pts[2], pts[3])
    return 0 if W==0 else dist(pts[0],pts[1])/W

cap = cv2.VideoCapture(0)
mp_face = mp.solutions.face_mesh

session = datetime.datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
path = os.path.join(LOG_DIR, session)

print("Recording to:", path)
print("d = drowsy | a = alert | q = quit")

with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp","ear","mar","label"])

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        label = "alert"
        while True:
            ret, frame = cap.read()
            if not ret: break

            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            ear = mar = 0
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                ear = (EAR(lm,LEFT_EYE,w,h)+EAR(lm,RIGHT_EYE,w,h))/2
                mar = MAR(lm,MOUTH,w,h)

            writer.writerow([datetime.datetime.now(), round(ear,4), round(mar,4), label])

            cv2.putText(frame,f"EAR:{ear:.3f} MAR:{mar:.3f}",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            cv2.putText(frame,f"LABEL:{label}",(10,70),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

            cv2.imshow("Record Data", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == ord('d'):
                label = "drowsy"
                play("beep.wav")
            if k == ord('a'):
                label = "alert"

cap.release()
cv2.destroyAllWindows()
