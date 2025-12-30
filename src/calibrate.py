import cv2
import mediapipe as mp
import numpy as np
import json
import time
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_PATH = os.path.join(BASE_DIR, "calibration.json")

mp_face = mp.solutions.face_mesh
LEFT_EYE  = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]
MOUTH     = [13,14,78,308]

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

def EAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    return (dist(pts[1],pts[5]) + dist(pts[2],pts[4])) / (2*dist(pts[0],pts[3]))

def MAR(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    return dist(pts[0],pts[1]) / dist(pts[2],pts[3])

def run_calibration():
    cap = cv2.VideoCapture(0)
    ear_vals, mar_vals = [], []

    print("Calibration started. Please look normally for 5 seconds.")

    with mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
        start = time.time()
        while time.time() - start < 5:
            ret, frame = cap.read()
            if not ret:
                break

            h,w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = fm.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                ear = (EAR(lm,LEFT_EYE,w,h)+EAR(lm,RIGHT_EYE,w,h))/2
                mar = MAR(lm,MOUTH,w,h)
                ear_vals.append(ear)
                mar_vals.append(mar)

            cv2.putText(frame,"Calibrating...",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    ear_med = float(np.median(ear_vals))
    mar_med = float(np.median(mar_vals))

    calibration = {
        "ear_blink_t": round(ear_med * 0.75, 3),
        "ear_microsleep_t": round(ear_med * 0.6, 3),
        "mar_yawn_t": round(mar_med * 1.5, 3),
        "created": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(CALIB_PATH, "w") as f:
        json.dump(calibration, f, indent=4)

    print("Calibration saved:", calibration)

if __name__ == "__main__":
    run_calibration()

