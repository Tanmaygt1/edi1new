import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESS_DIR = os.path.join(BASE_DIR, "logs")
OUT = os.path.join(BASE_DIR, "logs", "windows_features.csv")

rows = []

for file in os.listdir(SESS_DIR):
    if file.startswith("session_") and file.endswith(".csv"):
        df = pd.read_csv(os.path.join(SESS_DIR, file))

        # REQUIRED COLUMNS CHECK
        required = {"EAR", "MAR", "fatigue"}
        if not required.issubset(df.columns):
            continue

        for i in range(10, len(df)):
            ear_mean = df["EAR"].iloc[i-10:i].mean()
            mar_mean = df["MAR"].iloc[i-10:i].mean()
            fatigue = df["fatigue"].iloc[i]

            label = 1 if fatigue >= 50 else 0   # AUTO LABEL

            rows.append({
                "ear": ear_mean,
                "mar": mar_mean,
                "label": label
            })

if not rows:
    print("No valid session files found.")
else:
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("Saved features to:", OUT)
