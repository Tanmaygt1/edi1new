# src/generate_session_report.py

import os
import csv
import statistics

def generate_report(csv_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(base_dir, "logs", "session_reports")
    os.makedirs(report_dir, exist_ok=True)

    fname = os.path.basename(csv_path).replace(".csv", "_summary.txt")
    report_path = os.path.join(report_dir, fname)

    fatigue_vals = []
    yawns = 0
    microsleeps = 0
    blink_rates = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fatigue_vals.append(float(row["fatigue"]))
            yawns = max(yawns, int(row["yawns"]))
            microsleeps = max(microsleeps, int(row["microsleeps"]))
            blink_rates.append(int(row["blinks_per_min"]))

    avg_fatigue = round(statistics.mean(fatigue_vals), 2)
    peak_fatigue = round(max(fatigue_vals), 2)
    avg_blinks = round(statistics.mean(blink_rates), 2)

    if peak_fatigue < 30:
        risk = "LOW"
    elif peak_fatigue < 60:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    with open(report_path, "w") as r:
        r.write("DRIVER DROWSINESS SESSION REPORT\n")
        r.write("=" * 40 + "\n\n")
        r.write(f"Average Fatigue Score : {avg_fatigue}\n")
        r.write(f"Peak Fatigue Score    : {peak_fatigue}\n")
        r.write(f"Yawns Detected        : {yawns}\n")
        r.write(f"Microsleeps Detected  : {microsleeps}\n")
        r.write(f"Avg Blink Rate (/min) : {avg_blinks}\n\n")
        r.write(f"OVERALL RISK LEVEL    : {risk}\n")

    return report_path
