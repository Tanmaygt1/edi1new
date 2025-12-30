# src/train_model.py

import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_PATH = os.path.join(BASE_DIR, "logs", "windows_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------
if not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError(
        "windows_features.csv not found. Run extract_features.py first."
    )

df = pd.read_csv(FEATURES_PATH)

# Expected columns:
# ear, mar, label
print("Loaded data shape:", df.shape)
print("Columns:", list(df.columns))

# ---------------- PREPARE DATA ----------------
X = df[["ear", "mar"]]
y = df["label"]

# Convert labels if needed
# alert -> 0, drowsy -> 1
if y.dtype == object:
    y = y.map({"alert": 0, "drowsy": 1})

# ---------------- TRAIN / TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN MODEL ----------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nMODEL PERFORMANCE")
print("-----------------")
print("Accuracy:", round(acc, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------- SAVE MODEL ----------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to:", MODEL_PATH)
