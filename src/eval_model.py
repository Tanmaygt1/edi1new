import pickle, pandas as pd, os
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(BASE_DIR, "models", "rf_model.pkl")
DATA = os.path.join(BASE_DIR, "logs", "windows_features.csv")

clf = pickle.load(open(MODEL,"rb"))
df = pd.read_csv(DATA)

X = df[["ear","mar"]]
y = df["label"]

pred = clf.predict(X)
print(classification_report(y, pred))
