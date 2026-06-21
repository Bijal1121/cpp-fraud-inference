import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# 10 transaction features. 3 percent fraud cases.
X, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    weights=[0.97, 0.03],
    class_sep=1.8,
    random_state=42,
)

feature_names = [f"feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
report = classification_report(y_test, pred, output_dict=True)
auc = roc_auc_score(y_test, proba)

metrics = {
    "accuracy": report["accuracy"],
    "fraud_precision": report["1"]["precision"],
    "fraud_recall": report["1"]["recall"],
    "fraud_f1": report["1"]["f1-score"],
    "roc_auc": auc,
    "test_rows": len(X_test),
}
with open(ROOT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save sample rows for C++ inference.
sample_df = pd.DataFrame(X_test[:100], columns=feature_names)
sample_df["label"] = y_test[:100]
sample_df.to_csv(DATA_DIR / "sample_transactions.csv", index=False)

# Export model to ONNX. zipmap=False makes probabilities a tensor, easier for C++.
initial_types = [("input", FloatTensorType([None, X.shape[1]]))]
onx = convert_sklearn(
    model,
    initial_types=initial_types,
    options={id(model.named_steps["clf"]): {"zipmap": False}},
)
with open(MODEL_DIR / "fraud_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

print("Saved models/fraud_model.onnx")
print("Saved data/sample_transactions.csv")
print(json.dumps(metrics, indent=2))
