import time
from pathlib import Path

import numpy as np
import pandas as pd
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "models" / "fraud_model.onnx"
csv_path = ROOT / "data" / "sample_transactions.csv"

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_names = [o.name for o in session.get_outputs()]

df = pd.read_csv(csv_path)
X = df[[f"feature_{i}" for i in range(10)]].astype(np.float32).values

# Warm-up run
session.run(output_names, {input_name: X[:1]})

start = time.perf_counter()
for i in range(len(X)):
    session.run(output_names, {input_name: X[i:i+1]})
end = time.perf_counter()

avg_ms = ((end - start) / len(X)) * 1000
print(f"Python ONNX Runtime average latency: {avg_ms:.4f} ms per transaction")
print("Input:", input_name)
print("Outputs:", output_names)