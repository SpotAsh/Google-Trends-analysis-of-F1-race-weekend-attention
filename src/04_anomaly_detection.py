from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 1. Directory Setup
PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "f1_anomalies_features.csv"

print("Loading engineered feature dataset...")
df = pd.read_csv(INPUT_FILE)

# 2. Select Features for Unsupervised Anomaly Detection
# We evaluate peak search, mean search, spike ratio, and z-score
feature_cols = [
    "race_week_peak",
    "race_week_mean",
    "spike_factor",
    "spike_zscore",
]
X = df[feature_cols].fillna(0)

# 3. Isolation Forest Model
# Contamination set to ~10% based on expected high-interest F1 outliers
iso_forest = IsolationForest(
    n_estimators=100, contamination=0.10, random_state=42
)
df["iso_forest_anomaly"] = iso_forest.fit_predict(X)
# IsolationForest outputs -1 for anomalies, 1 for normal. Convert to binary flag:
df["is_iso_anomaly"] = df["iso_forest_anomaly"] == -1

# 4. Local Outlier Factor (Density-Based Detection)
lof = LocalOutlierFactor(n_neighbors=10, contamination=0.10)
df["is_lof_anomaly"] = lof.fit_predict(X) == -1

# 5. Model Consensus (Flagged by both Z-Score AND ML models)
df["is_high_confidence_anomaly"] = (
    df["is_anomaly_candidate"] & df["is_iso_anomaly"] & df["is_lof_anomaly"]
)

# 6. Save Model Output
output_file = PROCESSED_DIR / "f1_anomalies_detected.csv"
df.to_csv(output_file, index=False)

print(f"\nSUCCESS! Anomaly detection completed.")
print(f"Results exported to: {output_file.resolve()}\n")

# Summary of Detected Anomalies
print(f"Total Events Evaluated: {len(df)}")
print(f"Rule-Based Z-Score Candidates: {df['is_anomaly_candidate'].sum()}")
print(f"Isolation Forest Anomalies: {df['is_iso_anomaly'].sum()}")
print(f"Local Outlier Factor Anomalies: {df['is_lof_anomaly'].sum()}")
print(
    f"High-Confidence Consensus Anomalies: {df['is_high_confidence_anomaly'].sum()}"
)

print("\nTop High-Confidence Anomalies:")
print(
    df[df["is_high_confidence_anomaly"]][
        ["race_id", "grand_prix", "spike_factor", "spike_zscore"]
    ].to_string(index=False)
)