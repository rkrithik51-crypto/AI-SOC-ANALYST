"""
anomaly_detector.py
Isolation Forest-based anomaly detection for authentication and access logs.

Features engineered from raw logs:
  - failed_attempts, api_calls, files_accessed (direct)
  - is_foreign_login (binary: login from non-home country)
  - is_off_hours (binary: login outside 08:00-18:00)
  - is_vpn, is_tor (binary risk signals)
  - login_hour, day_of_week (temporal features)
  - event_type (encoded: LOGIN_FAILED=2, LOGIN_SUCCESS=1, else 0)

The model returns a continuous anomaly score in [0, 1] (higher = more anomalous)
and a binary flag. The score is derived from the raw Isolation Forest
decision_function which returns negative values for anomalies.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_COLS = [
    "failed_attempts",
    "api_calls",
    "files_accessed",
    "is_foreign_login",
    "is_off_hours",
    "is_vpn_int",
    "is_tor_int",
    "login_hour",
    "day_of_week",
    "event_type_enc",
]

MODEL_PATH = "models/isolation_forest.pkl"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["login_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_foreign_login"] = (df["country"] != "United States").astype(int)
    df["is_off_hours"] = ((df["login_hour"] < 8) | (df["login_hour"] > 18)).astype(int)
    df["is_vpn_int"] = df["is_vpn"].astype(int)
    df["is_tor_int"] = df["is_tor"].astype(int)
    df["event_type_enc"] = df["event_type"].map({
        "LOGIN_FAILED": 2,
        "LOGIN_SUCCESS": 1,
        "FILE_ACCESS": 0,
        "API_CALL": 0,
    }).fillna(0)
    return df


def train_model(
    df: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    save: bool = True,
) -> Pipeline:
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("iso_forest", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        ))
    ])
    pipeline.fit(X)

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"Model saved -> {MODEL_PATH}")

    return pipeline


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run train_model() first."
        )
    return joblib.load(MODEL_PATH)


def score_to_probability(raw_scores: np.ndarray) -> np.ndarray:
    """
    Convert Isolation Forest decision_function output to [0,1] anomaly probability.
    Raw scores are negative for anomalies and positive for normal points.
    We clip and invert so that 1.0 = highest anomaly.
    """
    clipped = np.clip(raw_scores, -0.5, 0.5)
    normalized = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-9)
    return 1.0 - normalized


def detect_anomalies(
    df: pd.DataFrame,
    pipeline: Pipeline = None,
    threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Run anomaly detection on a log DataFrame.
    Returns df with added columns:
      anomaly_score (float, 0-1), is_anomaly (bool), anomaly_flag (str)
    """
    if pipeline is None:
        pipeline = load_model()

    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)

    raw_scores = pipeline.decision_function(X)
    df["anomaly_score"] = score_to_probability(raw_scores)
    df["is_anomaly"] = df["anomaly_score"] >= threshold

    def flag(row):
        if not row["is_anomaly"]:
            return "NORMAL"
        if row["anomaly_score"] >= 0.85:
            return "CRITICAL"
        if row["anomaly_score"] >= 0.75:
            return "HIGH"
        return "MEDIUM"

    df["anomaly_flag"] = df.apply(flag, axis=1)
    return df


def aggregate_user_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate anomaly scores per user to produce a user-level risk report.
    Returns a DataFrame with one row per flagged user.
    """
    flagged = df[df["is_anomaly"]].copy()
    if flagged.empty:
        return pd.DataFrame()

    summary = (
        flagged.groupby("user")
        .agg(
            max_anomaly_score=("anomaly_score", "max"),
            mean_anomaly_score=("anomaly_score", "mean"),
            event_count=("user", "count"),
            countries=("country", lambda x: list(x.unique())),
            anomaly_types=("anomaly_injected", lambda x: list(x.dropna().unique())),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
        .sort_values("max_anomaly_score", ascending=False)
    )

    def risk_level(score):
        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.75:
            return "HIGH"
        if score >= 0.65:
            return "MEDIUM"
        return "LOW"

    summary["risk_level"] = summary["max_anomaly_score"].apply(risk_level)
    return summary


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.generate_logs import generate_logs

    print("Generating logs...")
    df = generate_logs(n_normal=2000, n_anomalies=5)

    print("\nTraining Isolation Forest model...")
    pipeline = train_model(df, contamination=0.08)

    print("\nScoring logs...")
    scored = detect_anomalies(df, pipeline)
    print(f"Anomalies detected: {scored['is_anomaly'].sum()} / {len(scored)}")

    print("\nUser risk summary:")
    risk = aggregate_user_risk(scored)
    print(risk[["user", "risk_level", "max_anomaly_score", "event_count", "countries"]].to_string(index=False))
