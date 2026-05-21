"""
main.py — FastAPI SOC Analyst API

Endpoints:
  POST /ingest          Upload a CSV log file for processing
  GET  /alerts          Get all flagged user alerts
  GET  /alerts/{user}   Get detailed incident report for a user
  GET  /logs            Get recent log events (with optional filters)
  GET  /stats           Dashboard summary statistics
  POST /retrain         Retrain the anomaly model on new data
  GET  /health          Health check

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import io
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.anomaly_detector import (
    train_model,
    detect_anomalies,
    aggregate_user_risk,
    load_model,
    engineer_features,
)
from utils.incident_summarizer import IncidentSummarizer
from data.generate_logs import generate_logs

app = FastAPI(
    title="SOC Analyst API",
    description="Security log analysis, anomaly detection, and AI-powered incident reporting",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (replace with Redis/DB in production)
_state: dict = {
    "raw_df": None,
    "scored_df": None,
    "risk_summary": None,
    "model": None,
    "last_ingested": None,
    "reports_cache": {},
}

LLM_BACKEND = os.environ.get("LLM_BACKEND", "claude")


# ---------- Startup: load or generate data ----------

@app.on_event("startup")
async def startup():
    log_path = "data/sample_logs/auth_logs.csv"
    model_path = "models/isolation_forest.pkl"

    if not os.path.exists(log_path):
        print("Generating sample logs...")
        _state["raw_df"] = generate_logs(output_path=log_path)
    else:
        _state["raw_df"] = pd.read_csv(log_path)

    if not os.path.exists(model_path):
        print("Training initial model...")
        _state["model"] = train_model(_state["raw_df"])
    else:
        _state["model"] = load_model()

    print("Scoring logs...")
    _state["scored_df"] = detect_anomalies(_state["raw_df"], _state["model"])
    _state["risk_summary"] = aggregate_user_risk(_state["scored_df"])
    _state["last_ingested"] = datetime.utcnow().isoformat()
    print("SOC API ready.")


# ---------- Response models ----------

class StatsResponse(BaseModel):
    total_events: int
    anomalies_detected: int
    users_flagged: int
    high_severity: int
    medium_severity: int
    last_ingested: Optional[str]


class AlertSummary(BaseModel):
    user: str
    risk_level: str
    max_anomaly_score: float
    event_count: int
    countries: list
    anomaly_types: list
    first_seen: str
    last_seen: str


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/stats", response_model=StatsResponse)
def stats():
    scored = _state["scored_df"]
    risk = _state["risk_summary"]
    if scored is None:
        raise HTTPException(500, "No data loaded")

    high = 0 if risk is None or risk.empty else int((risk["risk_level"].isin(["CRITICAL","HIGH"])).sum())
    med  = 0 if risk is None or risk.empty else int((risk["risk_level"] == "MEDIUM").sum())

    return StatsResponse(
        total_events=len(scored),
        anomalies_detected=int(scored["is_anomaly"].sum()),
        users_flagged=0 if risk is None else len(risk),
        high_severity=high,
        medium_severity=med,
        last_ingested=_state["last_ingested"],
    )


@app.get("/alerts")
def get_alerts(
    min_score: float = Query(0.0, description="Filter by minimum anomaly score"),
    level: Optional[str] = Query(None, description="Filter by risk level: CRITICAL, HIGH, MEDIUM"),
):
    risk = _state["risk_summary"]
    if risk is None or risk.empty:
        return []

    filtered = risk[risk["max_anomaly_score"] >= min_score]
    if level:
        filtered = filtered[filtered["risk_level"] == level.upper()]

    return filtered.to_dict(orient="records")


@app.get("/alerts/{user}")
def get_user_alert(user: str, generate_ai: bool = Query(False)):
    risk = _state["risk_summary"]
    scored = _state["scored_df"]

    if risk is None or risk.empty:
        raise HTTPException(404, "No risk data available")

    row = risk[risk["user"] == user]
    if row.empty:
        raise HTTPException(404, f"User '{user}' not found in flagged alerts")

    result = row.iloc[0].to_dict()
    result["events"] = scored[
        (scored["user"] == user) & (scored["is_anomaly"])
    ].sort_values("anomaly_score", ascending=False).head(20).to_dict(orient="records")

    if generate_ai:
        if user in _state["reports_cache"]:
            result["incident_report"] = _state["reports_cache"][user]
        else:
            summarizer = IncidentSummarizer(backend=LLM_BACKEND)
            try:
                report = summarizer.generate(row.iloc[0], scored)
                _state["reports_cache"][user] = report.to_dict()
                result["incident_report"] = report.to_dict()
            except Exception as e:
                result["incident_report"] = {"error": str(e)}

    return result


@app.get("/logs")
def get_logs(
    user: Optional[str] = None,
    event_type: Optional[str] = None,
    anomaly_only: bool = False,
    limit: int = Query(100, le=1000),
):
    scored = _state["scored_df"]
    if scored is None:
        raise HTTPException(500, "No data loaded")

    df = scored.copy()
    if user:
        df = df[df["user"] == user]
    if event_type:
        df = df[df["event_type"] == event_type.upper()]
    if anomaly_only:
        df = df[df["is_anomaly"]]

    df = df.sort_values("timestamp", ascending=False).head(limit)
    df["timestamp"] = df["timestamp"].astype(str)
    return df.to_dict(orient="records")


@app.post("/ingest")
async def ingest_logs(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    contents = await file.read()
    try:
        new_df = pd.read_csv(io.BytesIO(contents))
        required = {"timestamp", "user", "event_type", "country"}
        missing = required - set(new_df.columns)
        if missing:
            raise HTTPException(400, f"Missing required columns: {missing}")
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    _state["raw_df"] = new_df
    _state["scored_df"] = detect_anomalies(new_df, _state["model"])
    _state["risk_summary"] = aggregate_user_risk(_state["scored_df"])
    _state["last_ingested"] = datetime.utcnow().isoformat()
    _state["reports_cache"] = {}

    return {
        "message": "Logs ingested and scored",
        "total_events": len(new_df),
        "anomalies": int(_state["scored_df"]["is_anomaly"].sum()),
        "users_flagged": len(_state["risk_summary"]) if _state["risk_summary"] is not None else 0,
    }


@app.post("/retrain")
def retrain():
    df = _state["raw_df"]
    if df is None:
        raise HTTPException(400, "No data available to retrain on")
    _state["model"] = train_model(df, save=True)
    _state["scored_df"] = detect_anomalies(df, _state["model"])
    _state["risk_summary"] = aggregate_user_risk(_state["scored_df"])
    _state["reports_cache"] = {}
    return {"message": "Model retrained", "events": len(df)}
