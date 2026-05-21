"""
incident_summarizer.py
Generates AI-powered incident summaries from anomaly detection results.
Supports Anthropic Claude (default) or OpenAI as the LLM backend.

Usage:
    summarizer = IncidentSummarizer(backend="claude")
    report = summarizer.generate(user_risk_row, flagged_events)
    print(report.to_text())
"""

import os
import json
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class IncidentReport:
    user: str
    risk_level: str
    anomaly_score: float
    alert_title: str
    reasons: list[str]
    ai_summary: str
    recommended_actions: list[str]
    generated_at: str

    def to_text(self) -> str:
        reasons_text = "\n".join(f"  - {r}" for r in self.reasons)
        actions_text = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.recommended_actions))
        return f"""
{'='*60}
INCIDENT REPORT — {self.generated_at}
{'='*60}
Threat Level : {self.risk_level}
User         : {self.user}
Anomaly Score: {self.anomaly_score:.2f} (Isolation Forest)

Alert:
  {self.alert_title}

Detection Reasons:
{reasons_text}

AI Summary:
  {self.ai_summary}

Recommended Actions:
{actions_text}
{'='*60}
"""

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "risk_level": self.risk_level,
            "anomaly_score": self.anomaly_score,
            "alert_title": self.alert_title,
            "reasons": self.reasons,
            "ai_summary": self.ai_summary,
            "recommended_actions": self.recommended_actions,
            "generated_at": self.generated_at,
        }


class IncidentSummarizer:
    """
    Wraps an LLM to produce incident summaries from flagged event data.
    Backends: 'claude' (Anthropic) or 'openai'.
    Set API keys via environment variables:
      ANTHROPIC_API_KEY  for Claude
      OPENAI_API_KEY     for OpenAI
    """

    SYSTEM_PROMPT = """You are a senior SOC (Security Operations Center) analyst.
Your job is to produce concise, accurate, and actionable incident summaries
based on authentication log anomalies. Write in plain English for a technical audience.
Avoid excessive jargon. Be direct about the threat level and likely attack vector.
Output ONLY valid JSON matching the specified schema."""

    def __init__(self, backend: str = "claude", model: str = None):
        self.backend = backend.lower()
        if self.backend == "claude":
            self.model = model or "claude-opus-4-20250514"
        elif self.backend == "openai":
            self.model = model or "gpt-4o"
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose 'claude' or 'openai'.")

    def _build_prompt(self, user: str, risk_level: str, score: float, events: pd.DataFrame) -> str:
        event_summary = []
        for _, row in events.head(20).iterrows():
            event_summary.append({
                "timestamp": str(row.get("timestamp", "")),
                "event_type": row.get("event_type", ""),
                "country": row.get("country", ""),
                "failed_attempts": int(row.get("failed_attempts", 0)),
                "files_accessed": int(row.get("files_accessed", 0)),
                "api_calls": int(row.get("api_calls", 0)),
                "is_foreign": bool(row.get("is_foreign_login", False)),
                "is_off_hours": bool(row.get("is_off_hours", False)),
                "is_tor": bool(row.get("is_tor_int", False)),
                "anomaly_score": float(row.get("anomaly_score", 0)),
                "anomaly_type": row.get("anomaly_injected", "unknown"),
            })

        return f"""Analyze the following anomalous authentication events for user '{user}'.
Risk level: {risk_level}
Max anomaly score: {score:.2f} (0-1 scale, higher = more anomalous)

Events (sorted by anomaly score):
{json.dumps(event_summary, indent=2)}

Return a JSON object with exactly these fields:
{{
  "alert_title": "one sentence summarizing the core threat",
  "reasons": ["list", "of", "3-5", "specific", "detection", "reasons"],
  "ai_summary": "2-3 sentence plain-English summary of likely threat and context",
  "recommended_actions": ["list", "of", "3-5", "concrete", "remediation", "steps"]
}}"""

    def _call_claude(self, prompt: str) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=self.model,
            max_tokens=800,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return json.loads(response.content[0].text)

    def _call_openai(self, prompt: str) -> dict:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
        )
        return json.loads(response.choices[0].message.content)

    def generate(self, user_risk_row: pd.Series, flagged_events: pd.DataFrame) -> IncidentReport:
        user = user_risk_row["user"]
        risk_level = user_risk_row["risk_level"]
        score = float(user_risk_row["max_anomaly_score"])

        user_events = flagged_events[flagged_events["user"] == user].sort_values(
            "anomaly_score", ascending=False
        )

        prompt = self._build_prompt(user, risk_level, score, user_events)

        try:
            if self.backend == "claude":
                result = self._call_claude(prompt)
            else:
                result = self._call_openai(prompt)
        except Exception as e:
            result = {
                "alert_title": f"Anomalous activity detected for {user} (LLM unavailable: {e})",
                "reasons": [
                    f"Anomaly score: {score:.2f}",
                    f"Events flagged: {len(user_events)}",
                    f"Countries: {user_risk_row.get('countries', [])}",
                ],
                "ai_summary": (
                    "Automated detection flagged this user based on Isolation Forest scoring. "
                    "LLM summary unavailable — manual analyst review required."
                ),
                "recommended_actions": [
                    "Review all flagged events in the log feed",
                    "Check user's recent device and location history",
                    "Consider temporary account suspension pending review",
                ],
            }

        return IncidentReport(
            user=user,
            risk_level=risk_level,
            anomaly_score=score,
            alert_title=result.get("alert_title", ""),
            reasons=result.get("reasons", []),
            ai_summary=result.get("ai_summary", ""),
            recommended_actions=result.get("recommended_actions", []),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    def generate_batch(self, risk_summary: pd.DataFrame, flagged_events: pd.DataFrame) -> list[IncidentReport]:
        reports = []
        for _, row in risk_summary.iterrows():
            report = self.generate(row, flagged_events)
            print(report.to_text())
            reports.append(report)
        return reports


if __name__ == "__main__":
    # Demo with a mock risk row (no API key needed for the fallback path)
    mock_row = pd.Series({
        "user": "jsmith",
        "risk_level": "HIGH",
        "max_anomaly_score": 0.94,
        "mean_anomaly_score": 0.81,
        "event_count": 17,
        "countries": ["United States", "Germany"],
        "anomaly_types": ["impossible_travel_origin", "brute_force"],
    })
    mock_events = pd.DataFrame([
        {"user": "jsmith", "timestamp": "2025-01-01 09:02:00", "event_type": "LOGIN_SUCCESS",
         "country": "United States", "failed_attempts": 0, "files_accessed": 2, "api_calls": 5,
         "is_foreign_login": 0, "is_off_hours": 0, "is_tor_int": 0, "anomaly_score": 0.91,
         "anomaly_injected": "impossible_travel_origin"},
        {"user": "jsmith", "timestamp": "2025-01-01 10:14:00", "event_type": "LOGIN_SUCCESS",
         "country": "Germany", "failed_attempts": 14, "files_accessed": 3, "api_calls": 2,
         "is_foreign_login": 1, "is_off_hours": 0, "is_tor_int": 0, "anomaly_score": 0.94,
         "anomaly_injected": "impossible_travel_dest"},
    ])

    summarizer = IncidentSummarizer(backend="claude")
    report = summarizer.generate(mock_row, mock_events)
    print(report.to_text())
