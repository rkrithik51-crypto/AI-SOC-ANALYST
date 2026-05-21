# AI-SOC-ANALYST

Automated security log analysis, anomaly detection, and AI-powered incident reporting. Built to mirror real SOC tooling.

## Architecture

```
soc-analyst/
├── data/
│   ├── generate_logs.py        # Synthetic log generator (brute force, impossible travel, exfil, etc.)
│   └── sample_logs/            # Generated CSVs land here
├── models/
│   ├── anomaly_detector.py     # Isolation Forest pipeline (train + score + aggregate)
│   └── isolation_forest.pkl    # Saved model (auto-generated on first run)
├── api/
│   └── main.py                 # FastAPI backend (ingest, score, alert, report)
├── utils/
│   └── incident_summarizer.py  # LLM-powered incident reports (Claude or OpenAI)
├── dashboard/
│   └── dashboard.py            # Streamlit UI
└── requirements.txt
```

## Setup

```bash
git clone <repo>
cd soc-analyst
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your LLM key (for AI summaries):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
export LLM_BACKEND=openai       # defaults to claude
```

## Run

Terminal 1 — API backend:
```bash
cd soc-analyst
uvicorn api.main:app --reload --port 8000
```
On first launch, this auto-generates synthetic logs and trains the Isolation Forest model.

Terminal 2 — Streamlit dashboard:
```bash
cd soc-analyst
streamlit run dashboard/dashboard.py
```

Open http://localhost:8501

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /stats | Dashboard summary counts |
| GET | /alerts | All flagged users (filter by score / level) |
| GET | /alerts/{user} | Per-user incident detail + optional AI summary |
| GET | /logs | Log feed (filter by user, event type, anomaly) |
| POST | /ingest | Upload a new CSV log file |
| POST | /retrain | Retrain the Isolation Forest on current data |
| GET | /health | Health check |

## Detection Logic

The Isolation Forest model scores each log event across these features:

| Feature | Description |
|---------|-------------|
| failed_attempts | Login failure count in event window |
| api_calls | API request volume |
| files_accessed | File access count |
| is_foreign_login | Login from non-home country (binary) |
| is_off_hours | Login outside 08:00-18:00 (binary) |
| is_vpn_int | VPN connection detected |
| is_tor_int | TOR exit node detected |
| login_hour | Hour of day (0-23) |
| day_of_week | Day (0=Monday) |
| event_type_enc | LOGIN_FAILED=2, LOGIN_SUCCESS=1, other=0 |

Scores are normalized to [0, 1]. Events scoring above 0.65 are flagged. Per-user risk is aggregated with thresholds:

- CRITICAL >= 0.85
- HIGH >= 0.75
- MEDIUM >= 0.65

## Injected Anomaly Types

The log generator seeds five attack patterns:

1. Brute force — burst of failed logins followed by success
2. Impossible travel — successful logins from two distant countries within 90 min
3. Data exfiltration — abnormally high file access rate sustained over minutes
4. Off-hours foreign login — 2 AM login from a new country, often via TOR
5. API spike — automated burst targeting admin endpoints

## Using Real Logs

The API accepts any CSV with these required columns:

```
timestamp, user, event_type, country, ip, device,
failed_attempts, files_accessed, api_calls, is_vpn, is_tor
```

Compatible sources: Splunk CSV exports, Elastic SIEM exports, Auth0 logs, AWS CloudTrail (with minor transformation), Okta System Log exports, any Kaggle cybersecurity dataset with auth events.

## Extending

- Swap Isolation Forest for LOF or AutoEncoder in `anomaly_detector.py` — the scoring interface is the same.
- Add geographic distance calculation to `engineer_features()` for more precise impossible-travel detection.
- Connect a real-time log shipper (Filebeat, Fluentd) to the `/ingest` endpoint for live streaming.
- Add Slack/PagerDuty webhook calls in `incident_summarizer.py` to auto-alert on CRITICAL findings.

