"""
dashboard.py — Streamlit SOC Dashboard
Connects to the FastAPI backend and renders:
  - Live alert feed
  - Anomaly score distribution
  - User risk heat map
  - Per-user incident report viewer

Run with:
  streamlit run dashboard/dashboard.py
(Requires the FastAPI backend running at localhost:8000)
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="SOC Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; }
    .threat-high  { color: #E24B4A; font-weight: 600; }
    .threat-med   { color: #EF9F27; font-weight: 600; }
    .threat-low   { color: #639922; font-weight: 600; }
    .log-mono     { font-family: monospace; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------

with st.sidebar:
    st.title("🛡️ SOC Analyst")
    st.caption("Threat Detection & Incident Response")
    st.divider()

    uploaded = st.file_uploader("Upload log CSV", type=["csv"])
    if uploaded:
        resp = requests.post(f"{API_BASE}/ingest", files={"file": uploaded})
        if resp.ok:
            st.success(f"Ingested: {resp.json()['total_events']} events")
        else:
            st.error(f"Ingest failed: {resp.text}")

    if st.button("Retrain Model"):
        resp = requests.post(f"{API_BASE}/retrain")
        if resp.ok:
            st.success("Model retrained")
        else:
            st.error(resp.text)

    min_score = st.slider("Min anomaly score", 0.0, 1.0, 0.65, 0.01)
    level_filter = st.selectbox("Risk level filter", ["ALL", "CRITICAL", "HIGH", "MEDIUM"])
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)

if auto_refresh:
    st.experimental_rerun()


# ---------- Stats row ----------

def safe_get(url, default=None):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.ok else default
    except Exception:
        return default


stats = safe_get(f"{API_BASE}/stats", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Events", f"{stats.get('total_events', '-'):,}")
c2.metric("Anomalies", stats.get("anomalies_detected", "-"), delta_color="inverse")
c3.metric("Users Flagged", stats.get("users_flagged", "-"))
c4.metric("High/Critical", stats.get("high_severity", "-"), delta_color="inverse")
c5.metric("Medium", stats.get("medium_severity", "-"), delta_color="inverse")
st.caption(f"Last ingested: {stats.get('last_ingested', 'N/A')}")
st.divider()


# ---------- Main layout ----------

col_left, col_right = st.columns([1.4, 1])

with col_left:
    st.subheader("Flagged Users — Risk Overview")
    params = {"min_score": min_score}
    if level_filter != "ALL":
        params["level"] = level_filter
    alerts = safe_get(f"{API_BASE}/alerts?" + "&".join(f"{k}={v}" for k, v in params.items()), [])

    if not alerts:
        st.info("No alerts match the current filters.")
    else:
        alert_df = pd.DataFrame(alerts)
        alert_df["countries"] = alert_df["countries"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        alert_df["anomaly_types"] = alert_df["anomaly_types"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

        def style_risk(val):
            colors = {"CRITICAL": "#FCEBEB", "HIGH": "#FAEEDA", "MEDIUM": "#EAF3DE", "LOW": "#E6F1FB"}
            return f"background-color: {colors.get(val, '')}"

        styled = alert_df[[
            "user", "risk_level", "max_anomaly_score", "event_count", "countries", "anomaly_types"
        ]].style.map(style_risk, subset=["risk_level"]).format({"max_anomaly_score": "{:.3f}"})
        st.dataframe(styled, use_container_width=True, height=260)

    # Anomaly score distribution
    st.subheader("Anomaly Score Distribution")
    logs = safe_get(f"{API_BASE}/logs?anomaly_only=true&limit=500", [])
    if logs:
        log_df = pd.DataFrame(logs)
        fig = px.histogram(
            log_df, x="anomaly_score", nbins=40,
            color="anomaly_flag",
            color_discrete_map={"CRITICAL": "#E24B4A", "HIGH": "#EF9F27", "MEDIUM": "#639922"},
            labels={"anomaly_score": "Anomaly Score", "anomaly_flag": "Flag"},
            height=260,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Incident Report Viewer")
    if alerts:
        user_list = [a["user"] for a in alerts]
        selected_user = st.selectbox("Select user", user_list)
        gen_ai = st.checkbox("Generate AI summary (requires API key)", value=False)

        if selected_user:
            detail = safe_get(f"{API_BASE}/alerts/{selected_user}?generate_ai={str(gen_ai).lower()}", None)
            if detail:
                risk = detail.get("risk_level", "UNKNOWN")
                score = detail.get("max_anomaly_score", 0)

                level_color = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                st.markdown(f"### {level_color.get(risk, '⚪')} {risk} — `{selected_user}`")
                st.progress(float(score), text=f"Anomaly score: {score:.3f}")

                events = detail.get("events", [])
                if events:
                    with st.expander(f"Flagged events ({len(events)})", expanded=True):
                        ev_df = pd.DataFrame(events)[["timestamp", "event_type", "country", "failed_attempts", "anomaly_score"]].head(10)
                        st.dataframe(ev_df, use_container_width=True)

                report = detail.get("incident_report", {})
                if report and not report.get("error"):
                    st.markdown("#### AI Incident Summary")
                    st.info(report.get("ai_summary", ""))

                    reasons = report.get("reasons", [])
                    if reasons:
                        st.markdown("**Detection reasons:**")
                        for r in reasons:
                            st.markdown(f"- {r}")

                    actions = report.get("recommended_actions", [])
                    if actions:
                        st.markdown("**Recommended actions:**")
                        for i, a in enumerate(actions, 1):
                            st.markdown(f"{i}. {a}")
                elif gen_ai and report.get("error"):
                    st.error(f"AI summary error: {report['error']}")


# ---------- Log feed ----------

st.divider()
st.subheader("Recent Log Feed")
user_filter_log = st.text_input("Filter by user (optional)", placeholder="e.g. jsmith")
anom_only = st.checkbox("Anomalous events only", value=True)

log_params = f"anomaly_only={str(anom_only).lower()}&limit=200"
if user_filter_log:
    log_params += f"&user={user_filter_log}"

feed = safe_get(f"{API_BASE}/logs?{log_params}", [])
if feed:
    feed_df = pd.DataFrame(feed)
    display_cols = [c for c in ["timestamp","user","event_type","country","failed_attempts","api_calls","anomaly_score","anomaly_flag"] if c in feed_df.columns]
    st.dataframe(
        feed_df[display_cols].head(100).style.format({"anomaly_score": "{:.3f}"}),
        use_container_width=True,
        height=300,
    )
else:
    st.info("No logs to display.")
