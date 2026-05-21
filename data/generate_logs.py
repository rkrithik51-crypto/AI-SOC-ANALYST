"""
generate_logs.py
Generates realistic synthetic authentication and firewall logs for SOC training/testing.
Includes seeded anomalies: brute force, impossible travel, off-hours access, data exfiltration.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

random.seed(42)
np.random.seed(42)

USERS = [
    "jsmith", "mchen", "rwilson", "aparker", "ltorrez",
    "bkumar", "slee", "dmorris", "alopez", "fhansen"
]

COUNTRIES = {
    "United States": ("US", "10.0.{}.{}"),
    "Germany":       ("DE", "82.113.{}.{}"),
    "Canada":        ("CA", "192.168.{}.{}"),
    "Philippines":   ("PH", "112.198.{}.{}"),
    "Brazil":        ("BR", "177.71.{}.{}"),
    "Russia":        ("RU", "95.142.{}.{}"),
}

NORMAL_COUNTRY = "United States"

DEVICES = ["Windows10/Chrome", "macOS/Safari", "iPhone/Mobile", "Windows11/Edge", "Linux/Firefox"]


def random_ip(country: str) -> str:
    template = COUNTRIES[country][1]
    return template.format(random.randint(1, 254), random.randint(1, 254))


def business_hours_ts(base_date: datetime) -> datetime:
    hour = random.randint(8, 17)
    minute = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))


def off_hours_ts(base_date: datetime) -> datetime:
    hour = random.choice(list(range(0, 7)) + list(range(21, 24)))
    return base_date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def make_normal_event(user: str, ts: datetime) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "user": user,
        "event_type": random.choice(["LOGIN_SUCCESS", "FILE_ACCESS", "API_CALL"]),
        "country": NORMAL_COUNTRY,
        "ip": random_ip(NORMAL_COUNTRY),
        "device": random.choice(DEVICES),
        "failed_attempts": 0,
        "files_accessed": random.randint(1, 8),
        "api_calls": random.randint(1, 40),
        "is_vpn": False,
        "is_tor": False,
        "label": "normal",
        "anomaly_injected": None,
    }


def inject_brute_force(user: str, base_ts: datetime) -> list:
    events = []
    for i in range(random.randint(8, 20)):
        ts = base_ts + timedelta(seconds=i * random.randint(15, 45))
        ev = make_normal_event(user, ts)
        ev["event_type"] = "LOGIN_FAILED"
        ev["failed_attempts"] = i + 1
        ev["label"] = "anomaly"
        ev["anomaly_injected"] = "brute_force"
        events.append(ev)
    success = make_normal_event(user, base_ts + timedelta(minutes=random.randint(5, 12)))
    success["event_type"] = "LOGIN_SUCCESS"
    success["label"] = "anomaly"
    success["anomaly_injected"] = "brute_force_success"
    events.append(success)
    return events


def inject_impossible_travel(user: str, base_ts: datetime) -> list:
    country_b = random.choice([c for c in COUNTRIES if c != NORMAL_COUNTRY])
    gap_minutes = random.randint(40, 90)

    ev1 = make_normal_event(user, base_ts)
    ev1["event_type"] = "LOGIN_SUCCESS"
    ev1["label"] = "anomaly"
    ev1["anomaly_injected"] = "impossible_travel_origin"

    ev2 = make_normal_event(user, base_ts + timedelta(minutes=gap_minutes))
    ev2["country"] = country_b
    ev2["ip"] = random_ip(country_b)
    ev2["event_type"] = "LOGIN_SUCCESS"
    ev2["label"] = "anomaly"
    ev2["anomaly_injected"] = "impossible_travel_dest"
    return [ev1, ev2]


def inject_data_exfil(user: str, base_ts: datetime) -> list:
    events = []
    for i in range(random.randint(30, 60)):
        ts = base_ts + timedelta(seconds=i * 3)
        ev = make_normal_event(user, ts)
        ev["event_type"] = "FILE_ACCESS"
        ev["files_accessed"] = random.randint(5, 12)
        ev["label"] = "anomaly"
        ev["anomaly_injected"] = "data_exfiltration"
        events.append(ev)
    return events


def inject_off_hours(user: str, base_ts: datetime) -> list:
    ts = off_hours_ts(base_ts)
    ev = make_normal_event(user, ts)
    ev["event_type"] = "LOGIN_SUCCESS"
    ev["country"] = random.choice([c for c in COUNTRIES if c != NORMAL_COUNTRY])
    ev["ip"] = random_ip(ev["country"])
    ev["is_vpn"] = random.random() > 0.5
    ev["is_tor"] = random.random() > 0.7
    ev["label"] = "anomaly"
    ev["anomaly_injected"] = "off_hours_foreign_login"
    return [ev]


def inject_api_spike(user: str, base_ts: datetime) -> list:
    events = []
    for i in range(random.randint(15, 25)):
        ts = base_ts + timedelta(seconds=i * 20)
        ev = make_normal_event(user, ts)
        ev["event_type"] = "API_CALL"
        ev["api_calls"] = random.randint(180, 400)
        ev["label"] = "anomaly"
        ev["anomaly_injected"] = "api_spike"
        events.append(ev)
    return events


def generate_logs(
    n_normal: int = 2000,
    n_anomalies: int = 5,
    output_path: str = "sample_logs/auth_logs.csv"
) -> pd.DataFrame:
    base_date = datetime.now() - timedelta(hours=24)
    events = []

    # Normal traffic
    for _ in range(n_normal):
        user = random.choice(USERS)
        ts = base_date + timedelta(
            hours=random.uniform(0, 24),
            minutes=random.uniform(0, 60)
        )
        events.append(make_normal_event(user, ts))

    # Inject anomalies
    anomaly_funcs = [
        inject_brute_force,
        inject_impossible_travel,
        inject_data_exfil,
        inject_off_hours,
        inject_api_spike,
    ]

    for i, func in enumerate(anomaly_funcs[:n_anomalies]):
        user = USERS[i % len(USERS)]
        ts = base_date + timedelta(hours=random.uniform(1, 22))
        injected = func(user, ts)
        events.extend(injected)

    df = pd.DataFrame(events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} log events -> {output_path}")
    print(f"  Normal:  {(df['label']=='normal').sum()}")
    print(f"  Anomaly: {(df['label']=='anomaly').sum()}")
    return df


if __name__ == "__main__":
    generate_logs()
