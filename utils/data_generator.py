"""
data_generator.py - Generates synthetic UAV mission training data
Physics-informed dataset for training ML and Deep Learning models
"""

import numpy as np
import pandas as pd
from utils.config import FEATURE_NAMES, SEQ_LEN


# ─────────────────────────────────────────────
# TABULAR DATA (for ML Ensemble + Deep MLP)
# ─────────────────────────────────────────────
def generate_tabular_data(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate physics-informed UAV mission dataset.
    Each row = one mission with 10 features + risk label.
    
    Risk factors modelled from real UAV operation guidelines:
    - DGCA India regulations
    - Wind structural stress limits
    - Battery safety margins
    - Signal loss thresholds
    """
    rng = np.random.default_rng(seed)

    # ── Generate raw features ─────────────────
    distance     = rng.uniform(1,    150, n_samples)
    battery      = rng.uniform(10,   100, n_samples)
    wind         = rng.uniform(0.5,  25,  n_samples)
    payload      = rng.uniform(0,    5,   n_samples)
    signal       = rng.uniform(10,   100, n_samples)
    temp         = rng.uniform(-5,   45,  n_samples)
    altitude     = rng.uniform(10,   400, n_samples)
    humidity     = rng.uniform(20,   95,  n_samples)

    # ── Derived features ──────────────────────
    drain_rate   = (0.4
                    * (1 + payload * 0.08)
                    * (1 + np.maximum(0, wind - 5) * 0.015))
    flight_min   = (distance / 60.0) * 60
    batt_needed  = distance * drain_rate

    # ── Risk flags ────────────────────────────
    f_range      = distance   > 80
    f_battery    = battery    < 25
    f_batt_fit   = batt_needed > battery * 0.85
    f_wind_high  = wind       > 12
    f_wind_sev   = wind       > 16
    f_payload    = payload    > 3.5
    f_signal     = signal     < 30
    f_temp_hot   = temp       > 38
    f_temp_cold  = temp       < 0
    f_altitude   = altitude   > 350
    f_humid      = humidity   > 85

    # ── Composite risk score ──────────────────
    score = (
        f_range.astype(float)     * 1.5 +
        f_battery.astype(float)   * 2.0 +
        f_batt_fit.astype(float)  * 1.8 +
        f_wind_high.astype(float) * 1.5 +
        f_wind_sev.astype(float)  * 2.5 +
        f_payload.astype(float)   * 0.8 +
        f_signal.astype(float)    * 1.8 +
        f_temp_hot.astype(float)  * 0.7 +
        f_temp_cold.astype(float) * 0.7 +
        f_altitude.astype(float)  * 0.5 +
        f_humid.astype(float)     * 0.4
    )

    # ── Risk label ────────────────────────────
    critical = (
        f_wind_sev |
        (f_battery & f_range) |
        (f_signal  & f_range)
    )
    risk = ((score >= 2.5) | critical).astype(int)

    # Add 3% label noise for realism
    noise_idx = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)
    risk[noise_idx] = 1 - risk[noise_idx]

    return pd.DataFrame({
        "distance_km":          distance,
        "battery_pct":          battery,
        "wind_speed_mps":       wind,
        "payload_kg":           payload,
        "signal_strength_pct":  signal,
        "temperature_c":        temp,
        "altitude_m":           altitude,
        "humidity_pct":         humidity,
        "flight_time_min":      flight_min,
        "battery_drain_rate":   drain_rate,
        "risk":                 risk,
    })


# ─────────────────────────────────────────────
# SEQUENTIAL DATA (for LSTM + Transformer)
# ─────────────────────────────────────────────
def generate_sequential_data(n_routes: int = 3000,
                              seq_len:  int = SEQ_LEN,
                              seed:     int = 7):
    """
    Generate sequential flight route data.
    Each route has seq_len waypoints with telemetry drifting
    realistically across the route.

    Returns:
        X       : shape (n_routes, seq_len, n_features)
        y_step  : shape (n_routes, seq_len)   per-waypoint risk
        y_route : shape (n_routes,)            whole-route risk
    """
    rng    = np.random.default_rng(seed)
    n_feat = len(FEATURE_NAMES)

    X       = np.zeros((n_routes, seq_len, n_feat), dtype=np.float32)
    y_step  = np.zeros((n_routes, seq_len),          dtype=np.float32)

    for r in range(n_routes):
        # Initial mission parameters
        total_dist  = rng.uniform(20,  150)
        init_batt   = rng.uniform(60,  100)
        payload     = rng.uniform(0,   5)
        init_signal = rng.uniform(50,  100)
        base_wind   = rng.uniform(1,   14)
        base_temp   = rng.uniform(15,  40)
        altitude    = rng.uniform(50,  300)
        humidity    = rng.uniform(30,  90)

        # Wind drift across waypoints
        wind_drift = rng.normal(0, 1.5, seq_len).cumsum() * 0.4

        batt   = init_batt
        signal = init_signal

        for t in range(seq_len):
            frac     = t / (seq_len - 1)
            drain    = (0.4
                        * (1 + payload * 0.08)
                        * (1 + max(0, base_wind - 5) * 0.015))
            seg_dist = total_dist / (seq_len - 1)

            # Update telemetry
            batt   = max(0, batt - drain * seg_dist)
            signal = max(10, signal - 1.3 - float(rng.normal(0, 1.5)))
            wind   = max(0.3, base_wind + wind_drift[t] + float(rng.normal(0, 0.8)))
            temp   = base_temp + float(rng.normal(0, 1.2))

            X[r, t] = [
                total_dist, batt, wind, payload, signal,
                temp, altitude, humidity,
                frac * (total_dist / 60 * 60),
                drain,
            ]

            # Per-step risk score
            step_score = (
                (wind  > 12) * 1.5 +
                (wind  > 16) * 2.0 +
                (batt  < 25) * 2.0 +
                (batt  < 15) * 1.5 +
                (signal < 30) * 1.8 +
                (total_dist > 80) * 0.8
            )
            y_step[r, t] = 1.0 if step_score >= 2.2 else 0.0

    y_route = (y_step.max(axis=1) > 0).astype(np.float32)
    return X, y_step, y_route