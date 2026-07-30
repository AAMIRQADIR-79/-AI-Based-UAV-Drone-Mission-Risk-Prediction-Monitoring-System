"""
alerts.py - Alert engine, battery model, SITREP generator
"""

import time
import math
from dataclasses import dataclass, field
from typing import List
from utils.config import UAV_SPECS


# ─────────────────────────────────────────────
# ALERT DATA CLASS
# ─────────────────────────────────────────────
@dataclass
class Alert:
    level:    str    # critical / warning / caution / info
    code:     str
    category: str
    message:  str
    action:   str = ""


# ─────────────────────────────────────────────
# EVALUATE ALL ALERTS
# ─────────────────────────────────────────────
def evaluate_alerts(
    battery_pct:  float,
    wind_speed:   float,
    signal_pct:   float,
    payload_kg:   float,
    distance_km:  float,
    temperature:  float,
    risk_score:   float,
    altitude_m:   float = 100,
    humidity_pct: float = 60,
) -> List[Alert]:
    """
    Evaluate all UAV risk conditions and return sorted alert list.
    Order: critical → warning → caution → info
    """
    alerts = []

    # ── BATTERY ──────────────────────────────
    if battery_pct < UAV_SPECS["critical_battery_pct"]:
        alerts.append(Alert(
            level="critical", code="BATT_CRITICAL", category="battery",
            message=f"BATTERY CRITICAL: {battery_pct:.1f}% — Emergency RTH required!",
            action="Initiate Return-To-Home immediately",
        ))
    elif battery_pct < UAV_SPECS["warning_battery_pct"]:
        alerts.append(Alert(
            level="warning", code="BATT_LOW", category="battery",
            message=f"Battery low: {battery_pct:.1f}% — Plan landing soon",
            action="Begin return journey within 2 minutes",
        ))
    elif battery_pct < 40:
        alerts.append(Alert(
            level="caution", code="BATT_MEDIUM", category="battery",
            message=f"Battery at {battery_pct:.1f}% — Monitor drain rate",
            action="Monitor battery consumption",
        ))

    # ── WIND ─────────────────────────────────
    if wind_speed > UAV_SPECS["max_wind_safe_mps"]:
        alerts.append(Alert(
            level="critical", code="WIND_CRITICAL", category="wind",
            message=f"WIND CRITICAL: {wind_speed:.1f} m/s — Exceeds safe limit!",
            action="Abort mission — structural stress risk",
        ))
    elif wind_speed > 12:
        alerts.append(Alert(
            level="warning", code="WIND_HIGH", category="wind",
            message=f"High wind: {wind_speed:.1f} m/s — Stability affected",
            action="Reduce altitude to 80m, fly into wind",
        ))
    elif wind_speed > 8:
        alerts.append(Alert(
            level="caution", code="WIND_MOD", category="wind",
            message=f"Moderate wind: {wind_speed:.1f} m/s — Monitor attitude",
            action="Check stabilization system",
        ))

    # ── SIGNAL ───────────────────────────────
    if signal_pct < UAV_SPECS["signal_min_pct"]:
        alerts.append(Alert(
            level="critical", code="SIGNAL_CRITICAL", category="signal",
            message=f"SIGNAL CRITICAL: {signal_pct:.0f}% — Communication loss risk!",
            action="Return to last known good position",
        ))
    elif signal_pct < UAV_SPECS["signal_warn_pct"]:
        alerts.append(Alert(
            level="warning", code="SIGNAL_WEAK", category="signal",
            message=f"Weak signal: {signal_pct:.0f}% — Telemetry unreliable",
            action="Move to higher elevation or reduce range",
        ))
    elif signal_pct < 50:
        alerts.append(Alert(
            level="caution", code="SIGNAL_FAIR", category="signal",
            message=f"Signal fair: {signal_pct:.0f}% — Monitor link quality",
        ))

    # ── RANGE ────────────────────────────────
    if distance_km > UAV_SPECS["max_range_km"]:
        alerts.append(Alert(
            level="critical", code="RANGE_EXCEED", category="range",
            message=f"RANGE EXCEEDED: {distance_km:.0f} km — Beyond max range!",
            action="Select closer destination",
        ))
    elif distance_km > 100:
        alerts.append(Alert(
            level="warning", code="RANGE_HIGH", category="range",
            message=f"Long range mission: {distance_km:.0f} km — Battery critical",
            action="Plan battery checkpoint at midpoint",
        ))
    elif distance_km > 60:
        alerts.append(Alert(
            level="caution", code="RANGE_MED", category="range",
            message=f"Extended range: {distance_km:.0f} km — Monitor battery",
        ))

    # ── TEMPERATURE ──────────────────────────
    if temperature > UAV_SPECS["temp_max_c"]:
        alerts.append(Alert(
            level="warning", code="TEMP_HIGH", category="temperature",
            message=f"High temperature: {temperature:.1f}°C — Battery capacity reduced",
            action="Pre-cool battery, expect 15-20% capacity loss",
        ))
    elif temperature < UAV_SPECS["temp_min_c"]:
        alerts.append(Alert(
            level="warning", code="TEMP_LOW", category="temperature",
            message=f"Low temperature: {temperature:.1f}°C — Battery efficiency reduced",
            action="Warm battery to 15°C before flight",
        ))

    # ── PAYLOAD ──────────────────────────────
    if payload_kg > UAV_SPECS["max_payload_kg"]:
        alerts.append(Alert(
            level="critical", code="PAYLOAD_EXCEED", category="payload",
            message=f"PAYLOAD EXCEEDED: {payload_kg:.1f} kg — Over maximum limit!",
            action="Reduce payload before flight",
        ))
    elif payload_kg > 3.5:
        alerts.append(Alert(
            level="warning", code="PAYLOAD_HIGH", category="payload",
            message=f"Heavy payload: {payload_kg:.1f} kg — Maneuverability reduced",
            action="Reduce payload or reduce distance",
        ))

    # ── ALTITUDE ─────────────────────────────
    if altitude_m > UAV_SPECS["max_altitude_m"]:
        alerts.append(Alert(
            level="critical", code="ALT_EXCEED", category="altitude",
            message=f"ALTITUDE EXCEEDED: {altitude_m:.0f}m — DGCA violation!",
            action="Descend immediately — regulatory violation",
        ))

    # ── HUMIDITY ─────────────────────────────
    if humidity_pct > 90:
        alerts.append(Alert(
            level="caution", code="HUMIDITY_HIGH", category="environment",
            message=f"High humidity: {humidity_pct:.0f}% — Moisture risk to electronics",
            action="Check weatherproofing before flight",
        ))

    # ── AI RISK ──────────────────────────────
    if risk_score >= 80:
        alerts.append(Alert(
            level="critical", code="AI_CRITICAL", category="ai",
            message=f"AI RISK CRITICAL: {risk_score:.1f}% — Mission abort recommended",
            action="Review all risk factors before proceeding",
        ))
    elif risk_score >= 60:
        alerts.append(Alert(
            level="warning", code="AI_HIGH", category="ai",
            message=f"AI HIGH RISK: {risk_score:.1f}% — Proceed with extreme caution",
        ))
    elif risk_score >= 35:
        alerts.append(Alert(
            level="caution", code="AI_MODERATE", category="ai",
            message=f"AI MODERATE RISK: {risk_score:.1f}% — Monitor conditions",
        ))

    # Sort by priority
    priority = {"critical": 0, "warning": 1, "caution": 2, "info": 3}
    alerts.sort(key=lambda a: priority[a.level])
    return alerts


# ─────────────────────────────────────────────
# BATTERY MODEL
# ─────────────────────────────────────────────
def battery_drain_per_km(
    payload_kg:    float,
    wind_speed:    float,
    altitude_m:    float = 100,
    temperature_c: float = 25,
) -> float:
    """
    Physics-based battery drain model (% per km).
    Accounts for payload, wind, altitude, temperature.
    """
    base        = 0.40
    payload_f   = 1.0 + payload_kg   * 0.082
    wind_f      = 1.0 + max(0, wind_speed - 4) * 0.018
    altitude_f  = 1.0 + max(0, altitude_m - 100) * 0.0005
    temp_f      = 1.0 + max(0, 15 - temperature_c) * 0.008

    return round(base * payload_f * wind_f * altitude_f * temp_f, 5)


def estimate_remaining_range(
    battery_pct:  float,
    drain_per_km: float,
    reserve_pct:  float = 15,
) -> float:
    """Estimate remaining flyable distance in km."""
    usable = max(0, battery_pct - reserve_pct)
    return round(usable / drain_per_km, 1) if drain_per_km > 0 else 0


# ─────────────────────────────────────────────
# RISK COLOR AND LABEL
# ─────────────────────────────────────────────
def risk_color(score: float) -> str:
    if score >= 80: return "#FF0000"
    if score >= 60: return "#FF6600"
    if score >= 35: return "#FFAA00"
    return "#00CC00"


def risk_level(score: float) -> str:
    if score >= 80: return "Critical"
    if score >= 60: return "High Risk"
    if score >= 35: return "Moderate"
    return "Safe"


# ─────────────────────────────────────────────
# MISSION READINESS SCORE
# ─────────────────────────────────────────────
def compute_readiness(
    risk_score:      float,
    alerts:          list,
    nfz_conflicts:   int = 0,
    weather_risk:    float = 0,
) -> dict:
    """
    Compute overall mission readiness score (0-100).
    Higher = safer to fly.
    """
    deductions = risk_score * 0.4

    for a in alerts:
        if a.category != "ai":
            deductions += {
                "critical": 20,
                "warning":  10,
                "caution":   5,
                "info":      1,
            }.get(a.level, 0)

    deductions += nfz_conflicts * 8
    deductions += weather_risk  * 0.3

    readiness = max(0, min(100, 100 - deductions))

    if readiness >= 75:
        verdict = "✅ MISSION APPROVED"
        color   = "#00CC00"
    elif readiness >= 50:
        verdict = "⚠️ PROCEED WITH CAUTION"
        color   = "#FFAA00"
    elif readiness >= 25:
        verdict = "🔴 HIGH RISK — REVIEW REQUIRED"
        color   = "#FF6600"
    else:
        verdict = "🚨 MISSION ABORT RECOMMENDED"
        color   = "#FF0000"

    return {
        "readiness_score":  round(readiness, 1),
        "verdict":          verdict,
        "color":            color,
        "critical_count":   sum(1 for a in alerts if a.level == "critical"),
        "warning_count":    sum(1 for a in alerts if a.level == "warning"),
    }


# ─────────────────────────────────────────────
# SITREP GENERATOR
# ─────────────────────────────────────────────
def generate_sitrep(
    mission_id:   str,
    source:       str,
    destination:  str,
    distance_km:  float,
    bearing:      float,
    compass:      str,
    risk_score:   float,
    risk_level_s: str,
    alerts:       list,
    nfz_conflicts:list,
    weather:      dict,
    readiness:    dict,
) -> str:
    """
    Generate military-style Situation Report (SITREP).
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    critical  = [a for a in alerts if a.level == "critical"]
    warnings  = [a for a in alerts if a.level == "warning"]

    lines = [
        f"SITREP // {mission_id} // {timestamp}",
        "=" * 55,
        f"ROUTE    : {source.upper()} → {destination.upper()}",
        f"DISTANCE : {distance_km:.1f} KM | BEARING: {bearing:.0f}° {compass}",
        f"WEATHER  : {weather['temperature']}°C | WIND: {weather['wind_speed']} M/S {weather['wind_compass']} | {weather['description'].upper()}",
        f"AI RISK  : {risk_score:.1f}% — {risk_level_s.upper()}",
        f"READINESS: {readiness['readiness_score']:.0f}/100 — {readiness['verdict']}",
        "-" * 55,
    ]

    if nfz_conflicts:
        lines.append(f"NFZ      : {len(nfz_conflicts)} CONFLICT(S) DETECTED — REROUTE REQUIRED")
    else:
        lines.append("NFZ      : NO CONFLICTS DETECTED — ROUTE CLEAR")

    if critical:
        lines.append(f"CRITICAL : {len(critical)} CRITICAL ALERT(S)")
        for a in critical[:3]:
            lines.append(f"           ⚠ {a.message}")
    else:
        lines.append("CRITICAL : NO CRITICAL ALERTS")

    if warnings:
        lines.append(f"WARNINGS : {len(warnings)} WARNING(S)")
        for a in warnings[:3]:
            lines.append(f"           ! {a.message}")

    lines.append("-" * 55)

    if readiness["readiness_score"] >= 75:
        lines.append("ACTION   : CLEARED FOR LAUNCH")
    elif readiness["readiness_score"] >= 50:
        lines.append("ACTION   : RESOLVE WARNINGS BEFORE LAUNCH")
    else:
        lines.append("ACTION   : DO NOT LAUNCH — ABORT MISSION")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# REGULATORY COMPLIANCE CHECK
# ─────────────────────────────────────────────
def check_compliance(features: dict) -> list:
    """Check DGCA India UAV regulations."""
    violations = []

    if features.get("altitude_m", 0) > 400:
        violations.append({
            "rule":     "DGCA Rule 19",
            "desc":     "Max altitude 400m AGL without ATC permission",
            "severity": "violation",
        })
    if features.get("distance_km", 0) > 450:
        violations.append({
            "rule":     "DGCA BVLOS",
            "desc":     "Beyond Visual Line of Sight requires special permit",
            "severity": "violation",
        })
    if features.get("payload_kg", 0) > 5:
        violations.append({
            "rule":     "DGCA Weight Class",
            "desc":     "Payload exceeds small UAS category limit of 5kg",
            "severity": "violation",
        })
    if features.get("wind_speed_mps", 0) > 15:
        violations.append({
            "rule":     "Weather Minima",
            "desc":     "Wind speed exceeds standard operating limits",
            "severity": "warning",
        })

    return violations