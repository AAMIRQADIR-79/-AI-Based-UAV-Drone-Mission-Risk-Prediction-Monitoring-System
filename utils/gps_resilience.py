"""
gps_resilience.py - GPS-Denial Resilience via Dead Reckoning

Simulates GPS signal degradation (jamming/spoofing/loss of lock) and
estimates the drone's position using dead reckoning - the same
principle ships and aircraft used for centuries before GPS existed:
if you know your last confirmed position, your speed, and your
heading, you can estimate where you are now.
"""

import math
import random


# ─────────────────────────────────────────────
# GPS QUALITY MODEL
# ─────────────────────────────────────────────
GPS_QUALITY_LEVELS = {
    "excellent": {"min_satellites": 9, "hdop_max": 1.0, "accuracy_m": 3},
    "good":      {"min_satellites": 6, "hdop_max": 2.0, "accuracy_m": 8},
    "degraded":  {"min_satellites": 4, "hdop_max": 5.0, "accuracy_m": 25},
    "poor":      {"min_satellites": 2, "hdop_max": 10.0, "accuracy_m": 80},
    "lost":      {"min_satellites": 0, "hdop_max": 99.0, "accuracy_m": None},
}

# Minimum satellites needed for a mathematically reliable GPS fix
MIN_RELIABLE_SATELLITES = 4


def classify_gps_quality(n_satellites: int) -> str:
    """Classify current GPS quality band from satellite count."""
    if n_satellites >= 9:
        return "excellent"
    elif n_satellites >= 6:
        return "good"
    elif n_satellites >= 4:
        return "degraded"
    elif n_satellites >= 1:
        return "poor"
    else:
        return "lost"


def is_gps_reliable(n_satellites: int) -> bool:
    """The real-world minimum is 4 satellites for a valid 3D fix."""
    return n_satellites >= MIN_RELIABLE_SATELLITES


# ─────────────────────────────────────────────
# SIMULATE A JAMMING / DEGRADATION EVENT
# ─────────────────────────────────────────────
def simulate_gps_event(step: int, total_steps: int, seed: int = None,
                        force_jam_at: int = None) -> dict:
    """
    Simulates GPS satellite count across a mission. By default, GPS is
    healthy throughout. If force_jam_at is set to a waypoint index,
    a jamming event begins there and gradually degrades satellite lock
    for the remaining waypoints (simulating an operator flying near a
    jamming source, e.g. a sensitive installation).
    """
    rng = random.Random(seed or step)

    if force_jam_at is not None and step >= force_jam_at:
        # Progressive degradation once jamming starts
        steps_into_jam = step - force_jam_at
        satellites = max(0, 9 - steps_into_jam * 2 - rng.randint(0, 1))
    else:
        # Normal healthy GPS with minor natural variation
        satellites = rng.randint(8, 12)

    hdop = round(1.0 + max(0, 9 - satellites) * 0.4 + rng.uniform(-0.1, 0.3), 2)
    quality = classify_gps_quality(satellites)

    return {
        "satellites": satellites,
        "hdop": hdop,
        "quality": quality,
        "reliable": is_gps_reliable(satellites),
    }


# ─────────────────────────────────────────────
# DEAD RECKONING POSITION ESTIMATE
# ─────────────────────────────────────────────
def dead_reckoning_estimate(
    last_known_lat: float, last_known_lon: float,
    heading_deg: float, speed_kmh: float,
    seconds_since_last_fix: float,
    steps_without_gps: int = 1,
) -> dict:
    """
    Estimate current position using last known GPS fix, heading, and
    speed - the same math used by inertial navigation systems (INS)
    when GPS is unavailable.

    Also computes a growing uncertainty radius - the longer you go
    without GPS, the less trustworthy the estimate becomes, because
    small heading/speed errors accumulate over time (this is the
    real, well-documented behavior of dead reckoning / INS drift).
    """
    distance_km = speed_kmh * (seconds_since_last_fix / 3600.0)

    heading_rad = math.radians(heading_deg)
    R = 6371.0  # Earth radius km

    lat1 = math.radians(last_known_lat)
    lon1 = math.radians(last_known_lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_km / R) +
        math.cos(lat1) * math.sin(distance_km / R) * math.cos(heading_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(heading_rad) * math.sin(distance_km / R) * math.cos(lat1),
        math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2)
    )

    estimated_lat = math.degrees(lat2)
    estimated_lon = math.degrees(lon2)

    # Uncertainty grows with time/steps without a GPS fix.
    # Real INS drift is typically 1-5% of distance traveled per unit
    # time without correction; we use a conservative growing model.
    base_uncertainty_m = 15
    drift_per_step_m = 35
    uncertainty_radius_m = base_uncertainty_m + drift_per_step_m * steps_without_gps

    return {
        "estimated_lat": round(estimated_lat, 6),
        "estimated_lon": round(estimated_lon, 6),
        "distance_traveled_km": round(distance_km, 3),
        "uncertainty_radius_m": round(uncertainty_radius_m, 0),
        "steps_without_gps": steps_without_gps,
        "method": "Dead Reckoning (heading + speed + last fix)",
    }


# ─────────────────────────────────────────────
# GPS-AWARE RISK ADJUSTMENT
# ─────────────────────────────────────────────
def gps_risk_penalty(gps_status: dict) -> float:
    """
    Additional risk points to add to the AI risk score based on
    current GPS reliability. Feeds directly into your existing
    risk/alert pipeline as an extra input.
    """
    penalties = {
        "excellent": 0,
        "good": 2,
        "degraded": 18,
        "poor": 40,
        "lost": 60,
    }
    return penalties.get(gps_status["quality"], 0)


def get_gps_alert(gps_status: dict, dr_estimate: dict = None) -> dict:
    """Generate an alert dict matching your existing alert style."""
    q = gps_status["quality"]

    if q == "excellent" or q == "good":
        return None

    if q == "degraded":
        return {
            "level": "warning",
            "message": (
                f"GPS INTEGRITY DEGRADED - {gps_status['satellites']} "
                f"satellites, HDOP {gps_status['hdop']} - position "
                f"accuracy reduced"
            ),
            "action": "Monitor closely, prepare for possible loss of fix",
        }
    elif q == "poor":
        msg = (
            f"GPS SIGNAL POOR - {gps_status['satellites']} satellites - "
            f"SWITCHING TO INERTIAL NAVIGATION"
        )
        if dr_estimate:
            msg += (
                f" - est. position uncertainty ±"
                f"{dr_estimate['uncertainty_radius_m']:.0f}m"
            )
        return {
            "level": "critical",
            "message": msg,
            "action": "Reduce speed, consider RTH while position still estimable",
        }
    else:  # lost
        return {
            "level": "critical",
            "message": (
                "GPS FIX LOST - 0 satellites - FULL DEAD RECKONING MODE - "
                f"uncertainty growing ±{dr_estimate['uncertainty_radius_m']:.0f}m"
                if dr_estimate else
                "GPS FIX LOST - 0 satellites - FULL DEAD RECKONING MODE"
            ),
            "action": "IMMEDIATE RTH RECOMMENDED - position confidence critical",
        }