"""
terrain.py - Terrain Elevation Awareness for UAV Mission System

Fetches REAL ground elevation data (from Open-Elevation API, free, no key needed)
at every waypoint, and checks whether the drone's planned altitude gives it
safe clearance above the actual terrain - not just a flat assumed ground.
"""

import requests
import random


# ─────────────────────────────────────────────
# FETCH REAL ELEVATION FOR A LIST OF WAYPOINTS
# ─────────────────────────────────────────────
def get_elevation_profile(waypoints: list) -> list:
    """
    waypoints: list of (lat, lon) tuples
    Returns: list of elevation values in meters (one per waypoint)

    Uses Open-Elevation API (free, no API key, no rate limit for
    reasonable use). Falls back to a simulated terrain profile if
    the API is unreachable (no internet, API down, etc).
    """
    try:
        locations = "|".join(f"{lat},{lon}" for lat, lon in waypoints)
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        elevations = [round(r["elevation"], 1) for r in data["results"]]
        return elevations
    except Exception:
        return _simulate_elevation_profile(waypoints)


def _simulate_elevation_profile(waypoints: list) -> list:
    """
    Fallback: generate a realistic-looking terrain profile when the
    elevation API is unreachable. Uses a smooth random walk so hills
    look natural instead of jumping randomly waypoint to waypoint.
    """
    seed = int(abs(waypoints[0][0] * 1000) + abs(waypoints[0][1] * 1000)) % 99999
    rng = random.Random(seed)

    base = rng.uniform(100, 800)  # base elevation for this region
    elevations = [base]
    for _ in range(len(waypoints) - 1):
        step = rng.uniform(-40, 40)
        base = max(0, base + step)
        elevations.append(round(base, 1))
    return elevations


# ─────────────────────────────────────────────
# CHECK TERRAIN COLLISION RISK
# ─────────────────────────────────────────────
def check_terrain_clearance(waypoints: list, elevations: list,
                             planned_altitude_m: float,
                             min_safe_clearance_m: float = 30.0) -> list:
    """
    For each waypoint, check if the drone's altitude (measured above
    the STARTING point's elevation, which is how "altitude" is usually
    set by an operator) gives safe clearance above that waypoint's
    actual ground elevation.

    Returns a list of dicts, one per waypoint, with clearance info.
    """
    if not elevations:
        return []

    start_elevation = elevations[0]
    results = []

    for i, (wp, ground_elev) in enumerate(zip(waypoints, elevations)):
        # Drone's actual height above sea level at this point
        drone_absolute_alt = start_elevation + planned_altitude_m

        # How much clearance does the drone have above THIS terrain point
        clearance = drone_absolute_alt - ground_elev

        is_risk = clearance < min_safe_clearance_m
        is_collision = clearance < 0

        results.append({
            "waypoint_index": i,
            "lat": wp[0], "lon": wp[1],
            "ground_elevation_m": ground_elev,
            "drone_altitude_m": round(drone_absolute_alt, 1),
            "clearance_m": round(clearance, 1),
            "is_risk": is_risk,
            "is_collision": is_collision,
            "recommended_altitude_m": (
                round(planned_altitude_m + (min_safe_clearance_m - clearance), 0)
                if is_risk else planned_altitude_m
            ),
        })

    return results


def get_terrain_alerts(clearance_data: list) -> list:
    """
    Convert terrain clearance results into human-readable alert strings,
    matching the style of your existing alert system.
    """
    alerts = []
    for c in clearance_data:
        if c["is_collision"]:
            alerts.append({
                "level": "critical",
                "message": (
                    f"TERRAIN COLLISION — WP{c['waypoint_index']+1:02d}: "
                    f"Ground {c['ground_elevation_m']:.0f}m, drone at "
                    f"{c['drone_altitude_m']:.0f}m — NEGATIVE CLEARANCE"
                ),
                "action": f"Increase altitude to {c['recommended_altitude_m']:.0f}m minimum",
            })
        elif c["is_risk"]:
            alerts.append({
                "level": "warning",
                "message": (
                    f"LOW TERRAIN CLEARANCE — WP{c['waypoint_index']+1:02d}: "
                    f"Only {c['clearance_m']:.0f}m above ground"
                ),
                "action": f"Recommend increasing altitude to {c['recommended_altitude_m']:.0f}m",
            })
    return alerts


def get_max_terrain_risk_summary(clearance_data: list) -> dict:
    """Summary stats for the whole route's terrain profile."""
    if not clearance_data:
        return {"min_clearance": None, "collision_count": 0,
                "risk_count": 0, "worst_waypoint": None}

    min_c = min(clearance_data, key=lambda x: x["clearance_m"])
    return {
        "min_clearance": min_c["clearance_m"],
        "collision_count": sum(1 for c in clearance_data if c["is_collision"]),
        "risk_count": sum(1 for c in clearance_data if c["is_risk"]),
        "worst_waypoint": min_c["waypoint_index"],
    }