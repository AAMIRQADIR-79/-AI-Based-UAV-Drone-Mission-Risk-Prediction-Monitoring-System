"""
rth.py - Intelligent Return-To-Home Feasibility Calculator

Given the drone's CURRENT position, battery, and flight conditions,
calculates whether RTH is actually achievable - and if not, finds the
nearest reachable emergency landing zone instead.
"""

import math
from utils.geocoding import haversine_distance, get_bearing, bearing_to_compass


def check_rth_feasibility(
    current_lat: float, current_lon: float,
    home_lat: float, home_lon: float,
    current_battery_pct: float,
    battery_drain_rate: float,
    safety_margin_pct: float = 10.0,
) -> dict:
    """
    Determines if the drone can safely return to its home/launch point
    given current battery and the physics-based drain rate.
    """
    distance_home_km = haversine_distance(current_lat, current_lon,
                                           home_lat, home_lon)
    battery_needed_pct = distance_home_km * battery_drain_rate
    battery_after_rth = current_battery_pct - battery_needed_pct
    margin = battery_after_rth - safety_margin_pct

    bearing = get_bearing(current_lat, current_lon, home_lat, home_lon)
    compass = bearing_to_compass(bearing)

    feasible = margin >= 0

    if margin >= 15:
        confidence = "HIGH"
    elif margin >= 0:
        confidence = "MODERATE"
    elif margin >= -10:
        confidence = "LOW"
    else:
        confidence = "CRITICAL"

    return {
        "feasible": feasible,
        "confidence": confidence,
        "distance_home_km": round(distance_home_km, 2),
        "battery_needed_pct": round(battery_needed_pct, 1),
        "battery_after_rth_pct": round(battery_after_rth, 1),
        "safety_margin_used": round(margin, 1),
        "bearing_home": round(bearing, 0),
        "compass_home": compass,
        "verdict": (
            f"RTH FEASIBLE - will arrive home with "
            f"{battery_after_rth:.1f}% battery remaining"
            if feasible else
            f"RTH NOT FEASIBLE - would need {battery_needed_pct:.1f}% "
            f"but only {current_battery_pct:.1f}% available"
        ),
    }


def generate_emergency_landing_zones(
    home_lat: float, home_lon: float,
    dest_lat: float, dest_lon: float,
    n_zones: int = 4,
) -> list:
    """
    Generate plausible emergency landing zone candidates spread along
    and around the route corridor.
    """
    import random
    seed = int(abs(home_lat * 1000) + abs(dest_lon * 1000)) % 99999
    rng = random.Random(seed)

    zone_names = [
        "Open Field Alpha", "Agricultural Clearing Bravo",
        "Dry Riverbed Charlie", "Sports Ground Delta",
        "Fallow Land Echo", "Helipad Foxtrot",
    ]

    zones = []
    for i in range(n_zones):
        frac = rng.uniform(0.15, 0.85)
        base_lat = home_lat + frac * (dest_lat - home_lat)
        base_lon = home_lon + frac * (dest_lon - home_lon)
        offset_lat = rng.uniform(-0.03, 0.03)
        offset_lon = rng.uniform(-0.03, 0.03)

        zones.append({
            "id": f"ELZ-{i+1:02d}",
            "name": zone_names[i % len(zone_names)],
            "lat": round(base_lat + offset_lat, 5),
            "lon": round(base_lon + offset_lon, 5),
            "surface": rng.choice(["Grass", "Dirt", "Paved", "Sand"]),
            "suitable_for_landing": True,
        })
    return zones


def find_best_emergency_zone(
    current_lat: float, current_lon: float,
    zones: list,
    current_battery_pct: float,
    battery_drain_rate: float,
    safety_margin_pct: float = 8.0,
) -> dict:
    """
    Find the nearest emergency landing zone that is actually reachable
    with remaining battery.
    """
    reachable = []
    for zone in zones:
        dist = haversine_distance(current_lat, current_lon,
                                   zone["lat"], zone["lon"])
        batt_needed = dist * battery_drain_rate
        batt_remaining_after = current_battery_pct - batt_needed
        is_reachable = batt_remaining_after >= safety_margin_pct

        bearing = get_bearing(current_lat, current_lon,
                               zone["lat"], zone["lon"])

        reachable.append({
            **zone,
            "distance_km": round(dist, 2),
            "battery_needed_pct": round(batt_needed, 1),
            "battery_remaining_pct": round(batt_remaining_after, 1),
            "reachable": is_reachable,
            "bearing": round(bearing, 0),
            "compass": bearing_to_compass(bearing),
        })

    reachable.sort(key=lambda z: z["distance_km"])
    reachable_zones = [z for z in reachable if z["reachable"]]

    if reachable_zones:
        best = reachable_zones[0]
        return {
            "found": True,
            "best_zone": best,
            "all_zones": reachable,
            "message": (
                f"NEAREST SAFE ZONE: {best['name']} ({best['id']}) - "
                f"{best['distance_km']}km {best['compass']} - REACHABLE, "
                f"{best['battery_remaining_pct']}% battery remaining on arrival"
            ),
        }
    else:
        return {
            "found": False,
            "best_zone": None,
            "all_zones": reachable,
            "message": (
                "NO EMERGENCY ZONE REACHABLE - INSUFFICIENT BATTERY. "
                "RECOMMEND IMMEDIATE CONTROLLED LANDING AT CURRENT POSITION."
            ),
        }


def full_contingency_check(
    current_lat: float, current_lon: float,
    home_lat: float, home_lon: float,
    dest_lat: float, dest_lon: float,
    current_battery_pct: float,
    battery_drain_rate: float,
) -> dict:
    """
    Full decision-support check: first see if RTH works, and if not,
    immediately compute the best emergency alternative.
    """
    rth = check_rth_feasibility(
        current_lat, current_lon, home_lat, home_lon,
        current_battery_pct, battery_drain_rate)

    result = {"rth": rth, "emergency": None}

    if not rth["feasible"]:
        zones = generate_emergency_landing_zones(
            home_lat, home_lon, dest_lat, dest_lon)
        emergency = find_best_emergency_zone(
            current_lat, current_lon, zones,
            current_battery_pct, battery_drain_rate)
        result["emergency"] = emergency

    return result