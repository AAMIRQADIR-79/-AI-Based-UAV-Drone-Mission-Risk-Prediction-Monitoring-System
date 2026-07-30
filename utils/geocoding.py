"""
geocoding.py - Location resolution for UAV Mission System
Supports: city names, village names, any place in India/World,
          direct GPS coordinates, reverse geocoding
"""

import math
import time
import random
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# ─────────────────────────────────────────────
# Geocoder setup
# ─────────────────────────────────────────────
_geolocator = Nominatim(
    user_agent="uav_mission_intelligence_v2",
    timeout=10
)

# ─────────────────────────────────────────────
# GEOCODE ANY PLACE (city, village, area, district)
# ─────────────────────────────────────────────
def geocode_place(place_name: str):
    """
    Convert any place name to (latitude, longitude).
    Works for cities, villages, districts, landmarks.
    
    Returns: (lat, lon) tuple
    Raises: ValueError if place not found
    """
    try:
        location = _geolocator.geocode(place_name, timeout=10)
        if location is None:
            raise ValueError(
                f"Location not found: '{place_name}'\n"
                f"Try being more specific. Example:\n"
                f"  'Ambegaon, Maharashtra'\n"
                f"  'Sector 14, Karnal, Haryana'\n"
                f"  Or use GPS coordinates directly."
            )
        return round(location.latitude, 5), round(location.longitude, 5)

    except GeocoderTimedOut:
        raise ValueError("Geocoder timed out. Check your internet connection.")
    except GeocoderServiceError as e:
        raise ValueError(f"Geocoder service error: {e}")
    except Exception as e:
        raise ValueError(f"Could not find location '{place_name}': {e}")


# ─────────────────────────────────────────────
# PARSE GPS COORDINATES (typed directly by user)
# ─────────────────────────────────────────────
def parse_coordinates(coord_string: str):
    """
    Parse GPS coordinates typed by user.
    Accepts formats:
        "28.6139, 77.2090"
        "28.6139 77.2090"
        "28°36'50\"N 77°12'32\"E"
    
    Returns: (lat, lon) tuple
    """
    coord_string = coord_string.strip()

    # Try simple decimal format: "28.6139, 77.2090"
    try:
        parts = coord_string.replace(",", " ").split()
        if len(parts) >= 2:
            lat = float(parts[0])
            lon = float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return round(lat, 5), round(lon, 5)
    except Exception:
        pass

    raise ValueError(
        f"Could not parse coordinates: '{coord_string}'\n"
        f"Please use format: 28.6139, 77.2090"
    )


# ─────────────────────────────────────────────
# REVERSE GEOCODE (coordinates → place name)
# ─────────────────────────────────────────────
def reverse_geocode(lat: float, lon: float) -> str:
    """
    Convert GPS coordinates to a human readable place name.
    Example: (29.7255, 76.9106) → "Karnal, Haryana, India"
    """
    try:
        location = _geolocator.reverse(
            f"{lat}, {lon}",
            language="en",
            timeout=10
        )
        if location:
            # Extract meaningful parts from address
            addr = location.raw.get("address", {})
            parts = []
            for key in ["village", "town", "city", "county",
                        "state_district", "state", "country"]:
                if key in addr and addr[key]:
                    parts.append(addr[key])
                    if len(parts) >= 3:
                        break
            if parts:
                return ", ".join(parts)
            return location.address[:60]
    except Exception:
        pass
    return f"{lat:.4f}°N, {lon:.4f}°E"


# ─────────────────────────────────────────────
# HAVERSINE DISTANCE
# ─────────────────────────────────────────────
def haversine_distance(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    """
    Calculate straight-line distance between two GPS points.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)

    return round(2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 3)


# ─────────────────────────────────────────────
# COMPASS BEARING
# ─────────────────────────────────────────────
def get_bearing(lat1: float, lon1: float,
                lat2: float, lon2: float) -> float:
    """
    Calculate compass bearing from point 1 to point 2.
    Returns degrees (0-360), where 0 = North.
    """
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2) -
         math.sin(lat1) * math.cos(lat2) * math.cos(dlon))

    bearing = math.degrees(math.atan2(x, y))
    return round((bearing + 360) % 360, 1)


def bearing_to_compass(bearing: float) -> str:
    """Convert bearing degrees to compass direction (N, NE, E, etc.)"""
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    index = round(bearing / 22.5) % 16
    return directions[index]


# ─────────────────────────────────────────────
# INTERPOLATE ROUTE WAYPOINTS
# ─────────────────────────────────────────────
def interpolate_route(lat1: float, lon1: float,
                       lat2: float, lon2: float,
                       n_points: int = 12):
    """
    Generate evenly spaced waypoints between source and destination.
    Returns list of (lat, lon) tuples including start and end points.
    """
    n_points = max(2, n_points)
    waypoints = []
    for i in range(n_points):
        fraction = i / (n_points - 1)
        lat = round(lat1 + fraction * (lat2 - lat1), 5)
        lon = round(lon1 + fraction * (lon2 - lon1), 5)
        waypoints.append((lat, lon))
    return waypoints


# ─────────────────────────────────────────────
# ESTIMATE FLIGHT TIME
# ─────────────────────────────────────────────
def estimate_flight_time(distance_km: float,
                          payload_kg: float,
                          wind_mps: float) -> dict:
    """
    Estimate flight time and energy consumption.
    Returns dict with flight_time_min, effective_speed, energy_wh, feasible.
    """
    from utils.config import UAV_SPECS

    # Wind reduces effective speed
    effective_speed = max(10.0, UAV_SPECS["cruise_speed_kmh"] - wind_mps * 1.5)

    # Flight time in minutes
    flight_time_min = (distance_km / effective_speed) * 60

    # Power consumption
    base_power = UAV_SPECS["cruise_power_w"]
    payload_power = payload_kg * 25        # 25W per kg of payload
    wind_power = max(0, wind_mps - 3) * 15
    total_power = base_power + payload_power + wind_power

    # Energy needed
    energy_wh = (total_power * flight_time_min) / 60

    # Max range check
    max_range = (UAV_SPECS["battery_capacity_wh"] * 0.8 / total_power) * effective_speed

    return {
        "flight_time_min":  round(flight_time_min, 1),
        "effective_speed":  round(effective_speed, 1),
        "energy_wh":        round(energy_wh, 2),
        "total_power_w":    round(total_power, 1),
        "max_range_km":     round(max_range, 1),
        "feasible":         distance_km <= max_range,
    }


# ─────────────────────────────────────────────
# GENERATE NO-FLY ZONES ALONG ROUTE
# ─────────────────────────────────────────────
def generate_nfz(src_lat, src_lon, dst_lat, dst_lon) -> list:
    """
    Generate realistic simulated No-Fly Zones along the route.
    Returns list of NFZ dicts.
    """
    from utils.config import NFZ_TYPES

    rng = random.Random(
        int(abs(src_lat * 1000) + abs(dst_lon * 1000)) % 99999
    )

    n_zones = rng.randint(2, 5)
    zones = []
    severities = ["prohibited", "restricted", "warning"]
    radii = {
        "Airport Exclusion Zone":   5.0,
        "Military Restricted Area": 10.0,
        "Government Facility":       2.0,
        "Wildlife Sanctuary":        4.0,
        "Nuclear Facility":          8.0,
        "Urban No-Fly Zone":         3.0,
        "Industrial Area":           2.5,
    }

    for i in range(n_zones):
        fraction = rng.uniform(0.1, 0.9)
        nfz_lat = src_lat + fraction * (dst_lat - src_lat) + rng.uniform(-1.5, 1.5)
        nfz_lon = src_lon + fraction * (dst_lon - src_lon) + rng.uniform(-1.5, 1.5)
        nfz_type = rng.choice(NFZ_TYPES)

        zones.append({
            "id":         f"NFZ-{i+1:03d}",
            "name":       f"{nfz_type} {i+1}",
            "type":       nfz_type,
            "lat":        round(nfz_lat, 4),
            "lon":        round(nfz_lon, 4),
            "radius_km":  radii.get(nfz_type, 3.0),
            "severity":   rng.choice(severities),
        })

    return zones


# ─────────────────────────────────────────────
# CHECK NFZ CONFLICTS WITH ROUTE
# ─────────────────────────────────────────────
def check_nfz_conflicts(waypoints: list, nfz_list: list) -> list:
    """
    Check if any waypoints are inside or near NFZ zones.
    Returns list of conflict dicts.
    """
    conflicts = []
    for wp in waypoints:
        for nfz in nfz_list:
            dist = haversine_distance(wp[0], wp[1], nfz["lat"], nfz["lon"])
            buffer = nfz["radius_km"] * 1.5   # 1.5x safety buffer
            if dist < buffer:
                conflicts.append({
                    "waypoint":     wp,
                    "nfz_id":       nfz["id"],
                    "nfz_name":     nfz["name"],
                    "distance_km":  round(dist, 2),
                    "inside_zone":  dist < nfz["radius_km"],
                    "severity":     nfz["severity"],
                })
    return conflicts


# ─────────────────────────────────────────────
# GET MGRS-STYLE GRID REFERENCE
# ─────────────────────────────────────────────
def get_grid_reference(lat: float, lon: float) -> str:
    """
    Returns a simplified military-style grid reference string.
    Example: "44R KU 48521 37890"
    """
    # Simplified UTM zone calculation
    zone_number = int((lon + 180) / 6) + 1
    zone_letters = "CDEFGHJKLMNPQRSTUVWX"
    zone_index = min(int((lat + 80) / 8), len(zone_letters) - 1)
    zone_letter = zone_letters[max(0, zone_index)]

    # Grid numbers from decimal degrees
    grid_e = int((lon % 6) * 10000)
    grid_n = int((lat % 8) * 10000)

    return f"{zone_number}{zone_letter} {grid_e:05d} {grid_n:05d}"