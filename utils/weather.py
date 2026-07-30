"""
weather.py - Live weather fetching using OpenWeatherMap API
Fetches real weather at any GPS coordinate (city, village, remote area)
Falls back to simulation if API fails
"""

import requests
import random
import math
import time
from utils.config import OPENWEATHER_API_KEY


# ─────────────────────────────────────────────
# MAIN FUNCTION - Get weather at any GPS point
# ─────────────────────────────────────────────
def get_weather(lat: float, lon: float) -> dict:
    """
    Fetch live weather at exact GPS coordinates.
    Works for any location - city, village, forest, remote area.
    Falls back to realistic simulation if API fails.
    """
    result = _fetch_live_weather(lat, lon)
    if result:
        return result
    return _simulate_weather(lat, lon)


# ─────────────────────────────────────────────
# LIVE WEATHER FROM OPENWEATHERMAP API
# ─────────────────────────────────────────────
def _fetch_live_weather(lat: float, lon: float) -> dict:
    """
    Call OpenWeatherMap current weather API.
    Returns dict or None if failed.
    """
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}"
            f"&appid={OPENWEATHER_API_KEY}"
            f"&units=metric"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        # Extract all weather data
        main      = data["main"]
        wind      = data["wind"]
        clouds    = data["clouds"]
        weather   = data["weather"][0]
        sys_data  = data.get("sys", {})

        # Wind gust (not always available)
        wind_gust = wind.get("gust", round(wind["speed"] * 1.3, 1))

        # Visibility in km (API gives metres)
        visibility_m  = data.get("visibility", 10000)
        visibility_km = round(visibility_m / 1000, 1)

        return {
            # Core data
            "temperature":   round(main["temp"], 1),
            "feels_like":    round(main["feels_like"], 1),
            "temp_min":      round(main["temp_min"], 1),
            "temp_max":      round(main["temp_max"], 1),
            "humidity":      main["humidity"],
            "pressure":      main["pressure"],

            # Wind
            "wind_speed":    round(wind["speed"], 1),
            "wind_gust":     round(wind_gust, 1),
            "wind_direction":wind.get("deg", 0),
            "wind_compass":  _degrees_to_compass(wind.get("deg", 0)),

            # Sky
            "cloud_cover":   clouds["all"],
            "visibility_km": visibility_km,
            "description":   weather["description"].title(),
            "weather_id":    weather["id"],
            "weather_icon":  weather["icon"],

            # Beaufort scale
            "beaufort_force": _beaufort_force(wind["speed"]),
            "beaufort_desc":  _beaufort_description(wind["speed"]),

            # UAV specific risk from weather
            "uav_weather_risk": _calculate_weather_risk(
                wind["speed"], main["temp"],
                visibility_km, clouds["all"]
            ),

            # Source info
            "source":        "OpenWeatherMap (Live API)",
            "lat":           lat,
            "lon":           lon,
            "fetch_time":    time.strftime("%H:%M:%S"),
        }

    except requests.exceptions.ConnectionError:
        return None   # No internet - use simulation
    except requests.exceptions.Timeout:
        return None   # Timeout - use simulation
    except Exception:
        return None   # Any error - use simulation


# ─────────────────────────────────────────────
# WEATHER SIMULATION (when API not available)
# ─────────────────────────────────────────────
def _simulate_weather(lat: float, lon: float) -> dict:
    """
    Generate realistic simulated weather based on location.
    Uses latitude for temperature and seasonal patterns.
    """
    seed = int(abs(lat * 1000) + abs(lon * 1000)) % 100000
    rng  = random.Random(seed)

    # Temperature based on latitude (tropical = hot, polar = cold)
    base_temp = 35 - abs(lat) * 0.5
    temp      = round(base_temp + rng.uniform(-5, 8), 1)

    # Wind speed
    wind_speed = round(rng.uniform(1, 14), 1)
    wind_gust  = round(wind_speed * rng.uniform(1.1, 1.5), 1)
    wind_dir   = rng.randint(0, 359)

    humidity    = rng.randint(40, 85)
    pressure    = rng.randint(1008, 1020)
    cloud_cover = rng.randint(0, 80)
    visibility  = round(rng.uniform(5, 15), 1)

    conditions = [
        "Clear Sky", "Partly Cloudy", "Overcast",
        "Light Rain", "Haze", "Mist", "Smoke"
    ]
    description = rng.choice(conditions)

    return {
        "temperature":      temp,
        "feels_like":       round(temp - rng.uniform(0, 3), 1),
        "temp_min":         round(temp - 2, 1),
        "temp_max":         round(temp + 2, 1),
        "humidity":         humidity,
        "pressure":         pressure,
        "wind_speed":       wind_speed,
        "wind_gust":        wind_gust,
        "wind_direction":   wind_dir,
        "wind_compass":     _degrees_to_compass(wind_dir),
        "cloud_cover":      cloud_cover,
        "visibility_km":    visibility,
        "description":      description,
        "weather_id":       800,
        "weather_icon":     "01d",
        "beaufort_force":   _beaufort_force(wind_speed),
        "beaufort_desc":    _beaufort_description(wind_speed),
        "uav_weather_risk": _calculate_weather_risk(
            wind_speed, temp, visibility, cloud_cover
        ),
        "source":       "Simulated (No Internet)",
        "lat":          lat,
        "lon":          lon,
        "fetch_time":   time.strftime("%H:%M:%S"),
    }


# ─────────────────────────────────────────────
# GET WEATHER FORECAST (5 day / 3 hour)
# ─────────────────────────────────────────────
def get_forecast(lat: float, lon: float) -> list:
    """
    Fetch 5-day weather forecast at GPS coordinates.
    Returns list of forecast dicts (every 3 hours).
    Falls back to simulation if API fails.
    """
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/forecast"
            f"?lat={lat}&lon={lon}"
            f"&appid={OPENWEATHER_API_KEY}"
            f"&units=metric"
            f"&cnt=16"
        )
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        data = response.json()

        forecast = []
        for item in data["list"]:
            forecast.append({
                "datetime":    item["dt_txt"],
                "time":        item["dt_txt"][11:16],
                "temperature": round(item["main"]["temp"], 1),
                "wind_speed":  round(item["wind"]["speed"], 1),
                "humidity":    item["main"]["humidity"],
                "description": item["weather"][0]["description"].title(),
                "cloud_cover": item["clouds"]["all"],
            })
        return forecast

    except Exception:
        return _simulate_forecast(lat, lon)


def _simulate_forecast(lat: float, lon: float) -> list:
    """Simulate a 48-hour forecast."""
    base = _simulate_weather(lat, lon)
    forecast = []
    for h in range(16):
        hour = h * 3
        temp_offset = 3 * math.sin((hour - 6) * math.pi / 12)
        forecast.append({
            "datetime":    f"2025-07-{11 + h//8:02d} {hour%24:02d}:00:00",
            "time":        f"{hour%24:02d}:00",
            "temperature": round(base["temperature"] + temp_offset, 1),
            "wind_speed":  round(base["wind_speed"] + random.uniform(-2, 2), 1),
            "humidity":    base["humidity"] + random.randint(-5, 5),
            "description": base["description"],
            "cloud_cover": base["cloud_cover"],
        })
    return forecast


# ─────────────────────────────────────────────
# UAV WEATHER RISK CALCULATION
# ─────────────────────────────────────────────
def _calculate_weather_risk(wind_mps: float, temp_c: float,
                              vis_km: float, cloud_pct: int) -> dict:
    """
    Calculate UAV-specific risk from weather conditions.
    Returns dict with individual scores and composite.
    """
    # Wind risk (0-100)
    wind_risk = min(100, (wind_mps / 15.0) * 100)

    # Temperature risk (0-100)
    temp_risk = 0
    if temp_c > 38:
        temp_risk = min(100, (temp_c - 38) * 12)
    elif temp_c < 2:
        temp_risk = min(100, (2 - temp_c) * 10)

    # Visibility risk (0-100)
    vis_risk = max(0, min(100, (5 - vis_km) * 20)) if vis_km < 5 else 0

    # Cloud risk (0-100)
    cloud_risk = (cloud_pct / 100) * 30

    # Composite weighted score
    composite = (
        wind_risk  * 0.45 +
        temp_risk  * 0.20 +
        vis_risk   * 0.25 +
        cloud_risk * 0.10
    )

    return {
        "wind_risk":    round(wind_risk, 1),
        "temp_risk":    round(temp_risk, 1),
        "vis_risk":     round(vis_risk, 1),
        "cloud_risk":   round(cloud_risk, 1),
        "composite":    round(composite, 1),
        "safe_to_fly":  composite < 40,
    }


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def _degrees_to_compass(degrees: float) -> str:
    """Convert wind direction degrees to compass label."""
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW"
    ]
    index = round(degrees / 22.5) % 16
    return directions[index]


def _beaufort_force(wind_mps: float) -> int:
    """Get Beaufort scale force number from wind speed in m/s."""
    thresholds = [0.3, 1.6, 3.4, 5.5, 8.0,
                  10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
    for i, threshold in enumerate(thresholds):
        if wind_mps <= threshold:
            return i
    return 12


def _beaufort_description(wind_mps: float) -> str:
    """Get Beaufort scale description from wind speed in m/s."""
    descriptions = [
        "Calm", "Light Air", "Light Breeze", "Gentle Breeze",
        "Moderate Breeze", "Fresh Breeze", "Strong Breeze",
        "Near Gale", "Gale", "Strong Gale", "Storm",
        "Violent Storm", "Hurricane"
    ]
    return descriptions[_beaufort_force(wind_mps)]