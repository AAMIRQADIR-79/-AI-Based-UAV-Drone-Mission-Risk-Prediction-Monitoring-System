"""
config.py - Central configuration for UAV Mission Intelligence System
All constants, thresholds, API keys, and settings in one place
"""

# ─────────────────────────────────────────────
# YOUR API KEY - Already set for you
# ─────────────────────────────────────────────
OPENWEATHER_API_KEY = "OPENWEATHER_API_KEY"

# ─────────────────────────────────────────────
# UAV PHYSICAL LIMITS
# ─────────────────────────────────────────────
UAV_SPECS = {
    "max_range_km":          150.0,
    "max_payload_kg":          5.0,
    "max_wind_safe_mps":      15.0,
    "critical_battery_pct":   15.0,
    "warning_battery_pct":    25.0,
    "max_altitude_m":        400.0,
    "cruise_speed_kmh":       60.0,
    "battery_capacity_wh":   100.0,
    "signal_min_pct":         20.0,
    "signal_warn_pct":        35.0,
    "temp_max_c":             40.0,
    "temp_min_c":              0.0,
    "hover_power_w":         400.0,
    "cruise_power_w":        300.0,
}

RISK_LEVELS = {
    "safe":     (0,  35),
    "moderate": (35, 60),
    "high":     (60, 80),
    "critical": (80, 100),
}

FEATURE_NAMES = [
    "distance_km",
    "battery_pct",
    "wind_speed_mps",
    "payload_kg",
    "signal_strength_pct",
    "temperature_c",
    "altitude_m",
    "humidity_pct",
    "flight_time_min",
    "battery_drain_rate",
]

SEQ_LEN = 12

MISSION_TYPES = [
    "Surveillance",
    "Delivery",
    "Mapping / Survey",
    "Search & Rescue",
    "Agriculture Spray",
    "Inspection",
    "Photography",
    "Military Recon",
]

TERRAIN_TYPES = {
    "Urban":        {"wind_factor": 0.8,  "signal_factor": 0.85, "risk_factor": 1.2},
    "Suburban":     {"wind_factor": 0.9,  "signal_factor": 0.90, "risk_factor": 1.0},
    "Rural":        {"wind_factor": 1.1,  "signal_factor": 0.95, "risk_factor": 0.9},
    "Village":      {"wind_factor": 1.0,  "signal_factor": 0.88, "risk_factor": 0.95},
    "Mountain":     {"wind_factor": 1.4,  "signal_factor": 0.70, "risk_factor": 1.5},
    "Coastal":      {"wind_factor": 1.3,  "signal_factor": 0.92, "risk_factor": 1.2},
    "Desert":       {"wind_factor": 1.2,  "signal_factor": 0.98, "risk_factor": 1.1},
    "Forest":       {"wind_factor": 0.7,  "signal_factor": 0.80, "risk_factor": 1.0},
    "Agricultural": {"wind_factor": 1.05, "signal_factor": 0.93, "risk_factor": 0.85},
}

COLORS = {
    "bg_main":       "#1C1C1C",
    "bg_panel":      "#252525",
    "bg_card":       "#2D2D2D",
    "bg_dark":       "#141414",
    "green":         "#00CC00",
    "green_bright":  "#00FF41",
    "amber":         "#FFAA00",
    "red":           "#FF3333",
    "blue":          "#00AAFF",
    "white":         "#FFFFFF",
    "grey":          "#888888",
    "grey_light":    "#CCCCCC",
    "hud_sky":       "#1A5276",
    "hud_ground":    "#6E2F1A",
    "hud_horizon":   "#FFFFFF",
    "risk_safe":     "#00CC00",
    "risk_moderate": "#FFAA00",
    "risk_high":     "#FF6600",
    "risk_critical": "#FF0000",
    "chart_bg":      "#000000",
    "chart_grid":    "#333333",
    "chart_line":    "#00AAFF",
}

NFZ_TYPES = [
    "Airport Exclusion Zone",
    "Military Restricted Area",
    "Government Facility",
    "Wildlife Sanctuary",
    "Nuclear Facility",
    "Urban No-Fly Zone",
    "Industrial Area",
]

ALERT_LEVELS = {
    "critical": 0,
    "warning":  1,
    "caution":  2,
    "info":     3,
}

APP_SETTINGS = {
    "app_title":        "UAV Mission Intelligence System",
    "app_icon":         "🛸",
    "simulation_delay": 0.5,
    "n_waypoints":      12,
    "map_zoom":         8,
    "version":          "v2.0",
}