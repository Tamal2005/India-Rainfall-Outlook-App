"""Central configuration. Every path can be overridden with an environment variable
so the same image runs locally, in Docker, and on any PaaS."""
import os
from pathlib import Path

# <project root>/backend/app/config.py -> parents[2] == project root
ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data"))
DATA_DIR = Path(os.getenv("DATA_DIR", DATA_ROOT / "prepped"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", DATA_ROOT / "models"))
KNNR_DIR = Path(os.getenv("KNNR_DIR", DATA_ROOT / "disaggregation_knnr"))
FRONTEND_DIST = Path(os.getenv("FRONTEND_DIST", ROOT / "frontend" / "dist"))

# Comma-separated list of allowed origins, only needed if the frontend is hosted
# separately from the API (e.g. Vercel + Render). Same-origin deploys need nothing.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

MIN_DATE = "2001-01-01"
MAX_DATE = "2100-12-31"

CLIMATE_TARGETS_REQUIRED = [
    "temperature", "temperature_max", "temperature_min",
    "humidity", "wind_speed", "surface_pressure",
]

REGIME_FEATURE_COLS = [
    "distance_to_coast_km", "log_distance_to_coast_km",
    "rainfall_mean", "rainfall_std", "rainfall_p95", "rainfall_p99",
    "rain_rate_ge20", "rain_rate_ge50", "rain_rate_ge100", "rain_rate_ge150",
    "rainfall_extreme_ratio", "tail_risk_score",
]

SEVERITY_ORDER = ["no_rain", "light", "medium", "moderate", "heavy", "extreme"]
# Order of the severity network's 5 outputs (training: RAIN_CLASSES).
RAIN_CLASSES = SEVERITY_ORDER[1:]

CLIMATE_LABELS = {
    "temperature": ("Temperature", "°C"),
    "temperature_max": ("Maximum temperature", "°C"),
    "temperature_min": ("Minimum temperature", "°C"),
    "humidity": ("Humidity", "%"),
    "wind_speed": ("Wind speed", "m/s"),
    "surface_pressure": ("Surface pressure", "hPa"),
    "root_zone_soil_moisture": ("Root-zone soil moisture", "m³/m³"),
    "surface_soil_moisture": ("Surface soil moisture", "m³/m³"),
    "solar_radiation": ("Solar radiation", "W/m²"),
}

# Progress bars are on by default; set SHOW_PROGRESS=0 to silence them in server logs.
SHOW_PROGRESS = os.getenv("SHOW_PROGRESS", "1") != "0"
