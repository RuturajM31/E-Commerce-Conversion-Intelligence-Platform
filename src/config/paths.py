# paths.py
# This file stores the main folder paths used in the project.
# We keep paths here so the rest of the code stays clean and portable.

from pathlib import Path


# PROJECT_ROOT means the main project folder.
# This file is inside: src/config/paths.py
# parents[2] moves two levels up:
# paths.py -> config -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Main data folders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "retailrocket"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# Model folders
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"


# Report output folders
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


# Log and monitoring folders
LOGS_DIR = PROJECT_ROOT / "logs"
MONITORING_DIR = PROJECT_ROOT / "monitoring"