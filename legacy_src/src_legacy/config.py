from pathlib import Path

# Base project paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Default filenames (you'll plug real files in later)
PGA_STATS_FILE = RAW_DATA_DIR / "pga_2025_stats.csv"
COURSE_PROFILES_FILE = RAW_DATA_DIR / "course_profiles.csv"
PLAYER_COURSE_RESULTS_FILE = RAW_DATA_DIR / "player_course_results.csv"
