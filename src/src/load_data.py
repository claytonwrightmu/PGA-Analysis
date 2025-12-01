import pandas as pd
from .config import PGA_STATS_FILE, COURSE_PROFILES_FILE, PLAYER_COURSE_RESULTS_FILE

def load_pga_stats(path: str | None = None) -> pd.DataFrame:
    """
    Load season-level strokes gained data per player.

    Expected CSV columns (per row = one player for a season):
      - season
      - player_name
      - sg_total_per_round
      - sg_off_tee_per_round
      - sg_approach_per_round
      - sg_around_green_per_round
      - sg_putting_per_round
      - driving_distance_avg
      - fairway_pct
    """
    file_path = path or PGA_STATS_FILE
    return pd.read_csv(file_path)

def load_course_profiles(path: str | None = None) -> pd.DataFrame:
    """
    Load course setup information (length, rough, fairways, etc.).
    """
    file_path = path or COURSE_PROFILES_FILE
    return pd.read_csv(file_path)

def load_player_course_results(path: str | None = None) -> pd.DataFrame:
    """
    Load player performance by course (SG by course).
    """
    file_path = path or PLAYER_COURSE_RESULTS_FILE
    return pd.read_csv(file_path)
