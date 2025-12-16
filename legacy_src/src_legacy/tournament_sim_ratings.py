import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# CONFIG / PATHS
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
PLAYERS_PROCESSED = DATA_DIR / "Players" / "processed"
MASTER_PATH = PLAYERS_PROCESSED / "master_player_ratings.csv"


def load_master() -> pd.DataFrame:
    """
    Load the master player ratings table.
    """
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.strip()
    return df


def load_course_fit(course_profile_key: str) -> pd.DataFrame:
    """
    Load course-fit table for a given course profile.
    Expects a file named: course_fit_<course_profile_key>.csv
    with columns: player_name, CF_SG_per_round, CF_Index (optional).
    """
    path = PLAYERS_PROCESSED / f"course_fit_{course_profile_key}.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def z_score(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def simulate_rating_based(
    df: pd.DataFrame,
    n_sims: int = 5000,
    skill_weight: float = 0.38,
    course_volatility: float = 1.0,
    form_weight: float = 0.20,
    sigma_scale: float = 0.20,
    upset_scale: float = 0.15,
) -> pd.DataFrame:
    """
    Advanced tournament simulator.

    performance_i = skill_weight * strength_i
                    + eps_course_i
                    + eps_sigma_i
                    + eps_upset_i

    Where:
        strength_i  = blended z-scores of TalentScore, course-fit, and form
        eps_course  = N(0, course_volatility)
        eps_sigma   = N(0, sigma_scale * sigma_i)
        eps_upset   = zero-mean Exponential-based tail noise (creates true upsets)

    This creates:
        - course-specific volatility
        - player-specific volatility (sigma)
        - upside spikes / bust outcomes
    """

    df = df.copy()
    n_players = len(df)

    # -------------------------
    # 1) BUILD STRENGTH INDEX
    # -------------------------

    # Make sure required columns exist
    for col in ["TalentScore", "CF_SG_per_round", "form", "sigma"]:
        if col not in df.columns:
            df[col] = 0.0

    z_talent_raw = z_score(df["TalentScore"])
    z_cf_raw = z_score(df["CF_SG_per_round"])
    z_form_raw = z_score(df["form"])

    # Clip to avoid insane outliers
    z_talent = z_talent_raw.clip(-2.0, 2.0)
    z_cf = z_cf_raw.clip(-2.0, 2.0)
    z_form = z_form_raw.clip(-2.0, 2.0)

    # Base strength: bias toward talent, then course fit, then form
    strength = (
        0.65 * z_talent +
        0.25 * z_cf +
        0.10 * z_form
    )

    # Slight compression so even Scottie doesn't dominate fields
    mean_strength = strength.mean()
    strength = mean_strength + 0.75 * (strength - mean_strength)

    strength = strength.to_numpy(dtype=float)

    # -------------------------
    # 2) SIMULATION LOOPS
    # -------------------------

    win_counts = np.zeros(n_players, dtype=int)
    top5_counts = np.zeros(n_players, dtype=int)
    rank_sums = np.zeros(n_players, dtype=float)

    sigma_vals = df["sigma"].to_numpy(dtype=float)

    for _ in range(n_sims):
        # Course-level randomness (conditions, pin positions, etc.)
        eps_course = np.random.normal(
            loc=0.0,
            scale=course_volatility,
            size=n_players
        )

        # Player-specific volatility (how wild a guy is week-to-week)
        eps_sigma = np.random.normal(
            loc=0.0,
            scale=sigma_scale * np.maximum(sigma_vals, 0.0),
            size=n_players
        )

        # Upset engine: heavy-tailed, zero-mean
        raw_upset = np.random.exponential(scale=upset_scale, size=n_players)
        eps_upset = raw_upset - raw_upset.mean()

        perf = skill_weight * strength + eps_course + eps_sigma + eps_upset

        # higher perf = better
        ranks = (-perf).argsort()

        win_counts[ranks[0]] += 1
        top5_counts[ranks[:5]] += 1

        positions = np.empty_like(ranks)
        positions[ranks] = np.arange(1, n_players + 1)
        rank_sums += positions

    # -------------------------
    # 3) OUTPUT TABLE
    # -------------------------

    out = df[["player_name", "Tier", "TalentScore", "OverallRating"]].copy()
    out["win_pct"] = win_counts / n_sims * 100.0
    out["top5_pct"] = top5_counts / n_sims * 100.0
    out["avg_finish"] = rank_sums / n_sims

    out = out.sort_values("win_pct", ascending=False).reset_index(drop=True)
    return out
