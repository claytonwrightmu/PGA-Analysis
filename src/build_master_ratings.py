import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "Data" / "Players" / "processed"

SOURCE_PATH = DATA_PROCESSED / "player_talent_table.csv"
OUTPUT_PATH = DATA_PROCESSED / "master_player_ratings.csv"


# =========================
# HELPERS
# =========================

def load_base_table() -> pd.DataFrame:
    """
    Load the core talent table and normalize column names.
    This is the ONLY source of truth for the master ratings.
    """
    df = pd.read_csv(SOURCE_PATH)

    # Strip whitespace / weird chars from column names
    df.columns = df.columns.astype(str).str.strip()

    # Rename to standard names
    rename_map = {
        "player": "player_name",
        "tss_ott": "sg_ott",
        "tss_app": "sg_app",
        "tss_arg": "sg_arg",
        "tss_putt": "sg_putt",
        "Overall2K": "OverallRating",
        "Driving2K": "Power",
        "Approach2K": "Approach",
        "ShortGame2K": "ShortGame",
        "Putting2K": "Putting",
        "Consistency2K": "Consistency",
        "DogFactor2K": "DogFactor",
    }
    df = df.rename(columns=rename_map)

    # Make sure we have the basic stuff we expect
    required = [
        "player_name", "mu", "sigma", "form",
        "sg_ott", "sg_app", "sg_arg", "sg_putt",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in base table: {missing}")

    return df


def get_talent_column_name(df: pd.DataFrame) -> str:
    """
    Robustly find the TalentScore column, even if pandas
    has added suffixes like 'TalentScore_x'.
    """
    # exact match
    if "TalentScore" in df.columns:
        return "TalentScore"

    # look for variants like TalentScore_x, TalentScore_y, etc.
    candidates = [c for c in df.columns if c.startswith("TalentScore")]
    if len(candidates) == 1:
        df.rename(columns={candidates[0]: "TalentScore"}, inplace=True)
        return "TalentScore"
    elif len(candidates) > 1:
        # if somehow multiple, pick the first and rename
        chosen = candidates[0]
        df.rename(columns={chosen: "TalentScore"}, inplace=True)
        return "TalentScore"

    raise ValueError(f"No TalentScore-like column found. Columns: {list(df.columns)}")


# =========================
# TIERING
# =========================

def assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tiers based on TalentScore percentiles:
        S: top 5%
        A: next 15%
        B: next 30%
        C: next 30%
        D: bottom 20%
    """
    talent_col = get_talent_column_name(df)
    talent = df[talent_col]

    q95 = talent.quantile(0.95)
    q80 = talent.quantile(0.80)
    q50 = talent.quantile(0.50)
    q20 = talent.quantile(0.20)

    def _tier(x: float) -> str:
        if x >= q95:
            return "S"
        elif x >= q80:
            return "A"
        elif x >= q50:
            return "B"
        elif x >= q20:
            return "C"
        else:
            return "D"

    df["Tier"] = talent.apply(_tier)
    return df


# =========================
# ROUNDING / CLEANUP
# =========================

def round_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Core performance numbers
    two_dp = ["mu", "sg_ott", "sg_app", "sg_arg", "sg_putt"]
    three_dp = ["sigma", "form"]

    # TalentScore might not exist when this is first called
    if "TalentScore" in df.columns:
        two_dp.append("TalentScore")

    for col in two_dp:
        if col in df.columns:
            df[col] = df[col].astype(float).round(2)

    for col in three_dp:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    return df


# =========================
# MASTER PIPELINE
# =========================

def build_master_table():
    print("Loading data...")

    df = load_base_table()

    # Ensure TalentScore column is present and normalized
    talent_col = get_talent_column_name(df)
    if talent_col != "TalentScore":
        # get_talent_column_name already renamed it, just being explicit
        df.rename(columns={talent_col: "TalentScore"}, inplace=True)

    # Compute new tiers based on current TalentScore distribution
    df = assign_tiers(df)

    # Round decimals for cleanliness
    df = round_numeric(df)

    # Sort best to worst by TalentScore
    df = df.sort_values("TalentScore", ascending=False)

    # Reorder columns into a nice, game-ready layout
    preferred_order = [
        "player_name",
        "player_id",
        "mu",
        "sigma",
        "form",
        "sg_ott",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "TalentScore",
        "Tier",
        "OverallRating",
        "Power",
        "Approach",
        "ShortGame",
        "Putting",
        "Consistency",
        "DogFactor",
    ]
    cols = [c for c in preferred_order if c in df.columns]
    # add any extra columns at the end
    extras = [c for c in df.columns if c not in cols]
    df = df[cols + extras]

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved master ratings → {OUTPUT_PATH}")
    print(df.head(10))


if __name__ == "__main__":
    build_master_table()
