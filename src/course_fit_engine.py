import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
PLAYERS_PROCESSED = DATA_DIR / "Players" / "processed"
TOURNAMENT_DNA_DIR = DATA_DIR / "Tournaments" / "DNA"

MASTER_PATH = PLAYERS_PROCESSED / "master_player_ratings.csv"

SG_FEATURES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
ATTR_FEATURES = ["Power", "Approach", "ShortGame", "Putting", "Consistency", "DogFactor"]


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.strip()
    return df


def load_dna(course_profile_key: str) -> dict:
    path = TOURNAMENT_DNA_DIR / f"{course_profile_key}.json"
    if not path.exists():
        raise FileNotFoundError(f"DNA file not found for profile '{course_profile_key}' at {path}")
    with open(path, "r") as f:
        return json.load(f)


def z_score(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def compute_course_fit(master: pd.DataFrame, dna: dict) -> pd.DataFrame:
    df = master.copy()

    # Ensure columns exist; if missing, fill with 0
    for col in SG_FEATURES + ATTR_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    # Z-scores
    for col in SG_FEATURES + ATTR_FEATURES:
        df[f"Z_{col}"] = z_score(df[col])

    sg_weights = dna.get("sg_weights", {})
    attr_weights = dna.get("attr_weights", {})
    extra_weights = dna.get("extra_features", {})  # NEW
    scale = float(dna.get("scale", 0.35))

    # Combined z-based course fit index
    cf_index = np.zeros(len(df), dtype=float)

    for feat, w in sg_weights.items():
        z_col = f"Z_{feat}"
        if z_col in df.columns:
            cf_index += float(w) * df[z_col].to_numpy(dtype=float)

    for feat, w in attr_weights.items():
        z_col = f"Z_{feat}"
        if z_col in df.columns:
            cf_index += float(w) * df[z_col].to_numpy(dtype=float)
    
    # ---- Extra stats: GIR buckets, rough, scoring, putting, etc. ----
    for feat, w in extra_weights.items():
        if feat not in df.columns:
            print(f"[INFO] Extra feature '{feat}' not found in master table; skipping.")
            continue

        z_name = f"Z_extra_{feat}"
        if z_name not in df.columns:
            df[z_name] = z_score(df[feat])

        cf_index += float(w) * df[z_name].to_numpy(dtype=float)


    # Convert to strokes gained per round at this course archetype
    cf_sg_per_round = scale * cf_index

    out = df[["player_name"]].copy()
    out["CF_SG_per_round"] = cf_sg_per_round

    # Optional: keep the raw index for debugging
    out["CF_Index"] = cf_index

    return out.sort_values("CF_SG_per_round", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build course-fit table for a given course profile.")
    parser.add_argument(
        "--course-profile",
        required=True,
        help="course_profile_key (e.g. stadium_desert_scottsdale, bomber_ballstriking_riviera)",
    )
    args = parser.parse_args()

    course_key = args.course_profile
    dna = load_dna(course_key)

    print(f"=== Building course fit for profile: {course_key} ===")
    print(f"Name: {dna.get('name', '')}")
    print(f"Notes: {dna.get('notes', '')}")
    print("-" * 60)

    master = load_master()
    cf_table = compute_course_fit(master, dna)

    out_path = PLAYERS_PROCESSED / f"course_fit_{course_key}.csv"
    cf_table.to_csv(out_path, index=False)

    print(f"Saved course-fit table to: {out_path}")
    print("Top 10 by CF_SG_per_round:")
    print(cf_table.head(10))


if __name__ == "__main__":
    main()
