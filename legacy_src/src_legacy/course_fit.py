import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "Data" / "Players" / "processed"
MASTER_PATH = DATA_PROCESSED / "master_player_ratings.csv"

# Features we'll use for course fit
SG_FEATURES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
ATTR_FEATURES = ["Power", "Approach", "ShortGame", "Putting", "Consistency", "DogFactor"]

# Course profiles:
# Each profile has:
#  - sg_weights: weights on SG components (already strokes-based)
#  - attr_weights: weights on attribute ratings (2K-style ratings)
#  - scale: how many SG/round per 1 unit of CF_Z
COURSE_PROFILES = {
    # Long, modern, distance-rewarding but not insane penalty
    "bomber_friendly_modern": {
        "description": "Long, semi-wide, modern bomber track",
        "sg_weights": {
            "sg_ott": 1.3,
            "sg_app": 1.1,
            "sg_arg": 0.6,
            "sg_putt": 0.6,
        },
        "attr_weights": {
            "Power": 1.3,
            "Approach": 1.0,
            "ShortGame": 0.5,
            "Putting": 0.6,
            "Consistency": 0.7,
            "DogFactor": 0.6,
        },
        "scale": 0.09,
    },

    # Narrow fairways, thick rough, US Open-ish
    "penal_driving_us_open_style": {
        "description": "Narrow, thick rough, brutal misses",
        "sg_weights": {
            "sg_ott": 1.4,
            "sg_app": 1.2,
            "sg_arg": 0.7,
            "sg_putt": 0.6,
        },
        "attr_weights": {
            "Power": 0.6,
            "Approach": 1.1,
            "ShortGame": 0.7,
            "Putting": 0.6,
            "Consistency": 1.4,
            "DogFactor": 1.0,
        },
        "scale": 0.09,
    },

    # Short, positional, strategy emphasis
    "short_positional_strategy": {
        "description": "Short, doglegs, play-to-spots course",
        "sg_weights": {
            "sg_ott": 0.7,
            "sg_app": 1.2,
            "sg_arg": 1.0,
            "sg_putt": 1.0,
        },
        "attr_weights": {
            "Power": 0.4,
            "Approach": 1.2,
            "ShortGame": 1.0,
            "Putting": 1.0,
            "Consistency": 1.1,
            "DogFactor": 0.9,
        },
        "scale": 0.09,
    },

    # Augusta / elite second-shot + green complexes (Masters-like)
    "augusta_masters": {
        "description": "Augusta-style: second shot + wild greens + short game",
        "sg_weights": {
            "sg_ott": 0.9,
            "sg_app": 1.5,
            "sg_arg": 1.3,
            "sg_putt": 1.3,
        },
        "attr_weights": {
            "Power": 1.0,
            "Approach": 1.4,
            "ShortGame": 1.3,
            "Putting": 1.3,
            "Consistency": 1.2,
            "DogFactor": 1.3,
        },
        "scale": 0.10,
    },

    # Wedge & birdiefest
    "wedge_birdiefest": {
        "description": "Short-ish birdiefest, tons of wedges and low scores",
        "sg_weights": {
            "sg_ott": 0.7,
            "sg_app": 1.2,
            "sg_arg": 0.9,
            "sg_putt": 1.3,
        },
        "attr_weights": {
            "Power": 0.6,
            "Approach": 1.1,
            "ShortGame": 1.0,
            "Putting": 1.4,
            "Consistency": 0.9,
            "DogFactor": 1.0,
        },
        "scale": 0.09,
    },

    # TPC-style target golf with water and forced carries
    "water_target_tpc": {
        "description": "Water everywhere, target golf, punishment for big misses",
        "sg_weights": {
            "sg_ott": 1.0,
            "sg_app": 1.2,
            "sg_arg": 0.9,
            "sg_putt": 1.0,
        },
        "attr_weights": {
            "Power": 0.9,
            "Approach": 1.1,
            "ShortGame": 0.9,
            "Putting": 1.0,
            "Consistency": 1.2,
            "DogFactor": 1.2,
        },
        "scale": 0.09,
    },

    # Links / exposed wind / firm ground
    "wind_links_open": {
        "description": "Open Championship-style links: wind, firm, ground game",
        "sg_weights": {
            "sg_ott": 0.9,
            "sg_app": 1.1,
            "sg_arg": 1.2,
            "sg_putt": 1.1,
        },
        "attr_weights": {
            "Power": 0.8,
            "Approach": 1.0,
            "ShortGame": 1.4,
            "Putting": 1.2,
            "Consistency": 1.3,
            "DogFactor": 1.4,
        },
        "scale": 0.10,
    },

    # PGA Championship style: long, ball-striker heavy
    "pga_ballstriker_major": {
        "description": "Modern PGA: long, tee-to-green heavy, major setup",
        "sg_weights": {
            "sg_ott": 1.3,
            "sg_app": 1.3,
            "sg_arg": 0.8,
            "sg_putt": 0.7,
        },
        "attr_weights": {
            "Power": 1.2,
            "Approach": 1.3,
            "ShortGame": 0.7,
            "Putting": 0.7,
            "Consistency": 1.1,
            "DogFactor": 1.1,
        },
        "scale": 0.09,
    },

    # Classic ball-striker track like Riviera
    "classic_ballstriker": {
        "description": "Classic shot-making course, ball-striking premium",
        "sg_weights": {
            "sg_ott": 1.1,
            "sg_app": 1.3,
            "sg_arg": 0.9,
            "sg_putt": 0.8,
        },
        "attr_weights": {
            "Power": 1.0,
            "Approach": 1.3,
            "ShortGame": 0.9,
            "Putting": 0.8,
            "Consistency": 1.2,
            "DogFactor": 1.0,
        },
        "scale": 0.09,
    },
}


# =========================
# CORE FUNCTIONS
# =========================

def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.astype(str).str.strip()

    # sanity check
    for col in SG_FEATURES + ATTR_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' missing from master_player_ratings.csv")

    return df


def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Z_* columns for SG and attribute features across the whole field.
    """
    df = df.copy()
    for col in SG_FEATURES + ATTR_FEATURES:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            # if somehow zero variance, just set z = 0
            df[f"Z_{col}"] = 0.0
        else:
            df[f"Z_{col}"] = (df[col] - mean) / std
    return df


def compute_course_fit_for_profile(df: pd.DataFrame, profile_key: str) -> pd.DataFrame:
    """
    Given a profile key (from COURSE_PROFILES), compute:
      CF_Z: course fit score in z-space
      CF_SG_per_round: expected SG per round from fit
    """
    if profile_key not in COURSE_PROFILES:
        raise ValueError(f"Unknown course profile: {profile_key}")

    profile = COURSE_PROFILES[profile_key]
    sg_w = profile["sg_weights"]
    attr_w = profile["attr_weights"]
    scale = profile["scale"]

    df = df.copy()

    # SG contribution
    cf_sg = np.zeros(len(df), dtype=float)
    for feat, weight in sg_w.items():
        z_col = f"Z_{feat}"
        if z_col not in df.columns:
            raise ValueError(f"Missing z-score column: {z_col}")
        cf_sg += weight * df[z_col].to_numpy(dtype=float)

    # Attribute contribution
    cf_attr = np.zeros(len(df), dtype=float)
    for feat, weight in attr_w.items():
        z_col = f"Z_{feat}"
        if z_col not in df.columns:
            raise ValueError(f"Missing z-score column: {z_col}")
        cf_attr += weight * df[z_col].to_numpy(dtype=float)

    cf_z = cf_sg + cf_attr
    cf_sg_per_round = scale * cf_z

    out = df[[
        "player_name",
        "mu",
        "sigma",
        "form",
        "sg_ott",
        "sg_app",
        "sg_arg",
        "sg_putt",
        "OverallRating",
        "Tier",
    ] + ATTR_FEATURES].copy()

    out["CF_Z"] = cf_z
    out["CF_SG_per_round"] = cf_sg_per_round
    out["course_profile"] = profile_key
    out["course_description"] = profile["description"]

    # Sort by best course fit (highest CF_SG_per_round)
    out = out.sort_values("CF_SG_per_round", ascending=False).reset_index(drop=True)
    return out


def main():
    df = load_master()
    df = add_z_scores(df)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    for profile_key in COURSE_PROFILES.keys():
        fit_df = compute_course_fit_for_profile(df, profile_key)

        out_path = DATA_PROCESSED / f"course_fit_{profile_key}.csv"
        fit_df.to_csv(out_path, index=False)

        print(f"Saved course fit for '{profile_key}' → {out_path}")
        print(fit_df.head(5))
        print("-" * 60)


if __name__ == "__main__":
    main()
