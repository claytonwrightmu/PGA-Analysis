"""
player_talent_engine.py

Builds a per-player "talent table" from the merged strokes-gained data.

Outputs one row per player with:
- mu          = multi-year true-talent estimate (strokes gained per round)
- sigma       = volatility / uncertainty proxy
- form        = recent-form hook (currently == mu)
- TalentScore = same as mu (for clarity in outputs)
- Tier        = S/A/B/C/D buckets from TalentScore
- Overall2K   = 60–99 video-game style overall rating
- Driving2K, Approach2K, ShortGame2K, Putting2K, Consistency2K, DogFactor2K
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import re

import numpy as np
import pandas as pd

from .master_table import build_master_player_table

# Where to save outputs by default
DATA_PROCESSED_DIR = Path("Data/Players/processed")
DEFAULT_OUTPUT_PATH = DATA_PROCESSED_DIR / "player_talent_table.csv"

# Base strokes-gained components we care about
SG_COMPONENT_KEYS = {
    "ott": "sg_off_the_tee",
    "app": "sg_approach_the_green",
    "arg": "sg_around_the_green",
    "putt": "sg_putting",
}

# How much each SG component matters for "talent"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "ott": 0.30,
    "app": 0.45,
    "arg": 0.15,
    "putt": 0.10,
}


def _find_sg_cols(df: pd.DataFrame, base_key: str) -> Dict[str, float]:
    """
    Find all columns in df that correspond to a given SG component
    across multiple years, and assign weights by year.

    Example matches:
      'sg_off_the_tee__avg'
      'sg_off_the_tee_2024__avg'
      'sg_off_the_tee_2023__avg'
    """
    matches = [c for c in df.columns if base_key in c and "avg" in c]

    col_weights: Dict[str, float] = {}
    for col in matches:
        m = re.search(r"(20[0-9]{2})", col)
        if m:
            year = int(m.group(1))
            if year >= 2025:
                w = 1.0
            elif year == 2024:
                w = 0.6
            elif year == 2023:
                w = 0.3
            else:
                w = 0.1
        else:
            # no explicit year -> treat as "current" season
            w = 1.0
        col_weights[col] = w

    return col_weights


def _compute_sg_component_tss(df: pd.DataFrame, base_key: str) -> pd.Series:
    """
    For a given SG component (OTT/APP/ARG/PUTT), compute a multi-year
    true-skill estimate using a weighted blend of all matching columns.
    """
    col_weights = _find_sg_cols(df, base_key)
    if not col_weights:
        # component missing entirely; return zeros
        return pd.Series(0.0, index=df.index)

    cols = list(col_weights.keys())
    weights = np.array([col_weights[c] for c in cols], dtype=float)

    # handle NaNs by treating them as 0 for the blend
    values = df[cols].fillna(0).to_numpy()

    # weighted average across the matched columns
    weighted_sum = values @ weights
    denom = weights.sum()
    tss = weighted_sum / denom if denom > 0 else weighted_sum

    return pd.Series(tss, index=df.index)


def _add_mu_multi_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `mu` column as a multi-year true talent estimate.

    Steps:
    - Build blended OTT/APP/ARG/PUTT TSS components.
    - Combine them using DEFAULT_WEIGHTS.
    """
    df = df.copy()

    # Compute blended components
    tss_components: Dict[str, pd.Series] = {}
    for key, base in SG_COMPONENT_KEYS.items():
        tss_components[key] = _compute_sg_component_tss(df, base)

    # Combine into overall mu
    mu = np.zeros(len(df), dtype=float)
    for key, series in tss_components.items():
        w = DEFAULT_WEIGHTS.get(key, 0.0)
        mu += w * series.to_numpy()

    df["mu"] = mu

    # Also keep individual components for inspection
    for key, series in tss_components.items():
        df[f"tss_{key}"] = series

    return df


def _add_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `sigma` column as a simple volatility proxy.

    For now: std dev across the four TSS components.
    """
    df = df.copy()
    comp_cols = [c for c in df.columns if c.startswith("tss_")]

    if not comp_cols:
        df["sigma"] = 1.0
        return df

    df["sigma"] = df[comp_cols].std(axis=1)
    return df


def _add_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `form` column.

    Right now we don't have round-by-round data wired up,
    so we'll set form == mu as a placeholder.
    """
    df = df.copy()
    if "mu" not in df.columns:
        raise ValueError("mu must be computed before form.")
    df["form"] = df["mu"]
    return df


def _add_tiers(df, num_tiers: int = 5) -> pd.DataFrame:
    """
    Assign tier labels S, A, B, C, D based on percentile rank of TalentScore.

    Top 20%  -> S
    Next 20% -> A
    Next 20% -> B
    Next 20% -> C
    Bottom 20% -> D
    """
    df = df.copy()

    if "TalentScore" not in df.columns:
        raise ValueError("TalentScore must exist before tiering.")

    # Rank players: 1 = best TalentScore
    n = len(df)
    rank = df["TalentScore"].rank(ascending=False, method="first")
    frac = rank / n  # 0–1, lower = better

    tiers = []
    for f in frac:
        if f <= 0.20:
            tiers.append("S")
        elif f <= 0.40:
            tiers.append("A")
        elif f <= 0.60:
            tiers.append("B")
        elif f <= 0.80:
            tiers.append("C")
        else:
            tiers.append("D")

    df["Tier"] = tiers
    return df


def _scale_to_2k(series: pd.Series, floor: int = 60, cap: int = 99) -> pd.Series:
    """
    Scale a numeric series into a 2K-style rating between [floor, cap].
    """
    s_min = float(series.min())
    s_max = float(series.max())
    span = s_max - s_min if s_max != s_min else 1.0

    scaled = floor + (series - s_min) / span * (cap - floor)
    return scaled.clip(floor, cap).round(0).astype(int)


def _add_overall_and_attributes(talent: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
    - Overall2K (60–99)
    - Driving2K, Approach2K, ShortGame2K, Putting2K
    - Consistency2K (inverse sigma)
    - DogFactor2K (blend of Overall + Consistency)
    """
    talent = talent.copy()

    # Overall rating based on TalentScore
    talent["Overall2K"] = _scale_to_2k(talent["TalentScore"])

    # Component attributes from tss_*
    comp_map = {
        "tss_ott": "Driving2K",
        "tss_app": "Approach2K",
        "tss_arg": "ShortGame2K",
        "tss_putt": "Putting2K",
    }
    for comp_col, out_col in comp_map.items():
        if comp_col in talent.columns:
            talent[out_col] = _scale_to_2k(talent[comp_col])
        else:
            talent[out_col] = 75  # neutral placeholder if missing

    # Consistency: invert sigma (lower sigma = more consistent)
    if "sigma" in talent.columns:
        sigma = talent["sigma"]
        sigma_inv = sigma.max() - sigma
        talent["Consistency2K"] = _scale_to_2k(sigma_inv)
    else:
        talent["Consistency2K"] = 75

    # Dog Factor: blend Overall + Consistency
    overall_norm = (talent["Overall2K"] - 60) / (99 - 60)
    cons_norm = (talent["Consistency2K"] - 60) / (99 - 60)
    dog_raw = 0.6 * overall_norm + 0.4 * cons_norm
    talent["DogFactor2K"] = _scale_to_2k(dog_raw)

    return talent


def build_talent_table(
    save_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Build the full player talent table.

    Steps:
    1. Build the merged master player table from raw stats.
    2. Add multi-year mu (talent), sigma (volatility), form (recent form hook).
    3. Add TalentScore + Tiers.
    4. Add 2K overall + attribute ratings.
    5. Return a clean DataFrame and optionally save to CSV.
    """
    # 1) Build/refresh master table from raw Excel
    master = build_master_player_table()

    # 2) Add features
    df = _add_mu_multi_year(master)
    df = _add_sigma(df)
    df = _add_form(df)

    # 3) Select core columns
    core_cols = ["player", "player_id", "mu", "sigma", "form"]
    existing_core = [c for c in core_cols if c in df.columns]

    talent = (
        df[existing_core + [c for c in df.columns if c.startswith("tss_")]]
        .sort_values("mu", ascending=False)
        .reset_index(drop=True)
    )

    # 4) Add TalentScore + Tiers
    talent["TalentScore"] = talent["mu"]
    talent = _add_tiers(talent, num_tiers=5)

    # 5) Add Overall2K + attribute ratings
    talent = _add_overall_and_attributes(talent)

    # 6) Save main talent table
    if save_path is not None:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        talent.to_csv(save_path, index=False)

    # Also write compact ratings / attributes tables
    ratings_path = DATA_PROCESSED_DIR / "player_ratings.csv"
    ratings_cols = [
        "player",
        "player_id",
        "Overall2K",
        "Tier",
        "TalentScore",
        "mu",
        "sigma",
        "form",
    ]
    existing_ratings = [c for c in ratings_cols if c in talent.columns]
    talent[existing_ratings].to_csv(ratings_path, index=False)

    attrs_path = DATA_PROCESSED_DIR / "player_attributes.csv"
    attr_cols = [
        "player",
        "player_id",
        "Overall2K",
        "Driving2K",
        "Approach2K",
        "ShortGame2K",
        "Putting2K",
        "Consistency2K",
        "DogFactor2K",
        "Tier",
    ]
    existing_attrs = [c for c in attr_cols if c in talent.columns]
    talent[existing_attrs].to_csv(attrs_path, index=False)

    return talent
