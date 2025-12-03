"""
player_talent_engine.py

Builds a per-player "talent table" from the merged strokes-gained data.

Outputs one row per player with:
- mu    = overall true-talent estimate (strokes gained per round)
- sigma = volatility / uncertainty proxy
- form  = recent-form hook (currently == mu, easy to upgrade later)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

from .master_table import build_master_player_table

# Where to save outputs by default
DATA_PROCESSED_DIR = Path("Data/Players/processed")
DEFAULT_OUTPUT_PATH = DATA_PROCESSED_DIR / "player_talent_table.csv"

# Strokes-gained columns we care about
SG_COLS = [
    "sg_off_the_tee__avg",
    "sg_approach_the_green__avg",
    "sg_around_the_green__avg",
    "sg_putting__avg",
]

# How much each SG component matters for "talent"
DEFAULT_WEIGHTS: Dict[str, float] = {
    "sg_approach_the_green__avg": 0.45,
    "sg_off_the_tee__avg": 0.30,
    "sg_around_the_green__avg": 0.15,
    "sg_putting__avg": 0.10,
}


def _compute_mu(row: pd.Series, weights: Dict[str, float]) -> float:
    """Weighted strokes-gained talent score for one player."""
    total = 0.0
    for col, w in weights.items():
        if col in row and pd.notna(row[col]):
            total += w * float(row[col])
    return total


def _add_mu(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """Add `mu` column to the master player table."""
    df = df.copy()
    df["mu"] = df.apply(_compute_mu, axis=1, weights=weights)
    return df


def _add_sigma(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `sigma` column as a simple volatility proxy.

    For now: std dev across the four SG components.
    This isn't perfect round-to-round volatility, but it
    gives us a relative "spikiness" measure per player.
    """
    df = df.copy()
    available_cols = [c for c in SG_COLS if c in df.columns]

    if not available_cols:
        # Fallback: everyone gets same sigma if we somehow lost SG cols
        df["sigma"] = 1.0
        return df

    df["sigma"] = df[available_cols].std(axis=1)
    return df


def _add_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `form` column.

    Right now we don't have round-by-round data wired up,
    so we'll set form == mu as a placeholder.

    Later you can upgrade this to use:
    - last N rounds
    - last N events
    - major-only performance, etc.
    """
    df = df.copy()
    if "mu" not in df.columns:
        raise ValueError("mu must be computed before form.")
    df["form"] = df["mu"]
    return df


def build_talent_table(
    save_path: Optional[Path] = DEFAULT_OUTPUT_PATH,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Build the full player talent table.

    Steps:
    1. Build the merged master player table from raw stats.
    2. Add mu (talent), sigma (volatility), form (recent form hook).
    3. Return a clean DataFrame and optionally save to CSV.
    """
    # 1) Build/refresh master table from raw Excel
    master = build_master_player_table()

    # 2) Add features
    df = _add_mu(master, weights=weights)
    df = _add_sigma(df)
    df = _add_form(df)

    # 3) Select core columns
    core_cols = ["player", "player_id", "mu", "sigma", "form"]
    existing_core = [c for c in core_cols if c in df.columns]

    talent = (
        df[existing_core]
        .sort_values("mu", ascending=False)
        .reset_index(drop=True)
    )

    # 4) Save if requested
    if save_path is not None:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        talent.to_csv(save_path, index=False)

    return talent
