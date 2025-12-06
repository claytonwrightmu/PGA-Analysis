from pathlib import Path
import re
import pandas as pd

from .player_leaderboards import load_all_leaderboards

PROCESSED_DIR = Path("Data/Players/processed")

# 2025 strokes-gained columns we care about
SG_COLS_2025 = [
    "sg_off_the_tee__avg",
    "sg_approach_the_green__avg",
    "sg_around_the_green__avg",
    "sg_putting__avg",
]


def _normalize_stat_col(col: str) -> str:
    """
    Turn raw stat column names like 'AVG', 'Pct', 'TOTAL STROKES'
    into snake_case names like 'avg', 'pct', 'total_strokes'.
    """
    col = col.strip().lower()
    replacements = {
        "(": "",
        ")": "",
        "%": "pct",
        "+": "plus",
        "/": "_",
        ":": "",
    }
    for old, new in replacements.items():
        col = col.replace(old, new)

    col = col.replace(" ", "_")
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def build_master_player_table(
    save: bool = True,
    filename: str = "master_player_stats.csv",
) -> pd.DataFrame:
    """
    Merge all leaderboards in Data/Players/raw into a single player-level table.

    - Uses player_id + player as join keys when available.
    - Keeps ALL players across all files initially (outer joins).
    - Prefixes stat columns with the leaderboard key.
    - THEN filters to players who have real 2025 SG stats.
    - Fills missing numeric values with column means.
    """
    leaderboards = load_all_leaderboards()

    base_cols = ["player_id", "player"]
    ignore_cols = base_cols + ["rank", "movement"]

    master: pd.DataFrame | None = None

    for key, df in leaderboards.items():
        temp = df.copy()

        # If player_id is missing in a file, fall back to just 'player'
        join_cols = [c for c in base_cols if c in temp.columns]
        if "player" not in join_cols:
            # Can't join without player name; skip this file
            continue

        # figure out which columns are actual stats
        stat_cols = [c for c in temp.columns if c not in ignore_cols]
        if not stat_cols:
            continue

        # rename stat columns with prefix based on file key
        rename_map = {}
        for col in stat_cols:
            norm_col = _normalize_stat_col(col)
            new_name = f"{key}__{norm_col}"
            rename_map[col] = new_name

        temp = temp[join_cols + stat_cols].rename(columns=rename_map)

        if master is None:
            master = temp
        else:
            master = pd.merge(master, temp, on=join_cols, how="outer")

    if master is None or master.empty:
        raise ValueError("No leaderboards were merged. Check that files loaded correctly.")

    # ------------- FILTER TO ACTIVE 2025 PLAYERS (BEFORE fillna) -------------
    # We consider a player "active" if they have ANY non-zero 2025 SG component.
    sg_cols_present = [c for c in SG_COLS_2025 if c in master.columns]
    if sg_cols_present:
        # Treat NaN as 0 *only for this mask*
        sg_vals = master[sg_cols_present].fillna(0)
        active_mask = (sg_vals.abs().sum(axis=1) > 0)
        before = len(master)
        master = master[active_mask].reset_index(drop=True)
        print(f"Filtered inactive players using 2025 SG: {before} -> {len(master)}")
    else:
        print("WARNING: No 2025 SG columns found; skipping active-player filter.")

    # ------------- NOW do numeric fill -------------
    numeric_cols = master.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) > 0:
        master[numeric_cols] = master[numeric_cols].fillna(master[numeric_cols].mean())

    # any leftover NaNs (strings, odd stuff) -> 0
    master = master.fillna(0)

    # save to processed CSV if requested
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / filename
        master.to_csv(out_path, index=False)
        print(f"Saved master table to {out_path}")

    return master


def build_master_table(
    save: bool = True,
    filename: str = "master_player_stats.csv",
) -> pd.DataFrame:
    """
    Thin wrapper so other modules can call `build_master_table`
    without worrying about the internal function name.
    """
    return build_master_player_table(save=save, filename=filename)
