from pathlib import Path
import re
import pandas as pd

from .player_leaderboards import load_all_leaderboards

PROCESSED_DIR = Path("Data/Players/processed")


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


def build_master_player_table(save: bool = True,
                              filename: str = "master_player_stats.csv") -> pd.DataFrame:
    """
    Merge all leaderboards in Data/Players/raw into a single player-level table.

    - Uses player_id + player as join keys.
    - Keeps all stat columns, but prefixes them with the leaderboard key.
      Example: 'sg_off_the_tee__avg', 'gir_150_175_percentage__pct', etc.

    Returns:
        pandas.DataFrame with one row per player and many stat columns.
    """
    leaderboards = load_all_leaderboards()

    base_cols = ["player_id", "player"]
    ignore_cols = base_cols + ["rank", "movement"]

    master: pd.DataFrame | None = None

    for key, df in leaderboards.items():
        # Ensure we have the ID + name columns
        if not all(c in df.columns for c in base_cols):
            # If a file doesn't have player_id/player for some reason, skip it
            continue

        temp = df.copy()

        # Figure out which columns are actual stats
        stat_cols = [c for c in temp.columns if c not in ignore_cols]
        if not stat_cols:
            continue

        # Rename stat columns with a prefix based on the file key
        rename_map = {}
        for col in stat_cols:
            norm_col = _normalize_stat_col(col)
            new_name = f"{key}__{norm_col}"
            rename_map[col] = new_name

        temp = temp[base_cols + stat_cols].rename(columns=rename_map)

        if master is None:
            master = temp
        else:
            master = pd.merge(master, temp, on=base_cols, how="outer")

    if master is None or master.empty:
        raise ValueError("No leaderboards were merged. Check that files loaded correctly.")

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / filename
        master.to_csv(out_path, index=False)
        print(f"Saved master table to {out_path}")

    return master
