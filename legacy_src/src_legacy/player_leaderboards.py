from pathlib import Path
import re
import pandas as pd

RAW_PLAYER_DIR = Path("Data/Players/raw")


def _make_key_from_path(path: Path) -> str:
    key = path.stem.lower()

    replacements = {
        "(": "",
        ")": "",
        ":": "",
        "%": "pct",
        "+": "plus",
        "<": "lt",
        ">": "gt",
        "/": "_",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)

    key = key.replace(" ", "_")
    key = re.sub(r"_+", "_", key).strip("_")
    return key


def load_all_leaderboards():
    """
    Loads all Excel files in Data/Players/raw.
    Returns {clean_key: DataFrame}
    """
    if not RAW_PLAYER_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {RAW_PLAYER_DIR}")

    excel_files = list(RAW_PLAYER_DIR.glob("*.xlsx"))
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {RAW_PLAYER_DIR}")

    leaderboards = {}

    for path in excel_files:
        key = _make_key_from_path(path)
        df = pd.read_excel(path)

        df = df.rename(columns={
            "RANK": "rank",
            "MOVEMENT": "movement",
            "PLAYER_ID": "player_id",
            "PLAYER": "player",
        })

        leaderboards[key] = df

    return leaderboards
