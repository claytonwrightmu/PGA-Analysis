# src/build_master_ratings.py

import pathlib
import re
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

RAW_STATS_DIR = PROJECT_ROOT / "Players" / "raw_stats"
OUT_DIR = PROJECT_ROOT / "Data" / "players"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "master_player_ratings.csv"


# ========= name cleaning (same idea as before, no name_map needed) =========
def clean_name(x: str) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()

    # remove common suffixes
    s = re.sub(r"\b(JR|SR|II|III|IV)\b\.?$", "", s, flags=re.IGNORECASE).strip()

    # "Last, First" -> "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


# ========= robust excel reader =========
def read_stat_xlsx(path: pathlib.Path) -> pd.DataFrame:
    """
    Reads first sheet, finds player column + numeric stat column automatically.
    Returns: player_name, value
    """
    df = pd.read_excel(path)

    # find player column
    player_candidates = [c for c in df.columns if "player" in str(c).lower() or "name" in str(c).lower()]
    player_col = player_candidates[0] if player_candidates else df.columns[0]

    # find numeric column (pick the last numeric-like column)
    numeric_cols = []
    for c in df.columns:
        if c == player_col:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= max(5, int(0.3 * len(df))):
            numeric_cols.append(c)

    if not numeric_cols:
        raise ValueError(f"No numeric stat column found in {path.name}. Columns={list(df.columns)}")

    value_col = numeric_cols[-1]

    out = df[[player_col, value_col]].copy()
    out.columns = ["player_name_raw", "value"]
    out["player_name"] = out["player_name_raw"].apply(clean_name)
    out = out.dropna(subset=["player_name"])
    out = out[out["player_name"].astype(str).str.len() > 0]
    return out[["player_name", "value"]]


def build_master():
    files = sorted(RAW_STATS_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx found in {RAW_STATS_DIR}")

    master = None

    for f in files:
        stat_key = f.stem
        df = read_stat_xlsx(f)
        df.rename(columns={"value": stat_key}, inplace=True)
        df[stat_key + "_z"] = zscore(df[stat_key])

        keep = ["player_name", stat_key, stat_key + "_z"]
        df = df[keep]

        master = df if master is None else master.merge(df, on="player_name", how="outer")

    # keep players with at least 3 z-metrics
    z_cols = [c for c in master.columns if c.endswith("_z")]
    master["num_metrics"] = master[z_cols].notna().sum(axis=1)
    master = master[master["num_metrics"] >= 3].drop(columns=["num_metrics"])

    master.to_csv(OUT_PATH, index=False)
    print(f"[OK] wrote {OUT_PATH} | players={len(master)} | cols={len(master.columns)}")


if __name__ == "__main__":
    build_master()
