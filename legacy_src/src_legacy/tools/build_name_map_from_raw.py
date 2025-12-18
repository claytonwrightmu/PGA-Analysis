# src/tools/build_name_map_from_raw.py

import pathlib
import re
from collections import defaultdict
import pandas as pd
import difflib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

DATA_RAW_DIRS = [
    PROJECT_ROOT / "Data" / "raw_stats",
    PROJECT_ROOT / "Data" / "raw",
    PROJECT_ROOT / "data" / "raw",
]

DATA_PLAYERS = PROJECT_ROOT / "Data" / "players"
if not DATA_PLAYERS.exists():
    DATA_PLAYERS = PROJECT_ROOT / "data" / "players"

PLAYER_NAME_CANDIDATE_COLS = [
    "player",
    "player_name",
    "golfer",
    "name",
    "Player",
    "PLAYER",
    "Golfer",
    "PLAYER NAME",
    "Player Name",
    "playerName",
]


def find_player_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in PLAYER_NAME_CANDIDATE_COLS:
            cols.append(col)
        else:
            low = str(col).lower()
            if "player" in low or "golfer" in low:
                cols.append(col)
    return cols


def normalize_raw_name(raw: str) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    name = str(raw).strip()

    name = re.sub(r"\s+", " ", name)

    # Remove suffixes (common)
    name = re.sub(r"\b(JR|SR|II|III|IV)\b\.?$", "", name, flags=re.IGNORECASE).strip()

    # Handle "Last, First" -> "First Last"
    if "," in name:
        parts = [p.strip() for p in name.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            name = f"{parts[1]} {parts[0]}".strip()

    name = re.sub(r"\s+", " ", name).strip()
    return name


def gather_csv_paths() -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for d in DATA_RAW_DIRS:
        if d.exists():
            paths.extend(list(d.rglob("*.csv")))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def collect_all_raw_names() -> pd.DataFrame:
    all_names = set()
    csv_paths = gather_csv_paths()

    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        player_cols = find_player_columns(df)
        if not player_cols:
            continue

        for col in player_cols:
            series = df[col].dropna().astype(str).str.strip()
            for v in series.unique().tolist():
                if v and v.lower() not in {"nan", "none"}:
                    all_names.add(v)

    return pd.DataFrame({"raw_name": sorted(all_names)})


def build_clusters(raw_names: list[str]) -> dict[str, list[str]]:
    """
    Cluster name variants by normalized form first.
    """
    clusters = defaultdict(list)
    for rn in raw_names:
        clusters[normalize_raw_name(rn)].append(rn)
    return clusters


def pick_canonical_from_cluster(cluster: list[str]) -> str:
    """
    Pick the best canonical name from a cluster:
    - prefer already "First Last" style (no comma)
    - prefer shortest cleaned string
    """
    cleaned = [normalize_raw_name(x) for x in cluster if normalize_raw_name(x)]
    if not cleaned:
        return ""
    no_comma = [c for c in cleaned if "," not in c]
    pool = no_comma if no_comma else cleaned
    return min(pool, key=lambda x: len(x))


def build_name_map():
    DATA_PLAYERS.mkdir(parents=True, exist_ok=True)

    name_map_path = DATA_PLAYERS / "name_map.csv"
    suggestions_path = DATA_PLAYERS / "name_map_suggestions.csv"

    df = collect_all_raw_names()
    if df.empty:
        print("[WARN] No player-like names found in raw CSV directories.")
        return

    raw_names = df["raw_name"].tolist()
    clusters = build_clusters(raw_names)

    # Canonical guess for each raw_name based on its normalized cluster
    canonical_guess = []
    for rn in raw_names:
        norm = normalize_raw_name(rn)
        cluster = clusters.get(norm, [rn])
        canonical_guess.append(pick_canonical_from_cluster(cluster))

    df["canonical_name"] = canonical_guess

    # Extra: fuzzy assist for near-duplicates that normalization didn't catch
    # (only if it looks safe)
    canonicals = sorted(set(df["canonical_name"].dropna().astype(str).str.strip()))
    fixed = []
    for c in df["canonical_name"].astype(str).str.strip().tolist():
        if c in canonicals:
            fixed.append(c)
            continue
        m = difflib.get_close_matches(c, canonicals, n=1, cutoff=0.93)
        fixed.append(m[0] if m else c)
    df["canonical_name"] = fixed

    df.to_csv(suggestions_path, index=False)
    if not name_map_path.exists():
        df[["raw_name", "canonical_name"]].to_csv(name_map_path, index=False)
        print(f"[OK] Created {name_map_path}")
    else:
        print(f"[INFO] {name_map_path} already exists (not overwriting).")

    print(f"[OK] Wrote {suggestions_path}")


if __name__ == "__main__":
    build_name_map()
