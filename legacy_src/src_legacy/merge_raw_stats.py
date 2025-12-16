import pandas as pd
from pathlib import Path
import pandas.api.types as ptypes

ROOT = Path(__file__).resolve().parents[1]
RAW_STATS = ROOT / "Data" / "Players" / "raw_stats"
MASTER = ROOT / "Data" / "Players" / "processed" / "master_player_ratings.csv"

# Map raw filenames → new column name in master table
# (you can add more filenames here over time)
FILE_MAP = {
    # core stuff
    "Ball Speed Leaders.xlsx": "ball_speed",
    "Driving Accuracy Leaders.xlsx": "driving_accuracy",
    "GIR <125 Percentage.xlsx": "gir_lt_125",
    "GIR 125-150 Percentage.xlsx": "gir_125_150",
    "GIR 150-175 Percentage.xlsx": "gir_150_175",
    "GIR 175-200 Percentage.xlsx": "gir_175_200",
    "GIR 200+ Percentage.xlsx": "gir_200_plus",
    "GIR Percentage.xlsx": "gir_overall",

    # fairway / rough GIR + birdies
    "gir_%_fairway.xlsx": "gir_fairway_pct",
    "gir_%_not_fairway.xlsx": "gir_nonfairway_pct",
    "birdie%_from_fairway.xlsx": "birdie_fairway_pct",
    "birdie%_from_rough.xlsx": "birdie_rough_pct",

    # going for par 5s
    "going_for_it_%.xlsx": "go_for_it_pct",
    "going_for_it_hit_pct.xlsx": "go_for_it_hit_pct",

    # proximity / rough proximity
    "fairway_proximity.xlsx": "fairway_proximity",
    "left_rough_proximity.xlsx": "left_rough_proximity",
    "right_rough_proximity.xlsx": "right_rough_proximity",
    "Proximity to hole (Approach).xlsx": "proximity_approach",

    # putting buckets + 3-putts
    "Inside 10 ft percentage.xlsx": "putt_inside_10_pct",
    "putting_inside_5'.xlsx": "putt_0_5",
    "putting_5-10'.xlsx": "putt_5_10",
    "putting_10-15'.xlsx": "putt_10_15",
    "putting_15-20'.xlsx": "putt_15_20",
    "putting_20-25'.xlsx": "putt_20_25",
    "putting_>25'.xlsx": "putt_25_plus",
    "putting_3_putts.xlsx": "three_putt_rate",

    # par scoring + overall scoring
    "Par 3 average leaders.xlsx": "par3_scoring",
    "Par 4 average leaders.xlsx": "par4_scoring",
    "Par 5 average leaders.xlsx": "par5_scoring",
    "Scoring average leaders.xlsx": "scoring_avg",

    # short game
    "Sand save percentage.xlsx": "sand_save_pct",
    "Scrambling Leaders.xlsx": "scrambling_pct",

    # strokes gained breakdowns
    "SG Approach the green.xlsx": "sg_app_raw",
    "SG Around the green.xlsx": "sg_arg_raw",
    "SG Off the tee.xlsx": "sg_ott_raw",
    "SG Putting.xlsx": "sg_putt_raw",
    "SG Tee to green.xlsx": "sg_t2g_raw",
}

def clean_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    out = name.strip()
    # light cleanup – we can add more rules later if we see mismatches
    for suffix in [" Jr.", " Jr", " III", " II", ", Jr."]:
        out = out.replace(suffix, "")
    return out

def pick_stat_column(df: pd.DataFrame, player_col: str) -> str:
    """
    Heuristic: choose the first numeric column that isn't the player column.
    Most leaderboards are like: [Rank, Player, Stat, ...].
    """
    numeric_cols = [
        c for c in df.columns
        if c != player_col and ptypes.is_numeric_dtype(df[c])
    ]
    if numeric_cols:
        return numeric_cols[0]
    # fallback: last column
    return df.columns[-1]

def find_player_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    # 1) Prefer columns with 'name' in them
    name_cols = [c for c in cols if "name" in c.lower()]
    # 2) Then columns with 'player' but not 'id'
    player_cols = [c for c in cols if "player" in c.lower() and "id" not in c.lower()]

    candidates = name_cols + [c for c in player_cols if c not in name_cols]

    # Prefer ones that are text/object
    for c in candidates:
        if df[c].dtype == "object":
            return c

    if candidates:
        return candidates[0]

    raise ValueError("No suitable player/name column found")

def main():
    print("Loading master_player_ratings.csv …")
    master = pd.read_csv(MASTER)
    master["player_name"] = master["player_name"].apply(clean_name)

    for filename, new_col in FILE_MAP.items():
        path = RAW_STATS / filename
        if not path.exists():
            print(f"[SKIP] {filename} not found in raw_stats.")
            continue

        print(f"[MERGING] {filename} → {new_col}")

        df = pd.read_excel(path)
        df.columns = df.columns.str.strip()

               # find player column (prefer real names, not IDs)
        try:
            player_col = find_player_column(df)
        except ValueError:
            print(f"  [WARN] No player/name column found in {filename}, skipping.")
            continue


        df[player_col] = df[player_col].apply(clean_name)

        stat_col = pick_stat_column(df, player_col)
        print(f"  using column '{stat_col}' as stat for {new_col}")

        merge_df = df[[player_col, stat_col]].rename(
            columns={player_col: "player_name", stat_col: new_col}
        )

        master = master.merge(merge_df, on="player_name", how="left")

    # fill any missing values from unmatched players with 0
    master = master.fillna(0.0)

    master.to_csv(MASTER, index=False)
    print("✅ Updated master_player_ratings.csv with new stat columns.")
    print("New columns now include:")
    print([c for c in master.columns if any(c == v for v in FILE_MAP.values())])

if __name__ == "__main__":
    main()
