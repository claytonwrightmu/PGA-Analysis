import pandas as pd

def add_player_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a coarse 'archetype' label for each player based on strokes gained
    and driving stats.

    Expected columns (per row = one player for a season):
      - sg_off_tee_per_round
      - sg_approach_per_round
      - sg_putting_per_round
      - driving_distance_avg
      - fairway_pct

    Returns a copy of the DataFrame with:
      - percentile columns for key stats
      - 'archetype' (str)
    """
    df = df.copy()

    # Percentiles for core dimensions
    for col in [
        "sg_off_tee_per_round",
        "sg_approach_per_round",
        "sg_putting_per_round",
        "driving_distance_avg",
        "fairway_pct",
    ]:
        if col in df.columns:
            df[f"{col}_pct"] = df[col].rank(pct=True)

    def classify_row(row) -> str:
        dist = row.get("driving_distance_avg_pct", 0.5)
        acc = row.get("fairway_pct_pct", 0.5)
        sg_ott = row.get("sg_off_tee_per_round_pct", 0.5)
        sg_app = row.get("sg_approach_per_round_pct", 0.5)
        sg_putt = row.get("sg_putting_per_round_pct", 0.5)

        if dist > 0.8 and sg_ott > 0.75 and acc < 0.4:
            return "High-variance bomber"
        if dist > 0.75 and acc > 0.5 and sg_app > 0.6:
            return "Elite modern ball-striker"
        if dist < 0.4 and sg_app > 0.7:
            return "Short but elite irons"
        if sg_putt > 0.75 and sg_app < 0.5:
            return "Putter-led scorer"
        if sg_ott > 0.6 and sg_app > 0.6:
            return "Balanced tee-to-green"
        return "Neutral / mixed profile"

    df["archetype"] = df.apply(classify_row, axis=1)
    return df
