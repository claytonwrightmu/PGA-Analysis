def clean_player_stats(df):
    """
    Ensure the player stats DataFrame contains required columns and drop rows
    that are missing those key values.
    """

    required = [
        "player_name",
        "driving_distance",
        "driving_accuracy_pct",
        "scoring_avg"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with missing critical stats
    return df.dropna(subset=required)
