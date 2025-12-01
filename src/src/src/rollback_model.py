import pandas as pd

def simulate_rollback(
    df_players: pd.DataFrame,
    yards_lost: float = 15.0,
    sensitivity_off_tee: float = 0.02,
) -> pd.DataFrame:
    """
    Simulate the impact of a rollback ball on player strokes gained.

    Parameters
    ----------
    df_players : DataFrame
        Must include:
          - driving_distance_avg
          - sg_off_tee_per_round
          - sg_total_per_round
    yards_lost : float
        How many yards of average driving distance a typical player loses.
    sensitivity_off_tee : float
        Approximate SG_off_tee lost per yard of distance lost.

    Returns
    -------
    DataFrame
        Copy of df_players with additional columns:
          - driving_distance_rollback
          - sg_off_tee_per_round_rollback
          - sg_total_per_round_rollback_est
          - rollback_delta_sg_total
    """
    df = df_players.copy()

    for col in ["driving_distance_avg", "sg_off_tee_per_round", "sg_total_per_round"]:
        if col not in df.columns:
            raise ValueError(f"{col} column is required for rollback simulation.")

    # Simple distance loss (same for everyone to start)
    df["driving_distance_rollback"] = df["driving_distance_avg"] - yards_lost

    # Assume off-tee SG drops linearly with distance loss
    sg_ott_delta = yards_lost * sensitivity_off_tee
    df["sg_off_tee_per_round_rollback"] = df["sg_off_tee_per_round"] - sg_ott_delta

    # For v1, assume approach/putting unchanged – can be upgraded later.
    df["sg_total_per_round_rollback_est"] = df["sg_total_per_round"] - sg_ott_delta

    df["rollback_delta_sg_total"] = (
        df["sg_total_per_round_rollback_est"] - df["sg_total_per_round"]
    )

    return df
