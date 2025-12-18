import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class BayesianTalentEstimator:
    def __init__(
        self,
        prior_mean: float = 0.0,
        prior_std: float = 0.5,
        base_sigma: float = 0.3,
        sigma_decay: float = 0.01
    ):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.base_sigma = base_sigma
        self.sigma_decay = sigma_decay

    def estimate_player_talent(
        self,
        player_stats: pd.DataFrame,
        stat_columns: Dict[str, str] = None,
        round_variance: float = 6.25
    ) -> pd.DataFrame:
        if stat_columns is None:
            stat_columns = {
                "sg_ott": "sg_ott",
                "sg_app": "sg_app",
                "sg_arg": "sg_arg",
                "sg_putt": "sg_putt",
            }

        df = player_stats.copy()

        prior_var = self.prior_std ** 2

        for stat_name, col_name in stat_columns.items():
            if col_name not in df.columns:
                logger.warning(f"Column {col_name} not found")
                continue

            rounds_col = f"{stat_name}_rounds"
            if rounds_col not in df.columns:
                rounds_col = "total_rounds"

            if rounds_col not in df.columns:
                logger.warning(f"No rounds data for {stat_name}")
                continue

            observed = df[col_name].astype(float)
            n = df[rounds_col].astype(float)

            valid = (~observed.isna()) & (~n.isna()) & (n > 0)

            talent = pd.Series(np.nan, index=df.index, dtype=float)
            unc = pd.Series(np.nan, index=df.index, dtype=float)
            shrink_pct = pd.Series(np.nan, index=df.index, dtype=float)

            measurement_var = pd.Series(np.nan, index=df.index, dtype=float)
            measurement_var.loc[valid] = round_variance / n.loc[valid]

            shrinkage = pd.Series(np.nan, index=df.index, dtype=float)
            shrinkage.loc[valid] = prior_var / (prior_var + measurement_var.loc[valid])

            post_mean = pd.Series(np.nan, index=df.index, dtype=float)
            post_mean.loc[valid] = shrinkage.loc[valid] * observed.loc[valid] + (1 - shrinkage.loc[valid]) * self.prior_mean

            post_var = pd.Series(np.nan, index=df.index, dtype=float)
            post_var.loc[valid] = shrinkage.loc[valid] * measurement_var.loc[valid]
            post_std = np.sqrt(post_var)

            exp_factor = np.exp(-self.sigma_decay * n)
            post_std = np.sqrt(post_std**2 + (self.base_sigma * exp_factor) ** 2)

            talent.loc[valid] = post_mean.loc[valid]
            unc.loc[valid] = post_std.loc[valid]

            denom = (observed - self.prior_mean).abs() + 1e-10
            shrink_pct.loc[valid] = (talent.loc[valid] - observed.loc[valid]).abs() / denom.loc[valid]

            df[f"{stat_name}_talent"] = talent
            df[f"{stat_name}_uncertainty"] = unc
            df[f"{stat_name}_shrinkage_pct"] = shrink_pct

        return df
# --- compatibility export (pipeline expects this name) ---

def estimate_player_talents(player_data, prior_mean: float = 0.0, prior_std: float = 0.5):
    """
    Backwards-compatible wrapper.
    Pipeline imports estimate_player_talents, so we expose it here.

    If you already have a function with a different name, call it here.
    """
    estimator = BayesianTalentEstimator(prior_mean=prior_mean, prior_std=prior_std)
    df = estimator.estimate_player_talent(player_data)
    df = estimator.calculate_overall_talent(df)
    df = estimator.create_talent_tiers(df)
    return df
