import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "Data" / "Players" / "processed"
MASTER_PATH = DATA_PROCESSED / "master_player_ratings.csv"


def load_master():
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.strip()
    return df


def load_course_fit(course_profile_key: str):
    path = DATA_PROCESSED / f"course_fit_{course_profile_key}.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def simulate_tournament(
    df,
    n_rounds: int = 4,
    n_sims: int = 5000,
    form_weight: float = 0.3,
    mu_shrink: float = 0.3,
    cf_effect_scale: float = 0.5,
    sigma_scale: float = 3.0,
    min_sigma: float = 0.8,
    noise_sd: float = 0.6,
):
    """
    Simulate a stroke-play tournament with realistic (and generous) variance.

    - mu_shrink: shrink gaps in true talent toward field mean (0-1)
    - cf_effect_scale: scales down course-fit effect (0-1)
    - sigma_scale: multiplier on player sigma to increase randomness
    - min_sigma: floor on per-round sigma so nobody is 'too deterministic'
    - noise_sd: extra independent luck each round (weather, bounces, etc.)
    """

    n_players = len(df)

    mu_raw = df["mu"].to_numpy(dtype=float)
    cf_raw = df["CF_SG_per_round"].to_numpy(dtype=float)
    form = df["form"].to_numpy(dtype=float)
    sigma_raw = df["sigma"].to_numpy(dtype=float)

    # --- shrink mu gaps so #1 isn't totally untouchable ---
    mu_mean = mu_raw.mean()
    mu_shrunk = mu_mean + mu_shrink * (mu_raw - mu_mean)

    # --- tone down course-fit effect ---
    cf = cf_effect_scale * cf_raw

    # --- effective talent (per round) ---
    effective_mu = mu_shrunk + cf + form_weight * form

    # --- volatility (per round) ---
    sigma = np.maximum(min_sigma, sigma_raw * sigma_scale)

    win_counts = np.zeros(n_players, dtype=int)
    top5_counts = np.zeros(n_players, dtype=int)
    rank_sums = np.zeros(n_players, dtype=float)

    for _ in range(n_sims):
        base_perf = np.random.normal(
            loc=effective_mu,
            scale=sigma,
            size=(n_rounds, n_players),
        )

        noise = np.random.normal(
            loc=0.0,
            scale=noise_sd,
            size=(n_rounds, n_players),
        )

        perf = base_perf + noise
        total = perf.sum(axis=0)

        # best = highest total
        ranks = (-total).argsort()

        # winner
        win_counts[ranks[0]] += 1

        # top 5
        top5_counts[ranks[:5]] += 1

        # 1-based positions
        positions = np.empty_like(ranks)
        positions[ranks] = np.arange(1, n_players + 1)
        rank_sums += positions

    out = df[["player_name", "Tier", "TalentScore", "OverallRating"]].copy()
    out["win_pct"] = win_counts / n_sims * 100.0
    out["top5_pct"] = top5_counts / n_sims * 100.0
    out["avg_finish"] = rank_sums / n_sims

    out = out.sort_values("win_pct", ascending=False).reset_index(drop=True)
    return out


def main():
    # choose which archetype to simulate
    course_key = "augusta_masters"  # change this for other profiles

    print(f"Running tournament sim for course profile: {course_key}")

    master = load_master()
    course_fit = load_course_fit(course_key)

    # merge course fit into master table
    df = master.merge(
        course_fit[["player_name", "CF_SG_per_round"]],
        on="player_name",
        how="left",
    )

    # field: top 80 by TalentScore
    df = df.sort_values("TalentScore", ascending=False).head(80).reset_index(drop=True)

    results = simulate_tournament(df)

    out_path = DATA_PROCESSED / f"tournament_sim_{course_key}.csv"
    results.to_csv(out_path, index=False)

    print(f"Saved tournament simulation results to: {out_path}")
    print(results.head(10))


if __name__ == "__main__":
    main()
