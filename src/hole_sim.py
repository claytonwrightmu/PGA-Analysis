import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "Data" / "Players" / "processed"
MASTER_PATH = DATA_PROCESSED / "master_player_ratings.csv"
COURSE_DIR = ROOT / "Data" / "Courses"

# Base features
SG_FEATURES = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
ATTR_FEATURES = ["Power", "Approach", "ShortGame", "Putting", "Consistency", "DogFactor"]

# How each hole type weights your skills
HOLE_TYPE_WEIGHTS = {
    "long_par4": {
        "sg_weights": {"sg_ott": 1.1, "sg_app": 1.3, "sg_arg": 0.6, "sg_putt": 0.6},
        "attr_weights": {
            "Power": 1.0,
            "Approach": 1.3,
            "ShortGame": 0.5,
            "Putting": 0.6,
            "Consistency": 1.1,
            "DogFactor": 0.8,
        },
        "scale": 0.04,
    },
    "mid_par4": {
        "sg_weights": {"sg_ott": 1.0, "sg_app": 1.2, "sg_arg": 0.7, "sg_putt": 0.7},
        "attr_weights": {
            "Power": 0.9,
            "Approach": 1.2,
            "ShortGame": 0.7,
            "Putting": 0.7,
            "Consistency": 1.0,
            "DogFactor": 0.8,
        },
        "scale": 0.035,
    },
    "short_par4": {
        "sg_weights": {"sg_ott": 0.9, "sg_app": 1.1, "sg_arg": 0.8, "sg_putt": 0.9},
        "attr_weights": {
            "Power": 0.7,
            "Approach": 1.1,
            "ShortGame": 0.9,
            "Putting": 1.0,
            "Consistency": 0.9,
            "DogFactor": 1.0,
        },
        "scale": 0.035,
    },
    "drivable_par4": {
        "sg_weights": {"sg_ott": 1.2, "sg_app": 1.0, "sg_arg": 0.9, "sg_putt": 1.0},
        "attr_weights": {
            "Power": 1.3,
            "Approach": 1.0,
            "ShortGame": 1.0,
            "Putting": 1.0,
            "Consistency": 0.8,
            "DogFactor": 1.2,
        },
        "scale": 0.045,
    },
    "long_par3": {
        "sg_weights": {"sg_ott": 0.0, "sg_app": 1.4, "sg_arg": 0.8, "sg_putt": 0.8},
        "attr_weights": {
            "Power": 0.7,
            "Approach": 1.4,
            "ShortGame": 0.9,
            "Putting": 0.9,
            "Consistency": 1.1,
            "DogFactor": 0.9,
        },
        "scale": 0.04,
    },
    "mid_par3": {
        "sg_weights": {"sg_ott": 0.0, "sg_app": 1.2, "sg_arg": 0.9, "sg_putt": 0.9},
        "attr_weights": {
            "Power": 0.6,
            "Approach": 1.2,
            "ShortGame": 1.0,
            "Putting": 1.0,
            "Consistency": 1.0,
            "DogFactor": 0.9,
        },
        "scale": 0.035,
    },
    "short_par3_water": {
        "sg_weights": {"sg_ott": 0.0, "sg_app": 1.3, "sg_arg": 0.9, "sg_putt": 1.0},
        "attr_weights": {
            "Power": 0.5,
            "Approach": 1.3,
            "ShortGame": 1.0,
            "Putting": 1.1,
            "Consistency": 1.1,
            "DogFactor": 1.2,
        },
        "scale": 0.04,
    },
    "reachable_par5": {
        "sg_weights": {"sg_ott": 1.3, "sg_app": 1.2, "sg_arg": 1.0, "sg_putt": 0.9},
        "attr_weights": {
            "Power": 1.3,
            "Approach": 1.1,
            "ShortGame": 1.0,
            "Putting": 0.8,
            "Consistency": 0.9,
            "DogFactor": 1.1,
        },
        "scale": 0.045,
    },
    "reachable_par5_risky": {
        "sg_weights": {"sg_ott": 1.3, "sg_app": 1.2, "sg_arg": 1.1, "sg_putt": 0.9},
        "attr_weights": {
            "Power": 1.4,
            "Approach": 1.1,
            "ShortGame": 1.1,
            "Putting": 0.8,
            "Consistency": 0.8,
            "DogFactor": 1.3,
        },
        "scale": 0.05,
    },
    "reachable_par5_water": {
        "sg_weights": {"sg_ott": 1.3, "sg_app": 1.2, "sg_arg": 1.0, "sg_putt": 0.9},
        "attr_weights": {
            "Power": 1.3,
            "Approach": 1.1,
            "ShortGame": 1.0,
            "Putting": 0.9,
            "Consistency": 1.0,
            "DogFactor": 1.3,
        },
        "scale": 0.05,
    },
    "long_par4_water": {
        "sg_weights": {"sg_ott": 1.1, "sg_app": 1.3, "sg_arg": 0.8, "sg_putt": 0.7},
        "attr_weights": {
            "Power": 1.0,
            "Approach": 1.3,
            "ShortGame": 0.8,
            "Putting": 0.8,
            "Consistency": 1.2,
            "DogFactor": 1.2,
        },
        "scale": 0.045,
    },
    "long_par4_tight": {
        "sg_weights": {"sg_ott": 1.3, "sg_app": 1.2, "sg_arg": 0.7, "sg_putt": 0.7},
        "attr_weights": {
            "Power": 1.0,
            "Approach": 1.2,
            "ShortGame": 0.7,
            "Putting": 0.7,
            "Consistency": 1.3,
            "DogFactor": 1.0,
        },
        "scale": 0.045,
    },
}


def load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_PATH)
    df.columns = df.columns.str.strip()
    return df


def add_z_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SG_FEATURES + ATTR_FEATURES:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f"Z_{col}"] = 0.0
        else:
            df[f"Z_{col}"] = (df[col] - mean) / std
    return df


def load_course_layout(course_name: str) -> pd.DataFrame:
    path = COURSE_DIR / f"{course_name}.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def hole_fit_for_type(df_players: pd.DataFrame, hole_type: str) -> np.ndarray:
    if hole_type not in HOLE_TYPE_WEIGHTS:
        raise ValueError(f"Unknown hole_type '{hole_type}' in layout")

    profile = HOLE_TYPE_WEIGHTS[hole_type]
    sg_w = profile["sg_weights"]
    attr_w = profile["attr_weights"]
    scale = profile["scale"]

    cf = np.zeros(len(df_players), dtype=float)

    for feat, w in sg_w.items():
        z_col = f"Z_{feat}"
        cf += w * df_players[z_col].to_numpy(dtype=float)

    for feat, w in attr_w.items():
        z_col = f"Z_{feat}"
        cf += w * df_players[z_col].to_numpy(dtype=float)

    # strokes gained vs field on THIS hole (per round)
    return scale * cf


def export_hole_fit_matrix(
    df_players: pd.DataFrame,
    course_layout: pd.DataFrame,
    course_name: str,
):
    """
    Writes a long-form table:
      hole, par, hole_type, player_name, expected_sg_vs_field_per_round
    """
    holes = course_layout["hole"].tolist()
    pars = course_layout["par"].tolist()
    hole_types = course_layout["hole_type"].tolist()

    rows = []
    for hole, par, h_type in zip(holes, pars, hole_types):
        cf_vec = hole_fit_for_type(df_players, h_type)
        for player_name, cf_val in zip(df_players["player_name"], cf_vec):
            rows.append(
                {
                    "hole": hole,
                    "par": par,
                    "hole_type": h_type,
                    "player_name": player_name,
                    "expected_sg_vs_field_per_round": cf_val,
                }
            )

    df_out = pd.DataFrame(rows)
    out_path = DATA_PROCESSED / f"hole_fit_{course_name}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved hole-level fit table to: {out_path}")


def simulate_tournament_hole_by_hole(
    df_players: pd.DataFrame,
    course_layout: pd.DataFrame,
    n_rounds: int = 4,
    n_sims: int = 2000,
    form_weight: float = 0.3,
    mu_shrink: float = 0.3,
    cf_effect_scale: float = 0.6,
    sigma_scale: float = 2.5,
    min_sigma: float = 0.7,
    hole_noise_sd: float = 0.35,
):
    """
    Hole-by-hole tournament sim for analysis (not primary prediction engine).
    """

    n_players = len(df_players)
    mu_raw = df_players["mu"].to_numpy(dtype=float)
    form = df_players["form"].to_numpy(dtype=float)
    sigma_raw = df_players["sigma"].to_numpy(dtype=float)

    mu_mean = mu_raw.mean()
    mu_shrunk = mu_mean + mu_shrink * (mu_raw - mu_mean)

    base_mu = mu_shrunk + form_weight * form
    sigma = np.maximum(min_sigma, sigma_raw * sigma_scale)

    win_counts = np.zeros(n_players, dtype=int)
    top5_counts = np.zeros(n_players, dtype=int)
    rank_sums = np.zeros(n_players, dtype=float)

    hole_types = course_layout["hole_type"].tolist()
    n_holes = len(hole_types)

    per_hole_fit = []
    for h_type in hole_types:
        per_hole_fit.append(hole_fit_for_type(df_players, h_type))
    per_hole_fit = np.stack(per_hole_fit, axis=0)  # (n_holes, n_players)

    # dial down course-fit effect a bit
    per_hole_fit = cf_effect_scale * per_hole_fit

    for _ in range(n_sims):
        total_scores_vs_par = np.zeros(n_players, dtype=float)

        for _round in range(n_rounds):
            round_base = np.random.normal(
                loc=base_mu,
                scale=sigma,
                size=n_players,
            )
            round_base_per_hole = (round_base / n_holes).reshape(1, -1)

            hole_random = np.random.normal(
                loc=0.0,
                scale=hole_noise_sd,
                size=(n_holes, n_players),
            )

            hole_perf_vs_par = per_hole_fit + round_base_per_hole + hole_random
            total_scores_vs_par += hole_perf_vs_par.sum(axis=0)

        ranks = (-total_scores_vs_par).argsort()

        win_counts[ranks[0]] += 1
        top5_counts[ranks[:5]] += 1

        positions = np.empty_like(ranks)
        positions[ranks] = np.arange(1, n_players + 1)
        rank_sums += positions

    out = df_players[["player_name", "Tier", "TalentScore", "OverallRating"]].copy()
    out["win_pct"] = win_counts / n_sims * 100.0
    out["top5_pct"] = top5_counts / n_sims * 100.0
    out["avg_finish"] = rank_sums / n_sims

    return out.sort_values("win_pct", ascending=False).reset_index(drop=True)


def main():
    course_name = "augusta_national"

    master = load_master()
    master = master.sort_values("TalentScore", ascending=False).head(80).reset_index(drop=True)
    master = add_z_scores(master)

    layout = load_course_layout(course_name)

    # export static hole-level course fit table
    export_hole_fit_matrix(master, layout, course_name)

    # optional: run hole-based tournament sim (analysis only)
    results = simulate_tournament_hole_by_hole(master, layout)

    out_path = DATA_PROCESSED / f"tournament_sim_hole_{course_name}.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved hole-based tournament sim to: {out_path}")
    print(results.head(15))


if __name__ == "__main__":
    main()
