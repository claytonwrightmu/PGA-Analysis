import argparse
from pathlib import Path
import pandas as pd

from src.tournament_sim_ratings import (
    load_master,
    load_course_fit,
    simulate_rating_based,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
TOURNAMENTS_DIR = DATA_DIR / "Tournaments"
PROCESSED_DIR = TOURNAMENTS_DIR / "processed"
SCHEDULE_PATH = TOURNAMENTS_DIR / "schedule_2025.csv"

# --------------------------------
# COURSE / EVENT VOLATILITY CONFIG
# --------------------------------

# Roughly: 1.0 = normal, >1 = more chaotic, <1 = more stable
COURSE_VOLATILITY = {
    "wedge_birdiefest": 0.95,              # Kapalua style resort
    "stadium_desert_scottsdale": 1.05,     # WM Phoenix: some chaos
    "bomber_ballstriking_riviera": 1.20,   # Riviera: tough tee-to-green
    "us_open_penal": 1.40,                 # Bay Hill / US Open setups
    "water_target_tpc": 1.35,              # Sawgrass: water & nerves
    "augusta_masters": 1.30,
    "wind_links_open": 1.35,
    "eastlake_ballstriking": 1.10,
}

# Form importance per event (early season grows over time)
FORM_WEIGHT_BY_EVENT = {
    "sentry_2025": 0.05,
    "wmphoenix_2025": 0.15,
    "riviera_2025": 0.25,
    "bayhill_2025": 0.35,
    "players_2025": 0.45,
    # majors / end of season
    "masters_2025": 0.50,
    "pga_2025": 0.45,
    "usopen_2025": 0.45,
    "open_2025": 0.40,
    "tourchamp_2025": 0.40,
}

DEFAULT_FORM_WEIGHT = 0.25
DEFAULT_SIGMA_SCALE = 0.20
DEFAULT_UPSET_SCALE = 0.15


def load_schedule():
    df = pd.read_csv(SCHEDULE_PATH)
    df.columns = df.columns.str.strip()
    return df


def get_event(schedule, event_id):
    event_row = schedule.loc[schedule["event_id"] == event_id]
    if event_row.empty:
        raise ValueError(f"Event ID '{event_id}' not found in schedule.")
    return event_row.iloc[0]


def safe_course_fit(df_master, course_key):
    try:
        cf = load_course_fit(course_key)
        merged = df_master.merge(
            cf[["player_name", "CF_SG_per_round"]],
            on="player_name",
            how="left",
        )
        merged["CF_SG_per_round"] = merged["CF_SG_per_round"].fillna(0.0)
        return merged
    except FileNotFoundError:
        print(f"[WARN] Missing course fit for '{course_key}' — using neutral CF=0.")
        df_master["CF_SG_per_round"] = 0.0
        return df_master


def run_event(event_id, sims=5000):
    schedule = load_schedule()
    event = get_event(schedule, event_id)

    name = event["tournament_name"]
    course_key = event["course_profile_key"]
    field_size = int(event["field_size"])

    course_vol = COURSE_VOLATILITY.get(course_key, 1.0)
    form_weight = FORM_WEIGHT_BY_EVENT.get(event_id, DEFAULT_FORM_WEIGHT)

    print(f"\n=== Running Tournament Simulation ===")
    print(f"Event: {name}")
    print(f"Event ID: {event_id}")
    print(f"Course Profile: {course_key}")
    print(f"Field Size: {field_size}")
    print(f"Simulations: {sims}")
    print(f"Course volatility: {course_vol}")
    print(f"Form weight: {form_weight}")
    print("-------------------------------------")

    df_master = load_master()
    df_master = df_master.sort_values("TalentScore", ascending=False).head(field_size)

    df_with_cf = safe_course_fit(df_master, course_key)

    results = simulate_rating_based(
        df_with_cf,
        n_sims=sims,
        skill_weight=0.38,
        course_volatility=course_vol,
        form_weight=form_weight,
        sigma_scale=DEFAULT_SIGMA_SCALE,
        upset_scale=DEFAULT_UPSET_SCALE,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{event_id}_predictions.csv"
    results.to_csv(out_path, index=False)

    print(f"\nSaved predictions to: {out_path}")
    print("\nTop 10 by Win %:")
    print(results.head(10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--sims", type=int, default=5000)
    args = parser.parse_args()

    run_event(args.event_id, sims=args.sims)


if __name__ == "__main__":
    main()
