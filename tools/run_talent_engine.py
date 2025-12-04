import os
import sys
import pandas as pd

# --- set up imports so we can use the src package ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# Make sure the project root is on sys.path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.master_table import build_master_table
from src.player_talent_engine import build_talent_table


def main():
    # 1) build the full master player table
    master_df = build_master_table(save=True)

    # 2) run the talent engine
    talent_df = build_talent_table(
        master_df,
        n_tiers=5,
        labels=["S", "A", "B", "C", "D"],
        w_off_tee=1.0,
        w_approach=1.3,
        w_around=0.8,
        w_putting=1.0,
        use_total=True,
    )

    # 3) show top 25
    top_25 = talent_df[["player", "TalentScore", "Tier"]].head(25)
    print(top_25.to_string(index=False))

    # 4) save full talent table
    out_path = os.path.join(ROOT_DIR, "Data", "Players", "processed", "talent_table.csv")
    talent_df.to_csv(out_path, index=False)
    print(f"\nSaved full talent table to: {out_path}")


if __name__ == "__main__":
    main()
