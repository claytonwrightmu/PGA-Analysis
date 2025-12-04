import os
import sys
import pandas as pd

# --- set up imports from src/ ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ⚠️ adjust these imports to match your actual functions
from master_table import build_master_table  # or whatever builds your main player df
from player_talent_engine import build_talent_table  # we’ll standardize this


def main():
    # 1) build the full player/master table from your pipeline
    master_df: pd.DataFrame = build_master_table()

    # 2) run the talent engine on that table
    talent_df: pd.DataFrame = build_talent_table(
        master_df,
        n_tiers=5,
        labels=["S", "A", "B", "C", "D"],
        w_off_tee=1.0,
        w_approach=1.3,
        w_around=0.8,
        w_putting=1.0,
        use_total=True,
    )

    # 3) sort and show top 25 in the terminal
   top_25 = talent_df[["player", "TalentScore", "Tier"]].head(25)
    print(top_25.to_string(index=False))

    # 4) save full table
    out_path = os.path.join(ROOT_DIR, "Data", "Players", "processed", "talent_table.csv")
    talent_df.to_csv(out_path, index=False)
    print(f"\nSaved full talent table to: {out_path}")


if __name__ == "__main__":
    main()
