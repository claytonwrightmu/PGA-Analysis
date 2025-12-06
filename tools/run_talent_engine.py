import os
import sys
import pandas as pd

# set up paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# allow imports from src/
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.master_table import build_master_table
from src.player_talent_engine import build_talent_table


def main():
    print("Building master table...")
    build_master_table(save=True)

    print("Running talent engine...")
    df = build_talent_table()

    # sort by TalentScore (mu)
    df = df.sort_values("TalentScore", ascending=False).reset_index(drop=True)

    print("\nTOP 25 PLAYERS")
    print(df[["player", "TalentScore", "Tier"]].head(25).to_string(index=False))

    out_path = os.path.join(ROOT_DIR, "Data", "Players", "processed", "talent_table.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved full talent table to: {out_path}")


if __name__ == "__main__":
    main()
