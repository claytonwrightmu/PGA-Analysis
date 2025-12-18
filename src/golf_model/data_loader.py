"""
Data Loader Module
==================
Loads and merges PGA Tour statistics into unified player database.
Works with your current naming like:
- 23' SG OTT.xlsx
- 24' SG APP.xlsx
- SG Off the tee.xlsx
- SG Putting.xlsx
- 2025 Official World Golf Ranking.xlsx
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PGATourDataLoader:
    """Load and merge PGA Tour statistics from multiple sources."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

        # Internal stat keys used by the rest of the model
        self.stat_keys = [
            "sg_ott",
            "sg_app",
            "sg_arg",
            "sg_putt",
            "sg_t2g",
            "driving_distance",
            "driving_accuracy",
            "gir",
            "scrambling",
            "sand_save",
            "3putt_avoid",
            "putt_inside_10",
            "proximity",
        ]

        # Patterns that match YOUR filenames, case-insensitive
        self.pattern_map: Dict[str, List[str]] = {
            "sg_ott": ["sg off the tee", "sg ott"],
            "sg_app": ["sg approach the green", "sg app"],
            "sg_arg": ["sg around the green", "sg arg"],
            "sg_putt": ["sg putting", "sg putt"],
            "sg_t2g": ["sg teetogreen", "sg tee to green", "sg tee-to-green"],
            "driving_distance": ["ball speed leaders", "driving distance"],
            "driving_accuracy": ["driving accuracy"],
            "gir": ["gir percentage", "gir %", "gir percentage"],
            "scrambling": ["scrambling leaders"],
            "sand_save": ["sand save %", "sand save"],
            "3putt_avoid": ["3 putt avoidance"],
            "putt_inside_10": ["putt inside 10 ft"],
            "proximity": ["proximity to hole"],
        }

        # Common column fallbacks across different exports
        self.value_col_candidates = ["AVG", "AVERAGE", "VALUE"]
        self.rounds_col_candidates = ["MEASURED ROUNDS", "ROUNDS", "RNDS"]

    @staticmethod
    def _norm(s: str) -> str:
        return (
            s.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace("  ", " ")
            .strip()
        )

    def _year_tag(self, year: int) -> str:
        if year == 2023:
            return "23'"
        if year == 2024:
            return "24'"
        if year == 2025:
            return ""  # your current-year files have no 25' tag
        raise ValueError(f"Unsupported year: {year}")

    def _find_xlsx(self, year: int, patterns: List[str]) -> Optional[Path]:
        """Find first matching .xlsx in data_path."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        candidates = list(self.data_path.glob("*.xlsx"))
        if not candidates:
            return None

        tag = self._year_tag(year)
        tag_norm = self._norm(tag) if tag else ""

        patterns_norm = [self._norm(p) for p in patterns]

        # Pass 1: enforce year tag for 2023/2024
        for f in candidates:
            name = self._norm(f.name)
            if tag_norm and tag_norm not in name:
                continue
            for p in patterns_norm:
                if p in name:
                    return f

        # Pass 2: fallback without year tag filter
        for f in candidates:
            name = self._norm(f.name)
            for p in patterns_norm:
                if p in name:
                    return f

        return None

    def _pick_first_existing_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = {self._norm(c): c for c in df.columns}
        for cand in candidates:
            key = self._norm(cand)
            if key in cols:
                return cols[key]
        return None

    def load_year_stats(self, year: int, stats: Optional[List[str]] = None) -> pd.DataFrame:
        if stats is None:
            stats = self.stat_keys

        stat_dfs: Dict[str, pd.DataFrame] = {}

        for stat_name in stats:
            patterns = self.pattern_map.get(stat_name, [stat_name])
            file_path = self._find_xlsx(year, patterns)

            if not file_path:
                logger.warning(f"Could not find file for {stat_name} ({year}) in {self.data_path}")
                continue

            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

            # Must have PLAYER_ID
            if "PLAYER_ID" not in df.columns:
                logger.error(f"PLAYER_ID missing in file {file_path.name} for {stat_name}")
                continue

            value_col = self._pick_first_existing_col(df, self.value_col_candidates)
            rounds_col = self._pick_first_existing_col(df, self.rounds_col_candidates)

            stat_df = df[["PLAYER_ID"]].copy()
            if value_col:
                stat_df[stat_name] = df[value_col]
            if rounds_col:
                stat_df[f"{stat_name}_rounds"] = df[rounds_col]

            stat_dfs[stat_name] = stat_df

        if not stat_dfs:
            raise ValueError(f"No stats loaded for year {year} from {self.data_path}")

        merged = None
        for _, sdf in stat_dfs.items():
            merged = sdf if merged is None else merged.merge(sdf, on="PLAYER_ID", how="outer")

        merged["year"] = year
        logger.info(f"Loaded {year} data: {len(merged)} players, {len(stat_dfs)} stats")
        return merged

    def load_multi_year_stats(self, years: List[int], stats: Optional[List[str]] = None) -> pd.DataFrame:
        year_dfs = []
        for y in years:
            try:
                year_dfs.append(self.load_year_stats(y, stats))
            except Exception as e:
                logger.error(f"Failed to load year {y}: {e}")

        if not year_dfs:
            raise ValueError("No years loaded successfully")

        combined = pd.concat(year_dfs, ignore_index=True)
        logger.info(
            f"Loaded {len(years)} years: {len(combined)} player-years, "
            f"{combined['PLAYER_ID'].nunique()} unique players"
        )
        return combined

    def load_player_names(self, year: int = 2025) -> pd.DataFrame:
        # Use any SG file to get names
        file_path = self._find_xlsx(year, ["sg off the tee", "sg ott", "sg putting", "sg putt"])
        if not file_path:
            raise FileNotFoundError("Could not find any SG file to extract PLAYER names")

        df = pd.read_excel(file_path)
        if "PLAYER" not in df.columns:
            raise ValueError(f"PLAYER column missing in {file_path.name}")
        return df[["PLAYER_ID", "PLAYER"]].drop_duplicates()

    def load_world_rankings(self, year: int = 2025) -> pd.DataFrame:
        # Match your real file names
        file_path = self._find_xlsx(year, ["official world golf ranking", "world golf rankings"])
        if not file_path:
            raise FileNotFoundError(f"Could not find world rankings file for {year}")

        df = pd.read_excel(file_path)

        # Try to standardize
        rename_map = {
            "RANK": "world_rank",
            "AVG POINTS": "avg_points",
            "TOTAL POINTS": "total_points",
        }
        df = df.rename(columns=rename_map)

        keep = ["PLAYER_ID", "world_rank", "avg_points", "total_points"]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"World rankings missing columns: {missing} in {file_path.name}")

        return df[keep]

    def create_master_player_database(
        self,
        current_year: int = 2025,
        include_history_years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        current = self.load_year_stats(current_year)

        # Names
        names = self.load_player_names(current_year)
        current = current.merge(names, on="PLAYER_ID", how="left")

        # World rankings
        try:
            rankings = self.load_world_rankings(current_year)
            current = current.merge(rankings, on="PLAYER_ID", how="left")
        except Exception as e:
            logger.warning(f"Could not load rankings: {e}")

        # Historical career stats
        if include_history_years:
            hist = self.load_multi_year_stats(include_history_years, stats=["sg_ott", "sg_app", "sg_arg", "sg_putt"])
            key_stats = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
            career = hist.groupby("PLAYER_ID")[key_stats].agg(["mean", "std", "count"])
            career.columns = [f"{stat}_{agg}_career" for stat, agg in career.columns]
            career = career.reset_index()
            current = current.merge(career, on="PLAYER_ID", how="left")

        # Total rounds
        round_cols = [c for c in current.columns if c.endswith("_rounds")]
        if round_cols:
            current["total_rounds"] = current[round_cols].sum(axis=1, skipna=True)

        logger.info(f"Created master database: {len(current)} players, {len(current.columns)} columns")
        return current

    def calculate_composite_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sg_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
        if all(c in df.columns for c in sg_cols):
            df["sg_t2g"] = df[["sg_ott", "sg_app", "sg_arg"]].sum(axis=1)
            df["sg_total"] = df[sg_cols].sum(axis=1)
            df["sg_ballstriking"] = df["sg_ott"] + df["sg_app"]
        return df
