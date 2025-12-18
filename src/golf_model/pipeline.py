import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .data_loader import PGATourDataLoader
from .talent_estimation import estimate_player_talents
from .course_fit import calculate_course_fits, CourseFitCalculator
from .simulation import TournamentSimulator, TournamentConfig
from .config.model_config import MODEL_VERSION, LOG_PATH, OUTPUT_PATH, DEFAULT_DATA_PATH

# Ensure folders exist
LOG_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class GolfModelPipeline:
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = str(DEFAULT_DATA_PATH if data_path is None else Path(data_path))
        self.loader = PGATourDataLoader(self.data_path)
        self.player_data: Optional[pd.DataFrame] = None

        logger.info(f"Initialized Golf Model Pipeline v{MODEL_VERSION}")
        logger.info(f"Data path: {self.data_path}")

    def run_complete_pipeline(
    self,
    validate_data: bool = False,
    include_historical: bool = True,
    save_outputs: bool = True
) -> Dict:
        # validate_data is accepted for compatibility with your old command.
        # We are not running deep validation yet, since you do not have that module wired in.

        logger.info("=" * 80)
        logger.info("STARTING COMPLETE PIPELINE")
        logger.info("=" * 80)

        logger.info("[STEP 1] Loading player data...")
        self.player_data = self.loader.create_master_player_database(
            current_year=2025,
            include_history_years=[2023, 2024] if include_historical else None,
        )
        self.player_data = self.loader.calculate_composite_ratings(self.player_data)
        logger.info(f"Loaded {len(self.player_data)} players")
        # OPTIONAL: verify raw data files
         if validate_data:
        logger.info("[VALIDATION] Verifying raw data files...")
        from .verify_data_files import verify_data_files
        verify_data_files(self.data_path)

        logger.info("[STEP 2] Estimating player talents (Bayesian)...")
        self.player_data = estimate_player_talents(self.player_data)
        logger.info(f"Estimated talents for {len(self.player_data)} players")

        logger.info("[STEP 3] Calculating course fit scores...")
        self.player_data = calculate_course_fits(self.player_data, use_talent=True)
        fit_cols = [c for c in self.player_data.columns if c.startswith("fit_")]
        logger.info(f"Calculated fits for {len(fit_cols)} archetypes")

        summaries = self._generate_summaries()

        if save_outputs:
            logger.info("[STEP 4] Saving outputs...")
            self._save_outputs()

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)

        return {"player_data": self.player_data, "summaries": summaries}

    def _generate_summaries(self) -> Dict:
        df = self.player_data
        summaries: Dict = {}

        if df is None:
            return summaries

        if "overall_talent" in df.columns:
            summaries["top_10_overall"] = df.nlargest(10, "overall_talent")[
                ["PLAYER", "overall_talent", "talent_tier"]
            ].to_dict("records")

        fit_cols = [c for c in df.columns if c.startswith("fit_")]
        for fit_col in fit_cols:
            arch = fit_col.replace("fit_", "")
            cols = [c for c in ["PLAYER", fit_col, "overall_talent"] if c in df.columns]
            summaries[f"top_10_{arch}"] = df.nlargest(10, fit_col)[cols].to_dict("records")

        summaries["stats"] = {
            "n_players": int(len(df)),
            "mean_talent": float(df["overall_talent"].mean()) if "overall_talent" in df.columns else None,
            "std_talent": float(df["overall_talent"].std()) if "overall_talent" in df.columns else None,
        }
        return summaries

    def _save_outputs(self) -> None:
        if self.player_data is None:
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_master = OUTPUT_PATH / f"master_player_database_{ts}.csv"
        self.player_data.to_csv(out_master, index=False)
        logger.info(f"Saved: {out_master}")

        calc = CourseFitCalculator()
        for arch in calc.archetypes.keys():
            ranked = calc.rank_players_for_archetype(self.player_data, arch, top_n=50)
            out_file = OUTPUT_PATH / f"top_players_{arch}_{ts}.csv"
            ranked.to_csv(out_file, index=False)
            logger.info(f"Saved: {out_file}")

    def simulate_tournament(
        self,
        tournament_name: str,
        course_archetype: str,
        field_player_ids: Optional[List[int]] = None,
        n_simulations: int = 10000,
    ) -> pd.DataFrame:
        if self.player_data is None:
            raise ValueError("Run pipeline first")

        if field_player_ids:
            field = self.player_data[self.player_data["PLAYER_ID"].isin(field_player_ids)].copy()
        else:
            field = self.player_data.copy()

        config = TournamentConfig(
            name=tournament_name,
            course_archetype=course_archetype,
            field_size=len(field),
        )

        sim = TournamentSimulator()
        results = sim.simulate_tournament(field, config, n_simulations)

        out_file = OUTPUT_PATH / (
            f"simulation_{tournament_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results.to_csv(out_file, index=False)
        logger.info(f"Saved simulation: {out_file}")

        return results
