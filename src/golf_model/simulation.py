"""
Tournament Simulation Engine
============================
Monte Carlo simulation of PGA Tour tournaments with:
- Round-by-round score generation
- 36-hole cut logic
- Weather/conditions variance
- Player form fluctuation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TournamentConfig:
    """Configuration for a tournament simulation."""
    name: str
    course_archetype: str
    field_size: int
    rounds: int = 4
    cut_line: int = 65
    cut_after_round: int = 2
    par: int = 72
    weather_factor: float = 1.0  # 1.0 = normal, >1.0 = harder conditions


class TournamentSimulator:
    """Simulate PGA Tour tournament outcomes."""

    def __init__(
        self,
        base_scoring_std: float = 2.5,
        form_weight: float = 0.15,
        random_seed: Optional[int] = None,
    ):
        self.base_scoring_std = float(base_scoring_std)
        self.form_weight = float(form_weight)
        self.rng = np.random.default_rng(random_seed)

    def generate_round_score(
        self,
        talent: float,
        course_fit: float,
        uncertainty: float,
        par: int = 72,
        weather_factor: float = 1.0,
        form_adjustment: float = 0.0,
    ) -> int:
        """
        Expected SG = talent + course_fit + form_adjustment
        Actual SG ~ Normal(Expected, total_std)
        Score = par - SG
        """
        expected_sg = float(talent) + float(course_fit) + float(form_adjustment)

        total_std = np.sqrt(
            self.base_scoring_std**2
            + float(uncertainty) ** 2
            + (float(weather_factor) - 1.0) ** 2
        )

        actual_sg = self.rng.normal(expected_sg, total_std)
        score = par - actual_sg
        return int(np.round(score))

    def simulate_tournament(
        self,
        field: pd.DataFrame,
        config: TournamentConfig,
        n_simulations: int = 10000,
    ) -> pd.DataFrame:
        """
        field must contain:
        - PLAYER_ID
        - overall_talent
        - overall_uncertainty (optional, default 0.3)
        - fit_{config.course_archetype}
        - PLAYER (optional)
        - form (optional)
        """
        fit_col = f"fit_{config.course_archetype}"
        if fit_col not in field.columns:
            raise ValueError(f"Course fit for {config.course_archetype} not found (missing {fit_col})")

        if "PLAYER_ID" not in field.columns:
            raise ValueError("field missing PLAYER_ID")

        n_players = len(field)
        if n_players == 0:
            return field.copy()

        # Build fast index mapping once (huge speedup vs searching each sim)
        player_ids = field["PLAYER_ID"].to_numpy()
        id_to_pos = {int(pid): i for i, pid in enumerate(player_ids)}

        names = field["PLAYER"].to_numpy() if "PLAYER" in field.columns else None
        talents = field.get("overall_talent", pd.Series([0.0] * n_players)).to_numpy(dtype=float)
        fits = field[fit_col].to_numpy(dtype=float)
        uncert = field.get("overall_uncertainty", pd.Series([0.3] * n_players)).to_numpy(dtype=float)
        form_base = field.get("form", pd.Series([0.0] * n_players)).to_numpy(dtype=float)

        wins = np.zeros(n_players, dtype=int)
        top5 = np.zeros(n_players, dtype=int)
        top10 = np.zeros(n_players, dtype=int)
        top20 = np.zeros(n_players, dtype=int)
        made_cut = np.zeros(n_players, dtype=int)
        total_scores_sum = np.zeros(n_players, dtype=float)

        finish_positions = np.empty((n_simulations, n_players), dtype=float)

        logger.info(f"Running {n_simulations} simulations of {config.name}")

        for sim in range(n_simulations):
            results = self._simulate_single(
                player_ids=player_ids,
                talents=talents,
                fits=fits,
                uncert=uncert,
                form_base=form_base,
                config=config,
            )

            # results are aligned to player_ids order
            finishes = results["finish"]
            totals = results["total_score"]
            cuts = results["made_cut"]

            wins += (finishes == 1)
            top5 += (finishes <= 5)
            top10 += (finishes <= 10)
            top20 += (finishes <= 20)
            made_cut += cuts.astype(int)
            total_scores_sum += totals
            finish_positions[sim, :] = finishes

        summary = pd.DataFrame({"PLAYER_ID": player_ids})
        if names is not None:
            summary["PLAYER"] = names

        summary["win_probability"] = wins / n_simulations
        summary["top_5_probability"] = top5 / n_simulations
        summary["top_10_probability"] = top10 / n_simulations
        summary["top_20_probability"] = top20 / n_simulations
        summary["make_cut_probability"] = made_cut / n_simulations
        summary["avg_total_score"] = total_scores_sum / n_simulations

        summary["expected_finish"] = np.median(finish_positions, axis=0)
        summary["finish_10th_percentile"] = np.percentile(finish_positions, 10, axis=0)
        summary["finish_90th_percentile"] = np.percentile(finish_positions, 90, axis=0)

        summary[fit_col] = fits
        summary["overall_talent"] = talents

        summary = summary.sort_values("win_probability", ascending=False).reset_index(drop=True)
        return summary

    def _simulate_single(
        self,
        player_ids: np.ndarray,
        talents: np.ndarray,
        fits: np.ndarray,
        uncert: np.ndarray,
        form_base: np.ndarray,
        config: TournamentConfig,
    ) -> Dict[str, np.ndarray]:
        n = len(player_ids)

        # Form fluctuates per tournament
        form_adj = self.form_weight * self.rng.normal(form_base, 0.2)

        scores = np.zeros((n, config.rounds), dtype=int)
        totals = np.zeros(n, dtype=int)
        alive = np.ones(n, dtype=bool)

        for r in range(config.rounds):
            round_num = r + 1
            weather = config.weather_factor * (1.1 if round_num == 3 else 1.0)

            for i in range(n):
                if not alive[i]:
                    continue
                scores[i, r] = self.generate_round_score(
                    talent=talents[i],
                    course_fit=fits[i],
                    uncertainty=uncert[i],
                    par=config.par,
                    weather_factor=weather,
                    form_adjustment=form_adj[i],
                )

            totals = scores[:, : round_num].sum(axis=1)

            # Cut after specified round
            if config.cut_after_round and round_num == config.cut_after_round:
                alive = self._apply_cut_mask(totals, config.cut_line)

        # Finishes: rank by total score among those who made cut
        made_cut = alive if config.cut_after_round else np.ones(n, dtype=bool)

        finish = np.full(n, config.cut_line + 1, dtype=float)  # missed cut default
        if made_cut.any():
            idx = np.where(made_cut)[0]
            order = idx[np.argsort(totals[idx], kind="mergesort")]  # stable
            finish_pos = 1
            prev_score = None
            tie_count = 0
            for j in order:
                if prev_score is None or totals[j] != prev_score:
                    finish_pos += tie_count
                    tie_count = 1
                    prev_score = totals[j]
                else:
                    tie_count += 1
                finish[j] = finish_pos

        return {
            "finish": finish,
            "total_score": totals.astype(float),
            "made_cut": made_cut,
        }

    @staticmethod
    def _apply_cut_mask(totals: np.ndarray, cut_line: int) -> np.ndarray:
        """
        Top cut_line + ties make cut.
        totals are after cut round.
        """
        n = len(totals)
        if n <= cut_line:
            return np.ones(n, dtype=bool)

        order = np.argsort(totals, kind="mergesort")
        cut_score = totals[order[cut_line - 1]]
        return totals <= cut_score


def simulate_tournament(
    field_data: pd.DataFrame,
    tournament_name: str,
    course_archetype: str,
    n_simulations: int = 10000,
) -> pd.DataFrame:
    """Convenience wrapper."""
    config = TournamentConfig(
        name=tournament_name,
        course_archetype=course_archetype,
        field_size=len(field_data),
    )
    sim = TournamentSimulator()
    return sim.simulate_tournament(field_data, config, n_simulations)
