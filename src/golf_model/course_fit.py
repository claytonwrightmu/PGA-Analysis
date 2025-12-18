import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from .config.model_config import COURSE_ARCHETYPES, CourseArchetype

logger = logging.getLogger(__name__)

class CourseFitCalculator:
    def __init__(self, archetypes: Dict[str, CourseArchetype] = None):
        self.archetypes = archetypes or COURSE_ARCHETYPES

    def calculate_all_archetype_fits(
        self,
        player_df: pd.DataFrame,
        skill_columns: Dict[str, str] = None
    ) -> pd.DataFrame:
        if skill_columns is None:
            skill_columns = {
                "sg_ott": "sg_ott_talent",
                "sg_app": "sg_app_talent",
                "sg_arg": "sg_arg_talent",
                "sg_putt": "sg_putt_talent",
            }

        df = player_df.copy()

        # Build skill matrix (n_players x 4)
        skills = ["sg_ott", "sg_app", "sg_arg", "sg_putt"]
        X = np.zeros((len(df), len(skills)), dtype=float)

        for j, s in enumerate(skills):
            col = skill_columns.get(s, s)
            if col in df.columns:
                X[:, j] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
            elif s in df.columns:
                X[:, j] = pd.to_numeric(df[s], errors="coerce").fillna(0.0).values
            else:
                X[:, j] = 0.0

        # For each archetype, dot product
        for arch_name, arch in self.archetypes.items():
            w = np.array([arch.skill_weights.get(s, 0.0) for s in skills], dtype=float)
            df[f"fit_{arch_name}"] = X @ w

        return df
# --- compatibility export (pipeline expects this name) ---

def calculate_course_fits(player_data: pd.DataFrame, use_talent: bool = True) -> pd.DataFrame:
    """
    Backwards-compatible wrapper.
    Pipeline imports calculate_course_fits, so we expose it here.

    This wrapper uses CourseFitCalculator + percentiles the same way the Claude version did.
    """
    calculator = CourseFitCalculator()

    if use_talent:
        skill_columns = {
            "sg_ott": "sg_ott_talent",
            "sg_app": "sg_app_talent",
            "sg_arg": "sg_arg_talent",
            "sg_putt": "sg_putt_talent",
        }
    else:
        skill_columns = {
            "sg_ott": "sg_ott",
            "sg_app": "sg_app",
            "sg_arg": "sg_arg",
            "sg_putt": "sg_putt",
        }

    df = calculator.calculate_all_archetype_fits(player_data, skill_columns=skill_columns)
    df = calculator.calculate_fit_percentiles(df)
    return df
