from dataclasses import dataclass
from pathlib import Path
from typing import Dict

MODEL_VERSION = "0.2.1"

# Repo paths
REPO_ROOT = Path("/workspaces/PGA-Analysis")

# Your real raw Excel folder
DEFAULT_DATA_PATH = REPO_ROOT / "Data" / "Players" / "raw"

# Logs and outputs
LOG_PATH = REPO_ROOT / "logs"
OUTPUT_PATH = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class CourseArchetype:
    name: str
    description: str
    skill_weights: Dict[str, float]


# Minimal but functional archetype set.
# You can expand this later without touching pipeline/course_fit logic.
COURSE_ARCHETYPES: Dict[str, CourseArchetype] = {
    "bomber_paradise": CourseArchetype(
        name="bomber_paradise",
        description="Distance-friendly scoring, driver advantage",
        skill_weights={"sg_ott": 0.45, "sg_app": 0.30, "sg_arg": 0.10, "sg_putt": 0.15},
    ),
    "classic_ballstriking": CourseArchetype(
        name="classic_ballstriking",
        description="Approach play and tee-to-green precision matter most",
        skill_weights={"sg_ott": 0.25, "sg_app": 0.50, "sg_arg": 0.10, "sg_putt": 0.15},
    ),
    "accuracy_premium": CourseArchetype(
        name="accuracy_premium",
        description="Tight/penal courses, keep it in play and hit greens",
        skill_weights={"sg_ott": 0.30, "sg_app": 0.40, "sg_arg": 0.15, "sg_putt": 0.15},
    ),
    "major_championship": CourseArchetype(
        name="major_championship",
        description="Hard setup, scrambling and elite tee-to-green rise",
        skill_weights={"sg_ott": 0.30, "sg_app": 0.35, "sg_arg": 0.20, "sg_putt": 0.15},
    ),
}
