"""Central configuration: every tunable value in the project lives here."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
AQI_PATH = RAW_DIR / "cpcb_aqi_bulletins_2015_2026.csv"      # CPCB daily AQI bulletins
WEATHER_PATH = RAW_DIR / "delhi_weather_daily.json"           # ERA5 daily weather (Open-Meteo)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Reproducibility: all randomness derives from these seeds.
RF_SEED = 42
SEARCH_SEEDS = (0, 1, 2, 3, 4)   # independent runs of PSO and of random search
EVAL_SEEDS = tuple(range(100, 110))  # RF refits used to average out forest noise

CITY = "Delhi"
TARGET = "AQI"
TEST_START = "2024-01-01"        # everything from here on is held out from tuning

CV_FOLDS = 5                     # walk-forward folds used as the PSO fitness

# CPCB AQI categories: (upper bound inclusive, label)
AQI_BANDS = (
    (50, "Good"),
    (100, "Satisfactory"),
    (200, "Moderate"),
    (300, "Poor"),
    (400, "Very Poor"),
    (float("inf"), "Severe"),
)
HIGH_POLLUTION_AQI = 300         # "Very Poor" or worse: used for the binary alert AUC


@dataclass(frozen=True)
class PSOConfig:
    """Hyperparameters of the swarm itself."""
    n_particles: int = 12
    n_iterations: int = 25
    w_start: float = 0.9         # inertia at the first iteration (exploration)
    w_end: float = 0.4           # inertia at the last iteration (exploitation)
    c1: float = 1.5              # cognitive pull towards a particle's own best
    c2: float = 1.5              # social pull towards the swarm's best
    v_max: float = 0.2           # velocity limit, as a fraction of each search dimension


@dataclass(frozen=True)
class Param:
    """One Random Forest hyperparameter in the search space."""
    name: str
    low: float
    high: float
    integer: bool


SEARCH_SPACE = (
    Param("n_estimators", 50, 300, True),
    Param("max_depth", 3, 30, True),
    Param("min_samples_split", 2, 20, True),
    Param("min_samples_leaf", 1, 15, True),
    Param("max_features", 0.1, 1.0, False),
    Param("max_samples", 0.4, 1.0, False),
)

# Feature selection: after the hyperparameter dimensions, each particle has one
# extra dimension per feature. A feature is used when its coordinate is >= 0.5.
FEATURE_THRESHOLD = 0.5
MIN_FEATURES = 3                 # if fewer pass the threshold, keep the 3 highest
