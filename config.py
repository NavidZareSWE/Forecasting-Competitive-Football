from pathlib import Path


# --- Project root (this file lives at the repository root) ----------------
PROJECT = Path(__file__).resolve().parent

# --- Raw data sources (git-ignored) ----------------------------------------
SRC = PROJECT / "src"
DATA_DIR = SRC / "data"
STATSBOMB_DIR = DATA_DIR / "statsbomb_open_data" / "data"
FOOTBALL_DATA_DIR = DATA_DIR / "Football_Data"

# --- Generated outputs (git-ignored) ---------------------------------------
REPORTS_DIR = SRC / "reports"
PROCESSED_DIR = REPORTS_DIR / "processed"
FEATURE_DIR = REPORTS_DIR / "features"
VIS_DIR = REPORTS_DIR / "visualizations"

# --- League configuration --------------------------------------------------
LEAGUES: dict[tuple[int, int], str] = {
    (2, 27): "Premier League",
    (7, 27): "Ligue 1",
    (11, 27): "La Liga",
    (12, 27): "Serie A",
}

# --- Shared constants -------------------------------------------------------
SNAPSHOT_MINUTES: list[int] = list(range(0, 91, 5))
SPLIT_RATIOS: dict[str, float] = {
    "train": 0.60, "validation": 0.20, "test": 0.20}
CLASS_ORDER: list[str] = ["H", "D", "A"]
MARGIN_CLIP: tuple[int, int] = (-5, 5)
