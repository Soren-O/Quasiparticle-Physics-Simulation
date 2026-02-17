from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SETUPS_DIR = DATA_DIR / "setups"
SIMULATIONS_DIR = DATA_DIR / "simulations"
TEST_CASES_DIR = DATA_DIR / "test_cases"


def ensure_data_dirs() -> None:
    SETUPS_DIR.mkdir(parents=True, exist_ok=True)
    SIMULATIONS_DIR.mkdir(parents=True, exist_ok=True)
    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)

