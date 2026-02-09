from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"

TEST_CSV = DATA_DIR / "test.csv"

UCF101_VIDEOS_DIR = DATA_DIR / "UCF-101"

UCF101_EMBEDDINGS_DIR = DATA_DIR / "ucf101_embeddings"

MPVR_CLASS_DESC_JSON = DATA_DIR / "mpvr_descriptions.json"

# Results Paths
RESULTS_DIR = PROJECT_ROOT / "results"
