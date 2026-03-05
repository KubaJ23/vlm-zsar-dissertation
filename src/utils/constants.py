from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

# UCF101 Paths
TEST_CSV = DATA_DIR / "test.csv"

UCF101_VIDEOS_DIR = DATA_DIR / "UCF-101"

UCF101_EMBEDDINGS_DIR = DATA_DIR / "ucf101_embeddings"

# ActivityNet Paths
ACTIVITYNET_DIR = DATA_DIR / "activitynet"

ACTIVITYNET_VIDEOS_DIR = ACTIVITYNET_DIR / "videos"

ACTIVITYNET_EMBEDDINGS_DIR = ACTIVITYNET_DIR / "embeddings"

ACTIVITYNET_CSV = ACTIVITYNET_DIR / "activitynet_val.csv"


MPVR_CLASS_DESC_JSON = DATA_DIR / "mpvr_descriptions.json"

MPVR_ACTIVITYNET_CLASS_DESC_JSON = DATA_DIR / "activitynet_descriptions.json"

# Results Paths
RESULTS_DIR = PROJECT_ROOT / "results"
