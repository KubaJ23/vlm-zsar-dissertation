from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

TEST_CSV = DATA_DIR / "test.csv"

UCF101_VIDEOS_DIR = DATA_DIR / "UCF-101"

UCF101_EMBEDDINGS_DIR = DATA_DIR / "ucf101_embeddings"

MODEL_RESULTS_BASELINE = DATA_DIR / "model_results_baseline.csv"
