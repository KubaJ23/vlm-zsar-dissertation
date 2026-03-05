from src.utils.constants import (
    TEST_CSV,
    UCF101_EMBEDDINGS_DIR,
    UCF101_VIDEOS_DIR,
)
from src.utils.dataset_preparation import generate_embeddings

if __name__ == "__main__":
    generate_embeddings(TEST_CSV, UCF101_VIDEOS_DIR, UCF101_EMBEDDINGS_DIR)
