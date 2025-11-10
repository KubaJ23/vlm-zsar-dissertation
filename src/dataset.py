from pathlib import Path

import pandas as pd
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class UCF101SubsetDataset:
    """
    Dataset class to manage video files and their corresponding
    pre-computed frame embeddings.
    """

    def __init__(self, split="test"):
        self.embeddings_root = DATA_DIR / "ucf101_embeddings"
        self.videos_root = DATA_DIR / "ucf101_subset"

        csv_path = self.embeddings_root / f"{split}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        self._df = pd.read_csv(csv_path)

    def get_embeddings(self, video_index: int):
        if not (0 <= video_index < len(self._df)):
            raise IndexError("video_index out of range")

        row = self._df.iloc[video_index]
        embedding_path = self.embeddings_root / row["clip_path"].strip("/")

        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

        return torch.load(embedding_path)

    def get_video_path(self, video_index: int):
        if not (0 <= video_index < len(self._df)):
            raise IndexError("video_index out of range")

        row = self._df.iloc[video_index]
        video_relative_path = Path(row["clip_path"].strip("/")).with_suffix(".avi")
        video_path = self.videos_root / video_relative_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        return video_path

    def get_dataset_size(self):
        return len(self._df)

    def __len__(self):
        """Allows `len(dataset)` to work."""
        return self.get_dataset_size()

    def __getitem__(self, idx):
        """Allows indexing (e.g., `dataset[0]`) for use in a DataLoader."""
        video_path = self.get_video_path(idx)
        embeddings = self.get_embeddings(idx)
        return video_path, embeddings
