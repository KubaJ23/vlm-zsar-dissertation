from pathlib import Path

import numpy as np
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
        """
        Retrieves the pre-computed embeddings for a given video index.

        This method finds the file whether it is .pt or .npy,
        and always returns a NumPy array.
        """
        if not (0 <= video_index < len(self._df)):
            raise IndexError("video_index out of range")

        row = self._df.iloc[video_index]

        embeddings_relative_path = Path(row["clip_path"].strip("/")).with_suffix("")

        full_embeddings_path = self.embeddings_root / embeddings_relative_path

        # Check if the embeddings are stored as a NumPy array first, then check if they're a PyTorch tensor
        npy_path = full_embeddings_path.with_suffix(".npy")
        pt_path = full_embeddings_path.with_suffix(".pt")

        if npy_path.exists():
            # Load NumPy array directly
            return np.load(npy_path)
        elif pt_path.exists():
            # Load PyTorch tensor, convert to NumPy, and return
            tensor = torch.load(pt_path)
            return tensor.cpu().numpy()
        else:
            raise FileNotFoundError(
                f"No embedding file found for base path: {full_embeddings_path}"
                f"\nChecked for: {npy_path}"
                f"\nAnd for: {pt_path}"
            )

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
