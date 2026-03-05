from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.utils.clip import get_video_embeddings


def generate_embeddings(
    dataset_metadata_csv: Path, videos_dir: Path, embeddings_dir: Path
):
    """Generate and save CLIP frame embeddings for all videos listed in a CSV.

    Args:
        dataset_metadata_csv: CSV containing clip_path entries.
        videos_dir: Root directory containing the source videos.
        embeddings_dir: Root directory where embeddings should be saved.
    """
    if not dataset_metadata_csv.exists():
        raise FileNotFoundError(f"Could not find CSV: {dataset_metadata_csv}")

    df = pd.read_csv(dataset_metadata_csv)

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(df)} videos...")

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Generating embeddings", leave=False
    ):
        video_rel = Path(row["clip_path"].lstrip("/"))
        video_path = videos_dir / video_rel

        embed_rel = video_rel.with_suffix(".pt")
        embed_path = embeddings_dir / embed_rel

        # skip existing work
        if embed_path.exists():
            continue

        embed_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings = get_video_embeddings(video_path)

        if embeddings is None or embeddings.numel() == 0:
            print(f"Warning: no embeddings for {video_path}")
            continue

        torch.save(embeddings, embed_path)

    print(f"\nEmbeddings saved to {embeddings_dir}")
