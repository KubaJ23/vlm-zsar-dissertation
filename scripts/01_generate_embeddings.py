from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.clip import DEVICE, model, processor
from src.utils.constants import (
    TEST_CSV,
    UCF101_EMBEDDINGS_DIR,
    UCF101_VIDEOS_DIR,
)

_BATCH_SIZE = 64


def extract_all_frames(video_path: Path):
    """Extract all RGB frames from a video.

    Args:
        video_path: Path to the input video file.

    Returns:
        List of frames as NumPy arrays in RGB order.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


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

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        video_rel = Path(row["clip_path"].lstrip("/"))
        video_path = videos_dir / video_rel

        embed_rel = video_rel.with_suffix(".pt")
        embed_path = embeddings_dir / embed_rel

        # skip existing work
        if embed_path.exists():
            continue

        embed_path.parent.mkdir(parents=True, exist_ok=True)

        frames = extract_all_frames(video_path)
        if not frames:
            print(f"Warning: could not read {video_path}")
            continue

        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(frames), _BATCH_SIZE):
                batch = frames[start : start + _BATCH_SIZE]
                inputs = processor(
                    text=None, images=batch, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                feats = model.get_image_features(**inputs)
                all_embeddings.append(feats.cpu())

        if not all_embeddings:
            print(f"Warning: no embeddings for {video_path}")
            continue

        embedding_tensor = torch.cat(all_embeddings, dim=0)
        torch.save(embedding_tensor, embed_path)

    print(f"\nEmbeddings saved to {embeddings_dir}")


if __name__ == "__main__":
    generate_embeddings(TEST_CSV, UCF101_VIDEOS_DIR, UCF101_EMBEDDINGS_DIR)
