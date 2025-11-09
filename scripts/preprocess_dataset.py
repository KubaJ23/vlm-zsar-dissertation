"""Reads the raw UCF101 dataset and generates a new dataset that only contains the 'test' videos. Also attaches new metadata about videos in the 'test.csv' file"""

import shutil
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TEST_CSV = RAW_DATA_DIR / "test.csv"
RAW_TEST_VIDEOS = RAW_DATA_DIR / "test"

OUTPUT_DATA_DIR = DATA_DIR / "ucf101_subset"
OUTPUT_CSV_PATH = OUTPUT_DATA_DIR / "test.csv"
OUTPUT_VIDEOS_PATH = OUTPUT_DATA_DIR / "test"


def extract_video_metadata(video_path: Path) -> dict | None:
    """Extracts metadata (frame count, duration, resolution, etc.) for one video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = frame_count / fps if fps > 0 else 0
    return {
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "fps": fps,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def process_videos():
    """Load raw test.csv, collect metadata for each video, and save processed CSV."""
    print(f"Loading raw CSV from: {RAW_TEST_CSV}")

    if not RAW_TEST_CSV.exists():
        sys.exit(f"ERROR: File not found: {RAW_TEST_CSV}")

    df = pd.read_csv(RAW_TEST_CSV)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(df)} videos for metadata...")

    metadata = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata"):
        clip_name = row["clip_name"]
        video_path = RAW_DATA_DIR / row["clip_path"].lstrip("/")

        if not video_path.exists():
            print(f"Warning: Missing file, skipped: {video_path}")
            continue

        meta = extract_video_metadata(video_path)
        if meta:
            meta["clip_name"] = clip_name
            metadata.append(meta)
        else:
            print(f"Warning: Could not open {video_path}")

    if not metadata:
        sys.exit("ERROR: No valid video metadata extracted.")

    metadata_df = pd.DataFrame(metadata)
    df_processed = df.merge(metadata_df, on="clip_name")
    df_processed.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Metadata collected: {len(df_processed)} videos processed.")
    print(f"Saved to: {OUTPUT_CSV_PATH}")

    if (RAW_TEST_VIDEOS).exists():
        print(f"Copying videos from {RAW_TEST_VIDEOS} to {OUTPUT_VIDEOS_PATH}...")
        shutil.copytree(RAW_TEST_VIDEOS, OUTPUT_VIDEOS_PATH, dirs_exist_ok=True)
        print("Video folder copied successfully.")


if __name__ == "__main__":
    process_videos()
