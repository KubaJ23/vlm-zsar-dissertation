from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.utils.constants import DATA_DIR, TEST_CSV, UCF101_VIDEOS_DIR

# not a constant as it's only used in this script
TESTLIST_PATH = DATA_DIR / "testlist01.txt"


def extract_video_metadata(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return {
        "frame_count": frame_count,
        "duration_sec": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def build_test_csv():
    if not TESTLIST_PATH.exists():
        raise FileNotFoundError(f"Missing test list: {TESTLIST_PATH}")

    lines = [
        line.strip() for line in TESTLIST_PATH.read_text().splitlines() if line.strip()
    ]

    records = []
    for line in lines:
        relative_path = Path(line)
        label = relative_path.parent.name
        clip_name = relative_path.stem
        clip_path = str(relative_path)

        records.append(
            {
                "clip_name": clip_name,
                "clip_path": clip_path,
                "label": label,
            }
        )

    return pd.DataFrame(records)


def main():
    df = build_test_csv()

    metadata_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting metadata"):
        video_path = UCF101_VIDEOS_DIR / row["clip_path"].lstrip("/")
        md = extract_video_metadata(video_path)
        if not md:
            continue

        md["clip_name"] = row["clip_name"]
        metadata_rows.append(md)

    metadata_df = pd.DataFrame(metadata_rows)
    df_final = df.merge(metadata_df, on="clip_name", how="inner")
    df_final.to_csv(TEST_CSV, index=False)

    print(f"\nMetadata saved to {TEST_CSV}")


if __name__ == "__main__":
    main()
