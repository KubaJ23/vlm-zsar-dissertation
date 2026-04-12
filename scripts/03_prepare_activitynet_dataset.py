import shutil
from collections import Counter
from pathlib import Path

import fiftyone.zoo as foz
import pandas as pd
import tqdm
from fiftyone import Dataset

from src.utils.constants import (
    ACTIVITYNET_CSV,
    ACTIVITYNET_EMBEDDINGS_DIR,
    ACTIVITYNET_VIDEOS_DIR,
)
from src.utils.dataset_preparation import generate_embeddings


def export_zoo_to_local(dataset: Dataset, dest_video_dir: Path) -> list[dict]:
    """Exports videos from a zoo dataset of videos to a destination directory and returns a list of records with metadata for each video"""

    dest_video_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # zoo stores videos in its own dataset structure, this extracts those videos and saves them in the structure required for this project's pipeline
    for sample in tqdm(
        dataset, total=len(dataset), desc="Exporting videos from zoo dataset"
    ):
        labels = [d.label for d in sample.ground_truth.detections]
        label = Counter(labels).most_common(1)[0][0]

        class_folder = dest_video_dir / label.replace(" ", "_")
        class_folder.mkdir(parents=True, exist_ok=True)

        # path to video in zoo's dataset
        src = Path(sample.filepath)

        # destination path
        dest = class_folder / src.name

        # add record for CSV
        # this gets the relative path to this video
        rel_path = dest.relative_to(dest_video_dir)
        records.append(
            {
                "clip_path": rel_path,
                "label": label,
                "video_id": src.stem,
                "fo_id": sample.id,
            }
        )

        # allows this exporting to be resumed if not finished
        if dest.exists():
            continue

        shutil.copy2(src, dest)

    print(f"Exported {len(records)} videos to {dest_video_dir}")

    return records


if __name__ == "__main__":
    # load random validation videos
    dataset: Dataset = foz.load_zoo_dataset(
        "activitynet-200",
        split="validation",
        max_samples=1000,
        shuffle=True,
    )
    print("Loaded zoo dataset")

    records = export_zoo_to_local(dataset, ACTIVITYNET_VIDEOS_DIR)

    if len(records) > 0:
        df = pd.DataFrame(records)
        df.to_csv(ACTIVITYNET_CSV, index=False)

    print(f"ActivityNet CSV file saved to {ACTIVITYNET_CSV}")

    # Some video in the ActivityNet dataset could not be correctly downloaded from youtube so embeddings generation fails and skips some videos
    generate_embeddings(
        ACTIVITYNET_CSV, ACTIVITYNET_VIDEOS_DIR, ACTIVITYNET_EMBEDDINGS_DIR
    )

    print("\n")
    print("Done preparing ActivityNet dataset and precomputing embeddings.")
    print("\n")
