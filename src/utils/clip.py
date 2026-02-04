from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

# Set up model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    MODEL_ID, trust_remote_code=False, use_safetensors=True
).to(DEVICE)

processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)


def compute_frame_embeddings(
    frames: List[np.ndarray], batch_size: int = 32
) -> torch.Tensor:
    """
    Takes a list of RGB frames (numpy arrays) and returns their CLIP embeddings.
    """
    if not frames:
        return torch.empty(0)

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch = frames[start : start + batch_size]

            # Prepare inputs for CLIP
            inputs = processor(
                text=None, images=batch, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            # Encode images
            feats = model.get_image_features(**inputs)
            all_embeddings.append(feats.cpu())

    if not all_embeddings:
        return torch.empty(0)

    return torch.cat(all_embeddings, dim=0)


def get_video_embeddings(video_path: Path, batch_size: int = 32) -> torch.Tensor:
    """
    Reads a video file, extracts all frames, and computes the all frame embeddings.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return torch.empty(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"Warning: No frames found in {video_path}")
        return torch.empty(0)

    return compute_frame_embeddings(frames, batch_size=batch_size)


def extract_selected_frames(
    video_path: Path, frame_indices: list[int]
) -> list[np.ndarray]:
    """
    Extracts specific frames from a video corresponding to the provided indices.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    for idx in frame_indices:
        # Jump to the frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            # Convert OpenCV's BGR format to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            raise ValueError(
                f"Could not read frame at index {idx} from video {video_path}"
            )

    cap.release()
    return frames
