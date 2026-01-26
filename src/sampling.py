import random
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

# Handle MGSampler dependency: 'compare_ssim' was moved in newer scikit-image versions.
# We map it here so the original MGSampler logic works without changes.
try:
    from skimage.measure import compare_ssim
except ImportError:
    from skimage.metrics import structural_similarity as compare_ssim


class Sampler(ABC):
    @abstractmethod
    def sample(self, video_path: Path) -> list[int]:
        """
        Args:
            video_path: The input video path.

        Returns:
            A list of selected frame index positions.
        """
        pass


class UniformSampler(Sampler):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.name = f"UniformSampler{num_samples}"

    def sample(self, video_path: Path) -> list[int]:
        """
        Uniformly samples a fixed number of frame indices from a video.

        The number of samples is determined by `self.num_samples`.
        If `self.num_samples` is greater than the number of available frames,
        indices will be repeated.
        """

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError("Could not open video")

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        if num_frames <= 0:
            raise ValueError("Video contains no frames")

        indices_float = torch.linspace(
            start=0, end=num_frames - 1, steps=self.num_samples
        )

        # Convert to long (for indexing) and clamp (for safety)
        indices_int = torch.round(indices_float).long()
        indices_int = torch.clamp(indices_int, 0, num_frames - 1)

        return indices_int.tolist()


# Adapted from the MGSampler implementation:
# https://github.com/MCG-NJU/MGSampler/tree/main
# Full citation is provided in the project README.
class MotionGuidedSampler(Sampler):
    def __init__(self, num_samples: int, test_mode: bool = True):
        self.num_samples = num_samples
        self.test_mode = test_mode
        self.name = f"MGSampler{num_samples}"

    def _get_img_diff(self, video_path):
        """
        Direct implementation of the logic from MGSampler's file `generate_img_diff.py`
        """
        img = []
        diff_scores = []

        try:
            vid = imageio.get_reader(str(video_path), "ffmpeg")
            for num, im in enumerate(vid):
                img.append(im)

            if len(img) < 2:
                return []

            for i in range(len(img) - 1):
                tmp1 = cv2.cvtColor(img[i], cv2.COLOR_RGB2GRAY)
                tmp2 = cv2.cvtColor(img[i + 1], cv2.COLOR_RGB2GRAY)

                (score, diff) = compare_ssim(tmp1, tmp2, full=True)
                score = 1 - score
                diff_scores.append(score)

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return []

        return diff_scores

    def sample(self, video_path: Path) -> list[int]:
        """
        Direct implementation of the logic from MGSampler's file `loading.py` (SampleFrames class)
        """
        # calculate motion scores (simulating loading 'img_diff' from JSON)
        diff_scores = self._get_img_diff(video_path)

        if not diff_scores:
            raise ValueError("No frame differences available for the video.")

        # process scores, quare root and normalization from `loading.py`
        diff_scores = np.array(diff_scores)
        diff_scores = np.power(diff_scores, 0.5)
        diff_sum = np.sum(diff_scores)

        # check for division by zero
        if diff_sum == 0:
            diff_scores = np.ones_like(diff_scores) / len(diff_scores)
        else:
            diff_scores = diff_scores / diff_sum

        # Accumulate scores (Cumulative Distribution Function)
        count = 0
        pic_diff = []
        for i in range(len(diff_scores)):
            count = count + diff_scores[i]
            pic_diff.append(count)

        # helper function from `loading.py`
        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx + 1)
            except ValueError:
                return 0

        choose_index = []

        # select indices based on distribution
        # original code hardcoded loops for 16 clips, now it uses `self.num_samples` to be more general
        # while keeping the mathematical logic (center sampling vs random bin sampling).

        step = 1.0 / self.num_samples

        if self.test_mode:
            # sample the center of each probability bin (deterministic)
            half_step = step / 2.0
            for i in range(self.num_samples):
                target_prob = half_step + (i * step)
                choose_index.append(find_nearest(pic_diff, target_prob))
        else:
            # sample randomly within each probability bin (stochastic)
            for i in range(self.num_samples):
                start = i * step
                end = (i + 1) * step
                choose_index.append(find_nearest(pic_diff, random.uniform(start, end)))

        # clamp to valid range, should not be necessary but acts as a double check (total frames - 1)
        max_frame_idx = len(diff_scores)
        choose_index = [min(idx, max_frame_idx) for idx in choose_index]

        return choose_index
