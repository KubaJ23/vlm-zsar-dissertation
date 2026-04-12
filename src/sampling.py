import random
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu

# MGSampler dependency: compare_ssim was moved in newer scikit-image versions.
from skimage.metrics import structural_similarity


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

        indexes_float = torch.linspace(
            start=0, end=num_frames - 1, steps=self.num_samples
        )

        # need integers  for indexes, so round to nearest int
        indexes = torch.round(indexes_float).long()

        return indexes.tolist()


# Adapted from the MGSampler implementation:
# https://github.com/MCG-NJU/MGSampler/tree/main (Apache License 2.0)
# Full citation is provided in the project README.


# Optimisations made:
# - limiting resolution of video frames for the motion salience calculations
# - using decord for faster video reading
# - using numpy's built in cumulative sum method.
# - cache the motion salience scores for already processed videos
class MotionGuidedSampler(Sampler):
    _cache = {}  # Static variable so all MGsampler instances can use cache to avoid recomputation.

    def __init__(self, num_samples: int, test_mode: bool = True):
        self.num_samples = num_samples
        self.test_mode = test_mode
        self.name = f"MGSampler{num_samples}"

    def _get_img_diff(self, video_path):
        """
        Implementation of the logic from MGSampler's file `generate_img_diff.py` (though slightly more optimised to reduce experiment runtimes)
        """
        diff_scores = []
        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            num_frames = len(vr)

            if num_frames < 2:
                return []

            # reduce resolution to speed up the calculations (high resolution not necessary for detecting motion salience)
            target_size = (224, 224)

            prev_gray = cv2.resize(
                cv2.cvtColor(vr[0].asnumpy(), cv2.COLOR_RGB2GRAY), target_size
            )

            for i in range(1, num_frames):
                curr_gray = cv2.resize(
                    cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_RGB2GRAY), target_size
                )

                # get the SSIM score between the frames
                (score, _) = structural_similarity(prev_gray, curr_gray, full=True)
                diff_scores.append(1.0 - score)

                prev_gray = curr_gray

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return []

        return diff_scores

    def sample(self, video_path: Path) -> list[int]:
        """
        Implementation of the logic from MGSampler's file `loading.py` (SampleFrames class)
        """
        # calculate motion scores
        # diff_scores = self._get_img_diff(video_path)

        video_key = str(video_path)

        if video_key in MotionGuidedSampler._cache:
            diff_scores = MotionGuidedSampler._cache[video_key]
        else:
            diff_scores = self._get_img_diff(video_path)
            MotionGuidedSampler._cache[video_key] = diff_scores

        if not diff_scores:
            raise ValueError("No frame differences available for the video.")

        # process scores, quare root and normalization from `loading.py`
        diff_scores = np.power(np.array(diff_scores), 0.5)
        diff_sum = np.sum(diff_scores)

        if diff_sum == 0:
            diff_scores = np.ones_like(diff_scores) / len(diff_scores)
        else:
            diff_scores = diff_scores / diff_sum

        cumsum_diff = np.cumsum(diff_scores)

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
        # original code hardcoded loops for 16 clips, now I use `self.num_samples` to be more general
        # while keeping the mathematical logic (center sampling vs random bin sampling).

        step = 1.0 / self.num_samples

        if self.test_mode:
            # sample the center of each probability bin (deterministic)
            half_step = step / 2.0
            for i in range(self.num_samples):
                target_prob = half_step + (i * step)
                choose_index.append(find_nearest(cumsum_diff, target_prob))
        else:
            # sample randomly within each probability bin (stochastic)
            for i in range(self.num_samples):
                start = i * step
                end = (i + 1) * step
                choose_index.append(
                    find_nearest(cumsum_diff, random.uniform(start, end))
                )

        # clamp to valid range, should not be necessary but acts as a double check (total frames - 1)
        max_frame_idx = len(diff_scores)
        choose_index = [min(idx, max_frame_idx) for idx in choose_index]

        return choose_index
