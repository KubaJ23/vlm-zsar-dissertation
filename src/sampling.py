from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    @abstractmethod
    def sample(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass


class UniformSampler(Sampler):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def sample(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Uniformly samples a fixed number of frames from a video's embeddings.

        The number of samples is determined by `self.num_samples`.
        If `self.num_samples` is greater than the number of available frames,
        frames will be repeated.

        Args:
            embeddings: The input tensor of shape (num_frames, embed_dim).

        Returns:
            The new tensor of shape (self.num_samples, embed_dim).
        """
        num_frames = embeddings.shape[0]

        indices_float = torch.linspace(
            start=0,
            end=num_frames - 1,
            steps=self.num_samples,
            device=embeddings.device,
        )

        # Convert to long (for indexing) and clamp (for safety)
        indices_int = torch.round(indices_float).long()
        indices_int = torch.clamp(indices_int, 0, num_frames - 1)

        sampled_embeddings = embeddings[indices_int]

        return sampled_embeddings


class MotionGuidedSampler(Sampler):
    def sample(self, embeddings: torch.Tensor) -> torch.Tensor: ...
