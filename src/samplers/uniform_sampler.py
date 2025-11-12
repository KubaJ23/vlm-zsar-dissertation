import numpy as np


def sample_uniformly(embeddings: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Uniformly samples a fixed number of frames from a video's embeddings.

    If num_samples > num_frames, it will repeat frames.

    Args:
        embeddings: The input array of shape (num_frames, embed_dim).
        num_samples: The target number of frames to sample.

    Returns:
        The new array of shape (num_samples, embed_dim).
    """
    num_frames = embeddings.shape[0]

    indices_float = np.linspace(0, num_frames - 1, num=num_samples)

    indices_int = np.round(indices_float).astype(int)

    # Handle out of bounds indexes due to rounding
    indices_int = np.clip(indices_int, 0, num_frames - 1)

    sampled_embeddings = embeddings[indices_int]

    return sampled_embeddings
