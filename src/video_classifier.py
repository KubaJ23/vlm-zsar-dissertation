import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.aggregation import Aggregator
from src.prompting import ClassPrompter
from src.sampling import Sampler
from src.utils import clip


def classify_videos(
    pipeline: tuple[Sampler, Aggregator, ClassPrompter],
    video_embeddings: list[torch.Tensor],
    classes: list[str],
):
    sampler = pipeline[0]
    aggregator = pipeline[1]
    prompter = pipeline[2]

    class_to_text_embedding = prompter.get_prompt_embeddings()

    text_embeddings = torch.stack([class_to_text_embedding[cls] for cls in classes]).to(
        clip.DEVICE
    )

    # Normalize text embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    predictions = []

    with torch.no_grad():
        for frame_embeddings in video_embeddings:
            frame_embeddings = frame_embeddings.to(clip.DEVICE)

            sampled = sampler.sample(frame_embeddings)
            sampled = sampled.to(clip.DEVICE)

            video_embedding = aggregator.aggregate(sampled)

            video_embedding = F.normalize(video_embedding.to(clip.DEVICE), p=2, dim=-1)

            # Calculate cosine similarity to get a tensor of similarities between the video and each class
            similarities = video_embedding @ text_embeddings.T

            predicted_label = classes[similarities.argmax().item()]

            predictions.append(predicted_label)

    return predictions
