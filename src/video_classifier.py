from pathlib import Path

import torch
import torch.nn.functional as functional

from src.aggregation import Aggregator
from src.prompting import ClassPrompter
from src.sampling import Sampler
from src.utils import clip


class VideoClassifier:
    def __init__(
        self,
        classes: list[str],
    ):
        self.classes = classes
        self.set_sampler(None)
        self.set_aggregator(None)
        self.set_prompter(None)

    def set_sampler(self, sampler: Sampler) -> "VideoClassifier":
        self.sampler = sampler
        return self

    def set_aggregator(self, aggregator: Aggregator) -> "VideoClassifier":
        self.aggregator = aggregator
        return self

    def set_prompter(self, prompter: ClassPrompter) -> "VideoClassifier":
        self.prompter = prompter
        return self

    def classify(
        self,
        video_path: Path,
        frame_embeddings: torch.Tensor,
    ) -> str:
        class_to_text_embedding = self.prompter.get_prompt_embeddings_map()

        text_embeddings = torch.stack(
            [class_to_text_embedding[cls] for cls in self.classes]
        ).to(clip.DEVICE)

        # Normalize text embeddings for cosine similarity
        text_embeddings = functional.normalize(text_embeddings, p=2, dim=-1)

        with torch.no_grad():
            frame_embeddings = frame_embeddings.to(clip.DEVICE)

            selected_indexes = self.sampler.sample(video_path)

            num_embeddings = frame_embeddings.shape[0]

            if num_embeddings == 0:
                raise ValueError("No frame embeddings available for the video.")

            # Clamp every index to be at most (num_embeddings - 1)
            selected_indexes = [
                min(idx, num_embeddings - 1) for idx in selected_indexes
            ]

            sampled = frame_embeddings[selected_indexes]

            sampled = sampled.to(clip.DEVICE)

            video_embedding = self.aggregator.aggregate(sampled)

            video_embedding = functional.normalize(
                video_embedding.to(clip.DEVICE), p=2, dim=-1
            )

            # Calculate cosine similarity to get a tensor of similarities between the video and each class
            similarities = video_embedding @ text_embeddings.T

            predicted_label = self.classes[similarities.argmax().item()]

            return predicted_label
