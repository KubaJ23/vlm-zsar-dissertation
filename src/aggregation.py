from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            frame_embeddings: Matrix with dimensions num_frames x embed_dim
            text_embeddings: Matrix with dimensions num_classes x embed_dim
        Returns:
            A vector probabilities of shape (num_classes,)
        """
        pass


class MeanPooling(Aggregator):
    def __init__(self, eps: float = 1e-8):
        self.name = "MeanPooling"
        self.eps = eps

    def aggregate(
        self,
        frame_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate the video representation
        vid_embed = frame_embeddings.mean(dim=0, keepdim=True)  # 1 x d

        # normalisation for cosine similarity
        v_norm = F.normalize(vid_embed, eps=self.eps)
        t_norm = F.normalize(text_embeddings, eps=self.eps)

        # Calculate similarities and convert to probabilities
        sims = (v_norm @ t_norm.T).squeeze(0)
        prediction_temp = 0.01
        return torch.softmax(sims / prediction_temp, dim=0)


# Highly adapted from the "A CLIP-Hitchhiker’s Guide to Long Video Retrieval" implementation:
# https://github.com/m-bain/clip-hitchhiker
# Full citation is provided in the project README.
# This integration removed the batching that was included in the original implementation as well as a different mechanism for matrix multiplication (original used torch.einsum which was difficult to understand and unclear).
class QueryScoringAggregator(Aggregator):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        self.name = "QueryScoringAggregator"
        self.temperature = temperature
        self.eps = eps
        self.class_similarities = None
        self.saved_frame_weights = None

    def aggregate(
        self,
        frame_embeddings: torch.Tensor,  # num_frames x embed_dim
        text_embeddings: torch.Tensor,  # num_classes x embed_dim
    ) -> torch.Tensor:
        assert frame_embeddings.ndim == 2
        assert text_embeddings.ndim == 2

        # normalise text embeddings for cosine similarity
        t_norm_embeds = F.normalize(text_embeddings, eps=self.eps)

        # normalise video embeddings for cosine similarity
        vid_embeds_norm = F.normalize(frame_embeddings, eps=self.eps)

        # calculate cosine similarity between frames and text embeddings
        sim_mt = t_norm_embeds @ vid_embeds_norm.T

        # turn similarities into probability weights (num_classes x num_frames)
        scores = torch.softmax(sim_mt / self.temperature, dim=1)

        # get weighted sum of frame embeddings into a single embedding per class (num_classes x embed_dim)
        vid_embed_final = scores @ frame_embeddings

        # normalise the new video embedding for each class
        vid_embed_final_norm = F.normalize(vid_embed_final, eps=self.eps)

        # calculate similarity between text queries and the aggregated video vector (num_classes,1)
        sims = (t_norm_embeds * vid_embed_final_norm).sum(dim=1)
        self.class_similarities = sims

        self.save_frame_weights_for_most_probable_class(self.class_similarities, scores)

        # return final probabilities for each class
        prediction_temp = 0.01
        return torch.softmax(self.class_similarities / prediction_temp, dim=0)

    def save_frame_weights_for_most_probable_class(self, class_similarities, scores):
        """Saves frame weights for the most probable class for later analysis"""
        # get index of most probable class
        most_probable_class_idx = torch.argmax(class_similarities).item()

        # get the frame weights for the most probable class
        frame_weights = scores[most_probable_class_idx, :].cpu().numpy()

        self.saved_frame_weights = frame_weights
