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
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Calculate the video representation
        vid_embed = frame_embeddings.mean(dim=0, keepdim=True)  # 1 x d

        # L2 normalisation for cosine similarity
        v_norm = vid_embed / torch.max(
            vid_embed.norm(dim=1, keepdim=True),
            self.eps * torch.ones(1, 1, device=vid_embed.device),
        )
        t_norm = text_embeddings / torch.max(
            text_embeddings.norm(dim=1, keepdim=True),
            self.eps * torch.ones(1, 1, device=text_embeddings.device),
        )

        # Calculate similarities and convert to probabilities
        sims = torch.matmul(v_norm, t_norm.t()).squeeze(0)
        prediction_temp = 0.01
        return torch.softmax(sims / prediction_temp, dim=0)


# Adapted from the "A CLIP-Hitchhiker’s Guide to Long Video Retrieval" implementation:
# https://github.com/m-bain/clip-hitchhiker
# Full citation is provided in the project README.
class QueryScoringAggregator(Aggregator):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        self.name = "QueryScoringAggregator"
        self.temperature = temperature
        self.eps = eps
        self.class_similarities = None

    def aggregate(
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        assert frame_embeddings.ndim == 2
        assert text_embeddings.ndim == 2

        vid_embeds = frame_embeddings.unsqueeze(0)

        # L2 normalisation
        t_norm_embeds = text_embeddings / torch.max(
            text_embeddings.norm(dim=1, keepdim=True),
            self.eps * torch.ones(1, 1, device=text_embeddings.device),
        )
        vid_embeds_norm = vid_embeds / torch.max(
            vid_embeds.norm(dim=2, keepdim=True),
            self.eps * torch.ones(1, 1, device=vid_embeds.device),
        )

        # Text–frame similarity
        sim_mt = torch.einsum("a d, b v d -> a b v", t_norm_embeds, vid_embeds_norm)

        # Query scores over frames
        scores = torch.softmax(sim_mt / self.temperature, dim=2)

        # Weighted aggregation
        vid_embed_final = torch.einsum(
            "b v d, a b v -> a b v d", vid_embeds, scores
        ).sum(dim=2)

        # Normalise aggregated video embeddings
        vid_embed_final_norm = vid_embed_final / torch.max(
            vid_embed_final.norm(dim=2, keepdim=True),
            self.eps * torch.ones(1, 1, device=vid_embed_final.device),
        )

        sims = torch.einsum("a d, a b d -> a b", t_norm_embeds, vid_embed_final_norm)
        self.class_similarities = sims[:, 0]

        prediction_temp = 0.01
        return torch.softmax(self.class_similarities / prediction_temp, dim=0)
