from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Matrix with dimensions num_frames x embed_dim
            text_embeddings: required for Query-Scoring, contains embeddings of classes (ordered). dimensions classes x dim
        """
        pass


class MeanPooling(Aggregator):
    def __init__(self):
        self.name = "MeanPooling"

    def aggregate(
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        return frame_embeddings.mean(dim=0)


class QueryScoringAggregator:
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8):
        self.name = "QueryScoringAggregator"
        self.temperature = temperature
        self.eps = eps

    def aggregate(
        self, frame_embeddings: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        assert frame_embeddings.ndim == 2
        assert text_embeddings.ndim == 2

        # Match original shape: single video batch
        vid_embeds = frame_embeddings.unsqueeze(0)

        # L2 normalisation
        t_norm = text_embeddings.norm(dim=1)[:, None]
        v_norm = vid_embeds.norm(dim=2)[:, :, None]

        text_embeds_norm = text_embeddings / torch.max(
            t_norm, self.eps * torch.ones_like(t_norm)
        )
        vid_embeds_norm = vid_embeds / torch.max(
            v_norm, self.eps * torch.ones_like(v_norm)
        )

        # Text–frame similarity
        sim_mt = torch.einsum("a d, b v d -> a b v", text_embeds_norm, vid_embeds_norm)

        # Query scores over frames
        scores = torch.softmax(sim_mt / self.temperature, dim=2)

        # Weighted aggregation of original frame embeddings
        vid_embeds_weighted = torch.einsum(
            "b v d, a b v -> a b v d", vid_embeds, scores
        )
        vid_embed_final = vid_embeds_weighted.sum(dim=2)

        # Normalise aggregated embeddings
        vf_norm = vid_embed_final.norm(dim=2)[:, :, None]
        vid_embed_final_norm = vid_embed_final / torch.max(
            vf_norm, self.eps * torch.ones_like(vf_norm)
        )

        # Final similarity used for class selection
        sims = torch.einsum("a d, a b d -> a b", text_embeds_norm, vid_embed_final_norm)

        best_class = sims[:, 0].argmax()

        # Return embedding for selected class
        return vid_embed_final_norm[best_class, 0]
