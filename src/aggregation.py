from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass


class MeanPooling(Aggregator):
    def __init__(self):
        self.name = "MeanPooling"

    def aggregate(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.mean(dim=0)


class SimilarityWeightedAggregator(Aggregator):
    def aggregate(self, embeddings): ...
