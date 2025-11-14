# This file adapts or copies code from the official ActionCLIP implementation.
# Original source: https://github.com/sallymmx/ActionCLIP/blob/master/utils/Text_Prompt.py
#
# Original paper citation:
# @article{wang2021actionclip,
#   title={Actionclip: A new paradigm for video action recognition},
#   author={Wang, Mengmeng and Xing, Jiazheng and Liu, Yong},
#   journal={arXiv preprint arXiv:2109.08472},
#   year={2021}
# }

from abc import ABC, abstractmethod

import torch

from . import utils


class ClassPrompter(ABC):
    @abstractmethod
    def get_prompt_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass


class TemplatePrompts(ClassPrompter):
    _templates = [
        "a photo of action {}",
        "a picture of action {}",
        "Human action of {}",
        "{}, an action",
        "{} this is an action",
        "{}, a video of action",
        "Playing action of {}",
        "{}",
        "Playing a kind of action, {}",
        "Doing a kind of action, {}",
        "Look, the human is {}",
        "Can you recognise the action of {}?",
        "Video classification of {}",
        "A video of {}",
        "The man is {}",
        "The woman is {}",
    ]

    def get_prompt_embeddings(self, classes: list[str]) -> dict[str, torch.Tensor]:
        class_to_embedding = {}

        for cls in classes:
            descriptions = [t.format(cls) for t in self._templates]

            texts = utils.clip.processor(
                text=descriptions, return_tensors="pt", padding=True
            ).to(utils.clip.DEVICE)

            with torch.no_grad():
                embeddings = utils.clip.model.get_text_features(**texts)
                mean_embedding = embeddings.mean(dim=0)
                class_to_embedding[cls] = mean_embedding

        return class_to_embedding


class MPVRPrompts(ClassPrompter):
    def get_prompt_embeddings(self, classes: list[str]) -> dict[str, torch.Tensor]: ...
