import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils import clip


class ClassPrompter(ABC):
    @abstractmethod
    def get_prompt_embeddings(self) -> dict[str, torch.Tensor]:
        pass


# Adapted from the ActionCLIP implementation:
# https://github.com/sallymmx/ActionCLIP/blob/master/utils/Text_Prompt.py
# Full citation is provided in the project README.
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

    def __init__(self, classes: list[str]):
        self.name = "TemplatePrompts"

        class_to_embedding = {}

        for cls in classes:
            descriptions = [t.format(cls) for t in self._templates]

            texts = clip.processor(
                text=descriptions, return_tensors="pt", padding=True
            ).to(clip.DEVICE)

            with torch.no_grad():
                embeddings = clip.model.get_text_features(**texts)
                mean_embedding = embeddings.mean(dim=0)
                class_to_embedding[cls] = mean_embedding
        self.class_to_embedding = class_to_embedding

    def get_prompt_embeddings(self) -> dict[str, torch.Tensor]:
        return self.class_to_embedding


class MPVRPrompts(ClassPrompter):
    """Generates text embeddings for each class using MPVR-style descriptions from a JSON file."""

    def __init__(self, classes: list[str], descriptions_path: Path):
        self.name = "MPVRPrompts"
        self.class_to_embedding: dict[str, torch.Tensor] = {}

        if not descriptions_path.exists():
            raise FileNotFoundError(f"Descriptions file not found: {descriptions_path}")

        with descriptions_path.open("r") as f:
            original_data = json.load(f)

        descriptions_data = {}
        for class_name, descriptions in original_data.items():
            processed_key = class_name.title().replace(" ", "")
            descriptions_data[processed_key] = descriptions

        for cls in tqdm(classes, desc="Generating MPVR embeddings"):
            descriptions = descriptions_data.get(cls)
            if not descriptions:
                raise ValueError(f"No descriptions found for class: {cls}")

            texts = clip.processor(
                text=descriptions, return_tensors="pt", padding=True, truncation=True
            ).to(clip.DEVICE)

            with torch.no_grad():
                embeddings = clip.model.get_text_features(**texts)
                self.class_to_embedding[cls] = embeddings.mean(dim=0)

    def get_prompt_embeddings(self) -> dict[str, torch.Tensor]:
        return self.class_to_embedding
