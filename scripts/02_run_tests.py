from itertools import product

import torch
from tqdm import tqdm

from src.aggregation import MeanPooling
from src.prompting import MPVRPrompts, TemplatePrompts
from src.sampling import UniformSampler
from src.utils.constants import MODEL_RESULTS_CSV, MPVR_CLASS_DESC_JSON
from src.utils.dataset import VideoDataset
from src.video_classifier import classify_videos


def run_experiments():
    dataset = VideoDataset()
    classes = dataset.get_classes()
    num_videos = len(dataset)

    samplers = [UniformSampler(64)]

    aggregators = [MeanPooling()]

    prompters = [TemplatePrompts(classes), MPVRPrompts(classes, MPVR_CLASS_DESC_JSON)]

    # Produce all pipeline combinations
    pipeline_combos = list(product(samplers, aggregators, prompters))

    # Load embeddings once
    video_embeddings = [
        dataset.get_embeddings(i)
        for i in tqdm(range(num_videos), desc="Loading embeddings")
    ]

    df = dataset.df.copy()

    for sampler, aggregator, prompter in tqdm(
        pipeline_combos, desc="Running experiments"
    ):
        pipeline = (sampler, aggregator, prompter)

        pipeline_name = f"{sampler.name}_{aggregator.name}_{prompter.name}"

        with torch.no_grad():
            predictions = classify_videos(
                pipeline=pipeline,
                video_embeddings=video_embeddings,
                classes=classes,
            )

        df[pipeline_name] = predictions

    df.to_csv(MODEL_RESULTS_CSV, index=False)
    print(f"\nAll results saved to {MODEL_RESULTS_CSV}")


if __name__ == "__main__":
    run_experiments()
