from itertools import product

import torch
from tqdm import tqdm

from src.aggregation import MeanPooling
from src.prompting import MPVRPrompts, TemplatePrompts
from src.sampling import MotionGuidedSampler, UniformSampler
from src.utils.constants import MODEL_RESULTS_CSV, MPVR_CLASS_DESC_JSON
from src.utils.dataset import VideoDataset
from src.video_classifier import VideoClassifier


def run_experiments():
    dataset = VideoDataset()
    classes = dataset.get_classes()
    num_videos = len(dataset)

    samplers = [UniformSampler(16), MotionGuidedSampler(16)]
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

    video_classifier = VideoClassifier(classes)

    for sampler, aggregator, prompter in tqdm(
        pipeline_combos, desc="Running experiments"
    ):
        pipeline_name = f"{sampler.name}_{aggregator.name}_{prompter.name}"

        df[pipeline_name] = None

        video_classifier.set_sampler(sampler)
        video_classifier.set_aggregator(aggregator)
        video_classifier.set_prompter(prompter)

        for i in tqdm(range(num_videos), desc=f"Running {pipeline_name}", leave=False):
            prediction = video_classifier.classify(
                video_path=dataset.get_video_path(i),
                frame_embeddings=video_embeddings[i],
            )

            df.at[i, pipeline_name] = prediction

    df.to_csv(MODEL_RESULTS_CSV, index=False)
    print(f"\nAll results saved to {MODEL_RESULTS_CSV}")


if __name__ == "__main__":
    run_experiments()
