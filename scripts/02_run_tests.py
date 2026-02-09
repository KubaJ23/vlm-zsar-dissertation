import os
import time
from itertools import product
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.aggregation import MeanPooling, QueryScoringAggregator
from src.prompting import MPVRPrompts, TemplatePrompts
from src.sampling import MotionGuidedSampler, UniformSampler
from src.utils.constants import MPVR_CLASS_DESC_JSON, RESULTS_DIR
from src.utils.dataset import VideoDataset
from src.video_classifier import VideoClassifier


def run_experiments():
    dataset = VideoDataset()
    classes = dataset.get_classes()
    num_videos = len(dataset)

    OUTPUT_DIR = RESULTS_DIR / "test_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samplers = [UniformSampler(16), MotionGuidedSampler(16)]
    aggregators = [MeanPooling(), QueryScoringAggregator()]
    prompters = [TemplatePrompts(classes), MPVRPrompts(classes, MPVR_CLASS_DESC_JSON)]

    # TODO: get the uncertainties and statistical tests on results of experiemnts
    # TODO: find the processing time of each method.

    # TODO: Scatter plot of acccuracy vs processing time for each method.
    # TODO Why did the best method fail?
    # TODO Create matrix for the best method
    # TODO Create table of the most confused pairs of classes
    # TODO Could create a difference matrix between 2 models

    # samplers = [UniformSampler(1), UniformSampler(2)]
    # aggregators = [MeanPooling()]
    # prompters = [TemplatePrompts(classes)]

    pipeline_combos = [
        (samplers[0], aggregators[0], prompters[0]),
        (samplers[1], aggregators[0], prompters[0]),
        (samplers[0], aggregators[1], prompters[0]),
        (samplers[0], aggregators[0], prompters[1]),
    ]

    # pipeline_combos = list(product(samplers, aggregators, prompters))

    # Load embeddings once to save time
    video_embeddings = [
        dataset.get_embeddings(i)
        for i in tqdm(range(num_videos), desc="Loading embeddings", leave=False)
    ]

    video_classifier = VideoClassifier(classes)
    model_times = {}

    for sampler, aggregator, prompter in tqdm(
        pipeline_combos, desc="Running experiments", leave=False
    ):
        pipeline_name = f"{sampler.name}_{aggregator.name}_{prompter.name}"
        csv_path = OUTPUT_DIR / f"{pipeline_name}.csv"

        if csv_path.exists():
            print(
                f"Results for {pipeline_name} already exist at {csv_path}, skipping..."
            )
            continue

        # Dataframe to store probabilities of each class for each video for this pipeline
        prob_data = {cls: [0.0] * num_videos for cls in classes}
        prob_df = pd.DataFrame(prob_data)

        # Combine the metadata df with the empty probability columns
        pipeline_df = pd.concat([dataset.df.copy(), prob_df], axis=1)

        video_classifier.set_sampler(sampler)
        video_classifier.set_aggregator(aggregator)
        video_classifier.set_prompter(prompter)

        # Time running the pipeline on all videos
        start = time.time()

        for i in tqdm(range(num_videos), desc=f"Running {pipeline_name}", leave=False):
            probabilities = video_classifier.classify(
                video_path=dataset.get_video_path(i),
                frame_embeddings=video_embeddings[i],
            )

            # set the class probabilities
            pipeline_df.loc[i, classes] = probabilities.cpu().numpy()

        model_times[pipeline_name] = time.time() - start

        pipeline_df.to_csv(csv_path, index=False)
        print(f"Results for {pipeline_name} saved to {csv_path}")

    print("=" * 50)
    print(f"\nAll experimental results are available in: {OUTPUT_DIR}")
    print("\nProcessing times for each pipeline (seconds): " + str(model_times))


if __name__ == "__main__":
    run_experiments()
