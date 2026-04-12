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
from src.utils.constants import (
    ACTIVITYNET_CSV,
    ACTIVITYNET_EMBEDDINGS_DIR,
    ACTIVITYNET_VIDEOS_DIR,
    MPVR_ACTIVITYNET_CLASS_DESC_JSON,
    MPVR_CLASS_DESC_JSON,
    RESULTS_DIR,
    TEST_CSV,
    UCF101_EMBEDDINGS_DIR,
    UCF101_VIDEOS_DIR,
)
from src.utils.dataset import VideoDataset
from src.video_classifier import VideoClassifier


def run_experiments():
    # dataset = VideoDataset(
    #     UCF101_EMBEDDINGS_DIR,
    #     UCF101_VIDEOS_DIR,
    #     TEST_CSV,
    # )
    dataset = VideoDataset(
        ACTIVITYNET_EMBEDDINGS_DIR,
        ACTIVITYNET_VIDEOS_DIR,
        ACTIVITYNET_CSV,
    )

    classes = dataset.get_classes()
    num_videos = len(dataset)

    OUTPUT_DIR = RESULTS_DIR / "test_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    samplers = [
        UniformSampler(16),
        # MotionGuidedSampler(16),
    ]
    aggregators = [
        MeanPooling(),
        QueryScoringAggregator(),
    ]
    prompters = [
        TemplatePrompts(classes),
        MPVRPrompts(classes, MPVR_ACTIVITYNET_CLASS_DESC_JSON),
    ]

    # TODO Create matrix for the best method
    # TODO in the manual pipeline, include comparison of frames selected my uniform sampler and one by the Mgsampler. Also include the cumsum graph.
    # NOTE the MGsampler paper times their pipeline when using precomputed motion scores but i compute them, I treat it as inference cost.
    # CHALLENGE visualisating the results with so many classes
    # TODO put full confusion matrix into appendix of the best model

    # Run times: {'UniformSampler16_MeanPooling_TemplatePrompts': 185.21126174926758, 'MGSampler16_MeanPooling_TemplatePrompts': 5594.235224723816, 'UniformSampler16_QueryScoringAggregator_TemplatePrompts': 137.240483045578, 'UniformSampler16_MeanPooling_MPVRPrompts': 127.9118173122406}
    # Processing times for each pipeline (seconds): {'MGSampler16_MeanPooling_TemplatePrompts': 15721.757383823395, 'MGSampler16_QueryScoringAggregator_TemplatePrompts': 16108.781534671783}
    # Processing times for each pipeline (seconds) (ActivityNet): {'UniformSampler16_MeanPooling_TemplatePrompts': 80.9174792766571, 'UniformSampler16_MeanPooling_MPVRPrompts': 56.55835700035095, 'UniformSampler16_QueryScoringAggregator_TemplatePrompts': 56.51149821281433, 'UniformSampler16_QueryScoringAggregator_MPVRPrompts': 56.03473615646362, 'MGSampler16_MeanPooling_TemplatePrompts': 15228.461215496063, 'MGSampler16_MeanPooling_MPVRPrompts': 57.70642137527466, 'MGSampler16_QueryScoringAggregator_TemplatePrompts': 58.624419927597046, 'MGSampler16_QueryScoringAggregator_MPVRPrompts': 58.53251242637634}
    # Processing times for each pipeline (seconds) (ActivityNet): {'UniformSampler120_MeanPooling_TemplatePrompts': 88.68463039398193, 'UniformSampler120_MeanPooling_MPVRPrompts': 72.75543022155762, 'UniformSampler120_QueryScoringAggregator_TemplatePrompts': 87.4105179309845, 'UniformSampler120_QueryScoringAggregator_MPVRPrompts': 88.80076336860657, 'MGSampler120_MeanPooling_TemplatePrompts': 13547.717323064804, 'MGSampler120_MeanPooling_MPVRPrompts': 59.096999406814575, 'MGSampler120_QueryScoringAggregator_TemplatePrompts': 63.71704649925232, 'MGSampler120_QueryScoringAggregator_MPVRPrompts': 63.52103877067566}
    # UCF101 Processing times for each pipeline (seconds): {'MGSampler1_MeanPooling_TemplatePrompts': 2686.7785692214966, 'MGSampler2_MeanPooling_TemplatePrompts': 86.50166296958923, 'MGSampler4_MeanPooling_TemplatePrompts': 85.78026056289673, 'MGSampler8_MeanPooling_TemplatePrompts': 85.98095679283142, 'MGSampler16_MeanPooling_TemplatePrompts': 86.34726309776306, 'MGSampler32_MeanPooling_TemplatePrompts': 86.63276052474976, 'MGSampler64_MeanPooling_TemplatePrompts': 86.33498358726501, 'MGSampler128_MeanPooling_TemplatePrompts': 88.18419122695923}
    # activitynet Processing times for each pipeline (seconds): {'MGSampler1_MeanPooling_TemplatePrompts': 14968.512595891953, 'MGSampler2_MeanPooling_TemplatePrompts': 57.87101864814758, 'MGSampler4_MeanPooling_TemplatePrompts': 58.191577672958374, 'MGSampler8_MeanPooling_TemplatePrompts': 56.745466470718384, 'MGSampler16_MeanPooling_TemplatePrompts': 57.0536322593689, 'MGSampler32_MeanPooling_TemplatePrompts': 60.145596504211426, 'MGSampler64_MeanPooling_TemplatePrompts': 63.78253149986267, 'MGSampler128_MeanPooling_TemplatePrompts': 62.12326240539551, 'UniformSampler1_MeanPooling_TemplatePrompts': 91.17177510261536, 'UniformSampler2_MeanPooling_TemplatePrompts': 74.5673840045929, 'UniformSampler4_MeanPooling_TemplatePrompts': 74.64715814590454, 'UniformSampler8_MeanPooling_TemplatePrompts': 74.74318194389343, 'UniformSampler16_MeanPooling_TemplatePrompts': 71.71035242080688, 'UniformSampler32_MeanPooling_TemplatePrompts': 67.72666692733765, 'UniformSampler64_MeanPooling_TemplatePrompts': 68.92252779006958, 'UniformSampler128_MeanPooling_TemplatePrompts': 82.02165246009827}

    # activitynet Processing times for each pipeline (seconds): {
    # 'UniformSampler16_MeanPooling_TemplatePrompts': 70.34243988990784,
    # 'UniformSampler16_MeanPooling_MPVRPrompts': 59.461488246917725,
    # 'UniformSampler16_QueryScoringAggregator_TemplatePrompts': 60.39933371543884,
    # 'UniformSampler16_QueryScoringAggregator_MPVRPrompts': 60.199368953704834,
    # 'MGSampler16_MeanPooling_TemplatePrompts': 13936.429725885391,
    # 'MGSampler16_MeanPooling_MPVRPrompts': 103.70175337791443,
    # 'MGSampler16_QueryScoringAggregator_TemplatePrompts': 63.031410932540894,
    # 'MGSampler16_QueryScoringAggregator_MPVRPrompts': 61.86304330825806,
    #
    # 'UniformSampler120_MeanPooling_TemplatePrompts': 86.4507966041565,
    # 'UniformSampler120_MeanPooling_MPVRPrompts': 75.63450169563293,
    # 'UniformSampler120_QueryScoringAggregator_TemplatePrompts': 92.67772245407104,
    # 'UniformSampler120_QueryScoringAggregator_MPVRPrompts': 91.24755549430847,
    # 'MGSampler120_MeanPooling_TemplatePrompts': 61.02518820762634,
    # 'MGSampler120_MeanPooling_MPVRPrompts': 60.92303466796875,
    # 'MGSampler120_QueryScoringAggregator_TemplatePrompts': 89.68481278419495,
    # 'MGSampler120_QueryScoringAggregator_MPVRPrompts': 126.48594522476196}

    # Use ranking to compare models, compare models on mAP
    # AUC calculations
    # Manually compare specific videos that models disagree on, identify edge cases
    # mcnemars test to compare their accuracies, briers score with wilcoxon test with holm correction to compare model's calibration (combining both discrimination and calibration -  strictly proper scoring rule)
    # brier scores are all 0.99... which are similar but a mathematical reality of having 200 classes
    # finding optimal softmax temperature for each model and tuning each hyper parameters could change the results and outcomes of these models and improve their performance. Here, they were ran as out-of-the-box and from their original implementations as possible.
    # different datasets both have different types of videos, activity net videos are not standardized, different resolution and frame rate and are unclipped.
    # MGsampler was built,tested on and meant for supervised temporal CNNs and not for zero shot CLIP model - MGS possibly selects blurry confusing images that are independantly given to CLIP which provides bad understandnig, but it assumed a downstream CNNs that could connect the sequence of motion frames.
    # However, QS aggregation did show results despite being focused on video retrieval, so that did generalize well and improve accuracy when combined with MPVR.
    # compare models based on accuracy, calibration, and inference speed

    # What to do with old interim report results and analysis? Remove / use ?
    # Should i go back and reference the Activitynet dataset when explaining the plan for the project despite it being an addition later on?
    # calibration metric im using is brier score, the scores are all very close but tests show significantly different, is this enough to claim some models are better calibrated than others?
    # should i continue with these weird looking metrics?
    # limitation: not temperature scaling per model done - so calibration could be better
    # should i show the bar chart because it looks nice but repeats information that is in the results table?
    # should i include exact p values in a table or something or is the heatmap enough to just show there is a statistical difference?
    # c

    pipeline_combos = list(product(samplers, aggregators, prompters))

    # Load embeddings once to save time
    video_embeddings = []

    for i in tqdm(range(num_videos), desc="Loading embeddings", leave=False):
        try:
            video_embeddings.append(dataset.get_embeddings(i))
        except FileNotFoundError:
            video_embeddings.append(None)

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
        prob_preds = {cls: [float("nan")] * num_videos for cls in classes}
        prob_df = pd.DataFrame(prob_preds)

        # Combine the metadata df with the empty probability columns
        pipeline_df = pd.concat([dataset.df.copy(), prob_df], axis=1)

        video_classifier.set_sampler(sampler)
        video_classifier.set_aggregator(aggregator)
        video_classifier.set_prompter(prompter)

        # Time running the pipeline on all videos
        start = time.time()

        for i in tqdm(range(num_videos), desc=f"Running {pipeline_name}", leave=False):
            try:
                probabilities = video_classifier.classify(
                    video_path=dataset.get_video_path(i),
                    frame_embeddings=video_embeddings[i],
                )

                # set the class probabilities
                pipeline_df.loc[i, classes] = probabilities.cpu().numpy()
            except ValueError as e:
                print(
                    f"Error processing video index {i} ({dataset.get_video_path(i)}): {e}"
                )
                continue

        model_times[pipeline_name] = time.time() - start

        pipeline_df.to_csv(csv_path, index=False)
        print(f"\n\nResults for {pipeline_name} saved to {csv_path}\n\n")

    print("=" * 10)
    print(f"\nAll results are available in: {OUTPUT_DIR}")
    print("\nProcessing times for each pipeline (seconds): " + str(model_times))


if __name__ == "__main__":
    run_experiments()
