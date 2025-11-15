import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.aggregation import MeanPooling
from src.prompting import TemplatePrompts
from src.sampling import UniformSampler
from src.utils import clip
from src.utils.constants import MODEL_RESULTS_BASELINE
from src.utils.dataset import UCF101SubsetDataset


def run_test_pipeline():
    NUM_SAMPLES_PER_VIDEO = 16

    sampler = UniformSampler(num_samples=NUM_SAMPLES_PER_VIDEO)
    aggregator = MeanPooling()
    prompter = TemplatePrompts()

    dataset = UCF101SubsetDataset()
    num_videos = len(dataset)

    classes = sorted(dataset.df["label"].unique().tolist())

    class_to_text_embedding = prompter.get_prompt_embeddings(classes)

    text_embeddings = torch.stack([class_to_text_embedding[cls] for cls in classes]).to(
        clip.DEVICE
    )

    # Normalize text embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    print(f"Found {num_videos} videos and {len(classes)} classes.")

    correct_predictions = 0

    for idx, row in tqdm(
        dataset.df.iterrows(), total=num_videos, desc="Classifying Videos"
    ):
        true_label = row["label"]
        frame_embeddings = dataset.get_embeddings(idx).to(clip.DEVICE)

        sampled_embeddings = sampler.sample(frame_embeddings)

        video_embedding = aggregator.aggregate(sampled_embeddings)

        video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        # Calculate cosine similarity to get a tensor of similarities between the video and each class
        similarities = video_embedding @ text_embeddings.T

        prediction_index = similarities.argmax().item()
        predicted_label = classes[prediction_index]

        dataset.df.at[idx, "baseline_prediction"] = predicted_label

        if predicted_label == true_label:
            correct_predictions += 1

    dataset.df.to_csv(MODEL_RESULTS_BASELINE, index=False)

    accuracy = (correct_predictions / num_videos) * 100
    print()
    print("Completed running all tests.")
    print(f"Total Videos: {num_videos}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    run_test_pipeline()
