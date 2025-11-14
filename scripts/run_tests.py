import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.aggregation import MeanPooling
from src.prompting import TemplatePrompts
from src.sampling import UniformSampler
from src.utils import clip
from src.utils.dataset import UCF101SubsetDataset


def run_test_pipeline():
    """
    Executes the full zero-shot action recognition pipeline on the test dataset.
    """
    # --- 1. Configuration & Setup ---
    NUM_SAMPLES_PER_VIDEO = 16

    # Instantiate the components of our pipeline
    sampler = UniformSampler(num_samples=NUM_SAMPLES_PER_VIDEO)
    aggregator = MeanPooling()
    prompter = TemplatePrompts()

    print("Setting up dataset and text prompts...")
    dataset = UCF101SubsetDataset(split="test")
    num_videos = len(dataset)

    classes = sorted(dataset.df["class_name"].unique().tolist())

    class_to_text_embedding = prompter.get_prompt_embeddings(classes)

    text_embeddings = torch.stack([class_to_text_embedding[cls] for cls in classes]).to(
        clip.DEVICE
    )

    # Normalize text embeddings for cosine similarity
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    print(f"Setup complete. Found {num_videos} videos and {len(classes)} classes.")

    # --- 2. Video Processing and Classification ---
    correct_predictions = 0

    for i in tqdm(range(num_videos), desc="Classifying Videos"):
        # Get ground truth label and pre-computed frame embeddings
        ground_truth_label = dataset.df.iloc[i]["class_name"]
        frame_embeddings = dataset.get_embeddings(i).to(clip.DEVICE)

        sampled_embeddings = sampler.sample(frame_embeddings)

        video_embedding = aggregator.aggregate(sampled_embeddings)

        video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        # d. Calculate cosine similarity
        # The result is a tensor of similarities between the video and each class
        similarities = video_embedding @ text_embeddings.T

        prediction_index = similarities.argmax().item()
        predicted_label = classes[prediction_index]

        if predicted_label == ground_truth_label:
            correct_predictions += 1

    # --- 3. Evaluation ---
    accuracy = (correct_predictions / num_videos) * 100
    print("\n--- Test Results ---")
    print(f"Total Videos: {num_videos}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    run_test_pipeline()
