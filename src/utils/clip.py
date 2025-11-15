import torch
from transformers import CLIPModel, CLIPProcessor

# Set up model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    MODEL_ID, trust_remote_code=False, use_safetensors=True
).to(DEVICE)

processor = CLIPProcessor.from_pretrained(MODEL_ID)
