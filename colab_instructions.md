# Generating Embeddings Dataset From UCF101

This documents explains how to generate the **UCF101 embeddings dataset** using Google Colab. These are the steps that were followed to convert the original Kaggle dataset into `.pt` embedding files per video using the **CLIP model**.

## Process Overview

1. **Download the UCF101 dataset from Kaggle**
    - [UCF101 Action Recognition Dataset](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition).

2. **Generate subset dataset**
    - Use the preprocessing script in this repository to generate a smaller subset (`ucf101_subset`) containing selected videos and a new CSV file with new metadata for each video.

3. **Compress and upload**
    - Compress the `ucf101_subset` folder into a `.zip` file and upload it to Google Drive.

4. **Run the CLIP inference notebook on Colab**
    - It generates the embeddings for each video using CLIP.
    - A new embeddings dataset is saved once all cells have completed.

5. **Download the new embeddings dataset**
    - Save the generated dataset into your local `data` folder for downstream tasks.

## Running the CLIP Inference Notebook

1. **Prepare your dataset**
   - Ensure you have the folder:
     ```
     ucf101_subset/
     |- test.csv
     |_ test/ (videos)
     ```
   - Compress it into:
     ```
     ucf101_subset.zip
     ```

2. **Upload to Google Drive**
   - Place the `.zip` in:
     ```
     My Drive/Dissertation/data/ucf101_subset.zip
     ```

3. **Open the Colab notebook**
   - Launch the notebook directly from the link below or by uploading the [notebook from this repo](./notebooks/Google%20Colab/CLIP_UCF101_runner.ipynb):

     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jzWm_3UOYdw236R9C-28jEMhCFJ32ZTR)

   - This notebook loads the dataset, runs CLIP inference on every frame, and saves the output embeddings.

4. **Enable GPU acceleration**
   - In Colab, change runtime type to a GPU accelerated one.
   - Tested with **T4 GPU**.

5. **Adjust file paths (optional)**
   - You may edit the constants at the top of the notebook:
     - `DRIVE_ZIP_PATH` - path to your uploaded `.zip` dataset.
     - `GDRIVE_OUTPUT_DIR` - where to save the generated embeddings.

6. **Run all cells**
   - The notebook will:
     - Unzip the dataset
     - Load the CLIP model
     - Extract frame embeddings
     - Save `.pt` embedding files to your Drive

7. **Download your new dataset**
   - Once complete, download:
     ```
     My Drive/Dissertation/data/ucf101_embeddings/
     ```
   - Place this folder into your local project:
     ```
     data/ucf101_embeddings/
     ```

## Notes

- Approximate runtime: **~1 hour** for 1723 videos.
