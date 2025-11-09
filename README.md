# VLM-ZSAR-dissertation

Repository for dissertation project "An End-to-End Analysis of a Training-Free Pipeline for Zero-Shot Video Action Recognition"

Dataset from: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition

commands:

conda init cmd.exe

conda env create -f environment.yml

conda activate zsar_project

Syncs your environment with the file:

conda env update --name zsar_project --file environment.yml --prune

conda env export > environment.yml

conda deactivate
