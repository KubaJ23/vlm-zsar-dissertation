# VLM-ZSAR-Dissertation

Repository for the dissertation project: "An End-to-End Analysis of a Training-Free Pipeline for Zero-Shot Video Action Recognition".

Dataset: [UCF101 Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition)

## Setup

Initialise Conda (Windows):
```bash
conda init cmd.exe
```

Create the environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate zsar_project
```

Update environment to match environment.yml:
```bash
conda env update --name zsar_project --file environment.yml --prune
```

Export environment (optional, to update environment.yml):
```bash
conda env export > environment.yml
```

Deactivate when done:
```bash
conda deactivate
```
