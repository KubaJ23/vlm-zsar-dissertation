# VLM-ZSAR-Dissertation

Repository for the dissertation project: "An End-to-End Analysis of a Training-Free Pipeline for Zero-Shot Video Action Recognition".

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

## Dataset

This project uses the [UCF101 Action Recognition Dataset](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition).

Dataset paper citation:

@article{soomro2012ucf101,
  title={Ucf101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}

Dataset download link:

@misc{jansen2022ucf101kaggle,
  author={Matthew Jansen},
  title={UCF101 - Action Recognition},
  howpublished={\url{https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition}},
  year={2022},
  note={Accessed: 2025-11-05}
}

This repository already contains the generated embeddings dataset that are required for the action recognition. If you want to recreate the embeddings dataset yourself, follow the instructions [here](colab_instructions.md).
