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

## Citations

This project adapts parts of the implementations of some ZSAR algorithms. Below are citations to the sources of these implementations.

- Original source: https://github.com/sallymmx/ActionCLIP
```bibtex
@article{wang2021actionclip,
  title={Actionclip: A new paradigm for video action recognition},
  author={Wang, Mengmeng and Xing, Jiazheng and Liu, Yong},
  journal={arXiv preprint arXiv:2109.08472},
  year={2021}
}
```

- The `/data/mpvr_descriptions.json` file contains class descriptions copied from the original MPVR paper.
- Original source: https://github.com/jmiemirza/Meta-Prompting/blob/master/descriptions/gpt/UCF101.json
```bibtex
@inproceedings{mirza2024meta,
  title={Meta-prompting for automating zero-shot visual recognition with llms},
  author={Mirza, M Jehanzeb and Karlinsky, Leonid and Lin, Wei and Doveh, Sivan and Micorek, Jakub and Kozinski, Mateusz and Kuehne, Hilde and Possegger, Horst},
  booktitle={European Conference on Computer Vision},
  pages={370--387},
  year={2024},
  organization={Springer}
}
```
