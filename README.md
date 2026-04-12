# VLM-ZSAR-Dissertation

Repository for the dissertation project: "An End-to-End Analysis of a Training-Free Pipeline for Zero-Shot Video Action Recognition".

## Setup

Follow these steps to setup and run scripts for this project. Notebooks can be run by selecting the `zsar_project` kernel for each notebook. Scripts are stored in the `scripts` directory and can be executed by following the steps in the sections below. To run and configure experiments, modify and run the script `02_run_tests.py` in the `scripts` directory.

### Download the `data` directory

The code in this repository assumes there exists the `data` directory in the root. This contains the datasets, metadata, and precomputed embeddings used for running the experiments. The data folder is several GBs so it is not included with the project code.

After downloading this repository, replace the empty, placeholder [data folder](./data/) with the actual data folder from [here](temp).

### Install dependencies

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

Run a Script (must have activated environment + currently in root directory)
```bash
python -m scripts.SCRIPT_NAME
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

## File Structure

The project organises code and files into these directories:

- `/data` stores the datasets (videos), video embeddings, and metadata.
- `/notebooks` is for notebooks, which are used to analyse and visualise the results.
- `/scripts` directory contains scripts which setup the datasets, precompute embeddings, and run experiments.
- `/src` directory is for the python modules that contains reusable python code. These modules are imported by the notebooks, scripts, and other modules.

## Running Experiments

To run an experiment with a specific pipeline configuration, follow these steps:
- Define the sampler, aggregator or prompter class which inherit the appropriate `abstract base class` (ABC) in `/src`.
- Add instances of the classes to the lists of samplers, aggregators, and prompters in the script `02_run_tests.py`.
- Select your dataset by creating a `VideoDataset` object.
- Run the script. A pipeline for each combination of those components will run and save the results to the `/results` directory.



## Dataset

This project uses the [UCF101 Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) and ActivityNet dataset.

Dataset paper citations:

@article{soomro2012ucf101,
  title={Ucf101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}

@inproceedings{caba2015activitynet,
  title={Activitynet: A large-scale video benchmark for human activity understanding},
  author={Caba Heilbron, Fabian and Escorcia, Victor and Ghanem, Bernard and Carlos Niebles, Juan},
  booktitle={Proceedings of the ieee conference on computer vision and pattern recognition},
  pages={961--970},
  year={2015}
}

Dataset download link:

@misc{jansen2022ucf101kaggle,
  author={Matthew Jansen},
  title={UCF101 - Action Recognition},
  howpublished={\url{https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition}},
  year={2022},
  note={Accessed: 2025-11-05}
}

ActivityNet was downloaded using the FiftyOne tool as explained [here](http://activity-net.org/download.html).

The precomputed embeddings, datasets, and metadata are stored in the `data` directory. Scripts were used to generate the embeddings for each dataset.

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

- Original source: https://github.com/MCG-NJU/MGSampler/tree/main
```bibtex
@inproceedings{zhi2021mgsampler,
  title={Mgsampler: An explainable sampling strategy for video action recognition},
  author={Zhi, Yuan and Tong, Zhan and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF International conference on Computer Vision},
  pages={1513--1522},
  year={2021}
}
```

- Original source: https://github.com/m-bain/clip-hitchhiker
```bibtex
@article{bain2022clip,
  title={A clip-hitchhiker's guide to long video retrieval},
  author={Bain, Max and Nagrani, Arsha and Varol, G{\"u}l and Zisserman, Andrew},
  journal={arXiv preprint arXiv:2205.08508},
  year={2022}
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
