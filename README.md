### Overview

It is the repository for the thesis work called Beyond "Coarse-Grained Matching in Video-Text Retrieval." With all
the details of the work you may familiarize yourself using the thesis.pdf file (will be uploaded once approved).

### What is in here?

In the study we used the [X-CLIP](https://github.com/xuguohai/X-CLIP) model and [VATEX](https://eric-xw.github.io/vatex-website/about.html)
dataset for our experiments. This repository is the code we used for all our experiments. We are not legally allowed
to share the dataset collected, so this repository contains only the code we used to train the models. The most of the
code presented here was taken from [X-CLIP](https://github.com/xuguohai/X-CLIP) repository.

### How to run?

Use shell scripts in the `scripts` folder. The main one is `run_xclip.sh`. More details in the `README.md` file in the
folder

### Structure

*This repository is the fork of [X-CLIP](https://github.com/xuguohai/X-CLIP), so most of the code was not written by us

###### folders
- dataloader - folder with the code to load data. More details in the `README.md` file in the  folder itself
- helpers_scripts - **Python scripts** for some additional processing we needed (i.e. calculate the length of sentences in
the dataset, or generate sets with hard negatives and positives). More info in the `README.md` file in the  folder itself
- modules - we did not modify this folder from the [X-CLIP](https://github.com/xuguohai/X-CLIP)
- plots - Python scripts to generate plots. In the code you will find the result values for all the experiments.
- preprocess -  we did not modify this folder from the [X-CLIP](https://github.com/xuguohai/X-CLIP)
- scripts - **Shell scripts** to run everything. More details in the `README.md` file in the  folder itself
###### files
calculate_PosRank.py - file to calculate PosRank. To use it, you need the weights for the model 


