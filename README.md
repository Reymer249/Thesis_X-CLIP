### Overview



This repository contains the code for the thesis project titled "Beyond Coarse-Grained Matching in Video-Text Retrieval."
The full details of the work will be available in the `thesis.pdf` file, which will be uploaded upon approval.

### What is included?

This project builds upon the [X-CLIP](https://github.com/xuguohai/X-CLIP) model and utilizes the [VATEX](https://eric-xw.github.io/vatex-website/about.html) dataset for experimental evaluation.

Due to legal restrictions, we are unable to share the dataset used. This repository includes only the code used for training and evaluation.
Most of the code is adapted from the original [X-CLIP](https://github.com/xuguohai/X-CLIP) repository.

### What is excluded?

1. Dataset, including:
    - video clips 
    - `vatex_[train/val/test]_avail.txt` - txt files with ids of the videos (one id per line) we 
   consider for training
    - `captions.json` (also called in our code captions_avail_formatted.json) - json file with captions for videos of
   the format:

```
{
  "video_id_1": [
    "description string 1",
    "description string 2",
    "description string 3",
    ...
  ],
  "video_id_2": [
    "description string 1",
    "description string 2",
    ...
  ],
  ...
}
```

2. Hard negatives/positives json files of the format:

```
{
  "caption_id_1": [
    [
      "hard positive/negative",
      "POS tag"
    ],
    [
      "hard positive/negative",
      "POS tag"
    ],
    ...
  ],
  "caption_id_2": [
    [
      "hard positive/negative",
      "POS tag"
    ],
    ...
  ],
  ...
}
```

Where POS tag specifies the POS tag of the word which was changed. We don't use it anywhere in the code, so it is safe
to put anything into in. We added it just in case we need it in the future

You may find hard negatives for validation set in the [Chen and Hazel](https://github.com/JewelChen2019/Fine-grained-negatives/tree/main/X-CLIP_fine_grained_vp) repository, as our work is a continuation of
their research ([paper](https://arxiv.org/abs/2410.12407)).

### How to run?

Use shell scripts in the `scripts` folder. The main one is `run_xclip.sh`. More details in the `README.md` file in the
folder. Of course, you will need to get all the files from "What is excluded?" section.

### Structure

*This repository is the fork of [X-CLIP](https://github.com/xuguohai/X-CLIP), so most of the code was not written by us

###### folders
- dataloader - folder with the code to load data
- helpers_scripts - **Python scripts** for some additional processing we needed (i.e. calculate the length of sentences in
the dataset, or generate sets with hard negatives and positives). More info in the `README.md` file in the  folder itself
- modules - we did not modify this folder from the [X-CLIP](https://github.com/xuguohai/X-CLIP)
- plots - Python scripts to generate plots. In the code you will find the result values for all the experiments.
- preprocess -  we did not modify this folder from the [X-CLIP](https://github.com/xuguohai/X-CLIP)
- scripts - **Shell scripts** to run everything. More details in the `README.md` file in the  folder itself
###### files
- `calculate_PosRank.py` - file to calculate PosRank. To use it, you need the weights for the model, and visual features
together with visual features mask (both are created via `get_features.py`)
- `ex_sentences.txt` - txt file with the 25 sentences analyzed mentioned in the "3.4 Hard positives/negatives sampling" section, "Quality" paragraph
- `get_features.py` - the script to process videos into their encoded features. Why would we do that? It makes testing
more fast and robust
- `get_xclip.py` - I don't even know what is that, to be honest. We continued the work of
[Chen and Hazel](https://github.com/JewelChen2019/Fine-grained-negatives/tree/main/X-CLIP_fine_grained_vp), and this
file was in there, that is why it is here. It is not used in the project, as far as I traced, but I am scared to delete
it :)
- `main_clip4clip.py` - as X-CLIP is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), we have this file as
a parent class. Was not changed since [X-CLIP](https://github.com/xuguohai/X-CLIP)
- `main_xclip.py` - the main script to run X-CLIP **without** hard negatives/positives augmentation
- `main_xclip_aug.py` - the main script to run X-CLIP **with** hard negatives/positives augmentation

The two last files have minimal differences, but they use different files from modules folder. Also, `main_xclip.py` should
not have had the arguments like --do_neg_aug (the ones under "AUGMENTED PART" comment in the `main_xclip_aug.py`), but 
when we had to rerun the X-CLIP with no augmentations we already have rewritten the dataloaders to expect these
arguments. That is why the first file has them, but they should be turned off ( and they will not affect the training
if so). You can also run the plain X-CLIP from `main_xclip_aug.py` by just not setting --do_neg_aug and --do_pos_aug flags.

- `metrics.py` - was not changed since [X-CLIP](https://github.com/xuguohai/X-CLIP)
- `util.py` - was not changed since [X-CLIP](https://github.com/xuguohai/X-CLIP)




