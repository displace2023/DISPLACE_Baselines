<div align="left"><img src="../displace_img.png" width="550"/></div>

# About the Challenge
The DISPLACE challenge aims to address research issues related to speaker and language diarization in an inclusive manner. The goal of the challenge is to establish new benchmarks for speaker diarization (SD) in multilingual settings and language diarization (LD) in multi-speaker settings, using the same underlying dataset. 

In this challenge, we provide a conversational dataset for speaker and language diarization task. The unique attritibutes of this dataset are that it consists of multiple speakers speaking in a code mixed meeting environment which makes the mentioned tasks of speaker and langauge diarization challenging. The further details about the challenge can be found at [DISPLACE 2023](https://displace2023.github.io/). 

# Baseline for Language Dizarization
The implementation of the language diarization baseline is based on a Agglomerative Hierarchical Clustering over language embeddings extracted from a spoken language recognition model trained on the VoxLingua107 dataset using SpeechBrain. The model was based on the ECAPA-TDNN architecture ([1](https://arxiv.org/abs/2005.07143)). VoxLingua covers 107 different languages . We used this model as an feature (embeddings) extractor . We experimented this model on our own data with a range of different hop lengths and frame sizes. 
The steps involved for language diarization are speech activity detection, utterance-level feature extraction and followed by Agglomerative Hierarchical Clustering (AHC). 
```
@inproceedings{valk2021slt,
  title={{VoxLingua107}: a Dataset for Spoken Language Recognition},
  author={J{\"o}rgen Valk and Tanel Alum{\"a}e},
  booktitle={Proc. IEEE SLT Workshop},
  year={2021},
}

@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}

```
# Prerequisites
Run the following command to install the required Python packages:
```bash
$ pip install -r requirements.txt
```

# Running the baseline recipes
Change the path to
```bash
$ cd language_diarization/
```


## Dataset path
open ``run.sh`` in a text editor. The first section of this script defines paths to the Displace challenge DEV and EVAL releases and should look something like the following:

```bash
################################################################################
# Paths to Displace releases
################################################################################

DISPLACE_DEV_AUDIO_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Release/
DISPLACE_DEV_LABELS_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Label_Release

DISPLACE_EVAL_DATA_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Release/

```
  
Change the variables ``DISPLACE_DEV_AUDIO_DIR``, ``DISPLACE_DEV_LABELS_DIR``, and ``DISPLACE_EVAL_AUDIO_DIR`` so that they point to the roots of the Displace DEV Audio release and label (rttm) releases on your filesystem. Save your changes, exit the editor, and run:

```bash
./run.sh
```



# Expected results
**Table 2: Baseline language diarization results for the Displace development set(part 1) using embeddings extraction from pretrained ECAPA TDNN SPEECHBRAIN model and  followed by AHC clustering.**
# Expected results

Expected DER for the language diarization baseline system on the Displace challenge DEV set are presented in Table 2.


|  Method       | DER (Dev)   |  JER (Dev)  | 
| ------------  | ----------- | ----------- |
| AHC           |   46.56     |   73.55     |    

