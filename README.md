<div align="left"><img src="displace_img.png" width="550"/></div>

# About the Challenge
The DISPLACE challenge aims to address research issues related to speaker and language diarization in an inclusive manner. The goal of the challenge is to establish new benchmarks for speaker diarization (SD) in multilingual settings and language diarization (LD) in multi-speaker settings, using the same underlying dataset. 

In this challenge, we provide a conversational dataset for speaker and language diarization task. The unique attritibutes of this dataset are that it consists of multiple speakers speaking in a code mixed meeting environment which makes the mentioned tasks of speaker and langauge diarization challenging. The further details about the challenge can be found at [DISPLACE 2023](https://displace2023.github.io/). 

# Updates
[24/01/2023]: We have released the code to compute the Baseline for speaker diarization. 

[25/01/2023]: We have made updates to the run.sh and scoring scripts please update your files. 

The results are computed on development set part 1 and will be updated with more data. 

# Baseline for speaker diarization 
The implementation of the speaker diarization baseline is largely similar to the  [Third DIHARD Speech Diarization Challenge (DIHARD III)](https://dihardchallenge.github.io/dihard3/). 
This baseline has been decribed in DIHARD III baseline paper :
- Ryant, Neville et. al. (2021) “The Third DIHARD Diarization Challenge,” in Proc. INTERSPEECH, 2021. ([paper](https://www.isca-speech.org/archive/interspeech_2021/ryant21_interspeech.html))

The steps involve speech activity detection, front-end feature extraction, x-vector extraction, PLDA scoring followed by Agglomerative Hierarchical Clustering (AHC). The resegmentation is applied to refine speaker assignment using [VB-HMM](https://www.fit.vutbr.cz/research/groups/speech/publi/2018/diez_odyssey2018_63.pdf). 

X-vector extractor is a 13-layer TDNN model which follows the BIG DNN architecture from Zeinali et. al., 2019 ([paper](https://arxiv.org/pdf/1910.12592.pdf)).

The x-vector extractor, PLDA model, UBM-GMM, and total variability matrix were trained on [VoxCeleb I and II](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/). Prior to PLDA scoring and clustering, the x-vectors are centered and whitened using statistics estimated from the DISPLACE development set part 1.

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


# Installation
  
## Step 1: Clone the repo and create a new virtual environment

Clone the repo:

```bash
git clone https://github.com/displace2023/Baseline.git
cd Baseline
```

While not required, we recommend running the recipes from a fresh virtual environment. If using ``virtualenv``:

```bash
virtualenv venv
source venv/bin/activate
```

Alternately, you could use ``conda`` or ``pipenv``. Make sure to activate the environment before proceeding.



## Step 2: Installing Python dependencies

Run the following command to install the required Python packages:

```bash
pip install -r requirements/core.txt
```

```bash
pip install -r requirements/sad.txt
```


## Step 3: Installing remaining dependencies

We also need to install [Kaldi](https://github.com/kaldi-asr/kaldi) and [dscore](https://github.com/nryant/dscore). To do so, run the the installation scripts in the ``tools/`` directory:

```bash
cd tools
./install_kaldi.sh
./install_dscore.sh
cd ..
```

Please check the output of these scripts to ensure that installation has succeeded. If succesful, you should see ``Successfully installed {Kaldi,dscore}.`` printed at the end. If installation of a component fails, please consult the output of the relevant installation script for additional details. If you already have the packages installed creating a softlink to the packages also works.


## Step 4: Running the baselines

Navigate to the ```speaker_diarization``` or ```language_diarization``` directories and follow the instructions in ```README.md``` to run the respective baseline systems.
  
<!-- ## Pretrained SAD model

We have placed a copy of the TDNN+stats SAD model used to produce these results on [Zenodo](https://zenodo.org/). To use this model, download and unarchive the [tarball](https://zenodo.org/record/4299009), then move it to ``speaker_diarization/exp``. -->
