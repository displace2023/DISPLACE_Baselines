#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.


################################################################################
# Configuration
################################################################################
nj=4
decode_nj=4
stage=0
sad_train_stage=0
sad_decode_stage=0
diarization_stage=0
vb_hmm_stage=0
# If following is "true", then SAD output will be evaluated against reference
# following decoding stage. This step requires the following Python packages be
# installed:
#
# - pyannote.core
# - pyannote.metrics
# - pandas
eval_sad=false



################################################################################
# Paths to DISPLACE 2023 releases
################################################################################
DISPLACE_DEV_DIR=/data1/kaustubhk/Displace_DEV_1
DISPLACE_EVAL_DIR=/data1/kaustubhk/Displace_EVAL_1

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh



################################################################################
# Prepare data directories
################################################################################
if [ $stage -le 0 ]; then
  echo "$0: Preparing data directories..."

  # dev
  if [ -d "$DISPLACE_DEV_DIR" ]; then 
  local/make_data_dir.py \
  --rttm-dir $DISPLACE_DEV_DIR/data/rttm \
    data/displace_dev_fbank \
    $DISPLACE_DEV_DIR/data/wav
  ./create_utt2spk_spk2utt.sh data/displace_dev_fbank $DISPLACE_DEV_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_dev_fbank/
  else
    echo "${DISPLACE_DEV_DIR} does not exist"
    exit 1
  fi
  # eval
  if [ -d "$DISPLACE_EVAL_DIR" ]; then 
  local/make_data_dir.py \
    --rttm-dir $DISPLACE_EVAL_DIR/data/rttm \
    data/displace_eval_fbank \
    $DISPLACE_EVAL_DIR/data/wav 
   ./create_utt2spk_spk2utt.sh data/displace_eval_fbank $DISPLACE_EVAL_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_eval_fbank/
  else
    echo "${DISPLACE_EVAL_DIR} does not exist"
  fi
fi


#####################################
# SAD decoding.
#####################################
if [ $stage -le 1 ]; then
  echo "$0: Applying SAD model to DEV/EVAL..."
  for dset in dev_fbank; do
    local/segmentation/detect_speech_activity.sh \
      --nj $nj --stage $sad_decode_stage \
      data/displace_${dset} exp/dihard3_sad_tdnn_stats \
      mfcc exp/displace_sad_tdnn_stats_decode_${dset} \
      data/displace_${dset}_seg
    done
fi


################################################################################
# Perform first-pass diarization using AHC.
################################################################################
period=0.25
if [ $stage -le 2 ]; then
  
  echo "$0: Performing first-pass diarization of DEV..."
  local/diarize_fbank.sh \
    --nj $nj --stage $diarization_stage \
    --tune true --period $period \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model \
    data/displace_dev_fbank_seg/ exp/displace_diarization_nnet_1a_dev_fbank
fi


if [ $stage -le -3 ]; then
  echo "$0: Performing first-pass diarization of EVAL using threshold "
  echo "$0: obtained by tuning on DEV..."
  thresh=$(cat exp/displace_diarization_nnet_1a_dev_fbank/tuning/thresh_best)
  local/diarize_fbank.sh \
    --nj $nj --stage $diarization_stage \
    --thresh $thresh --tune false --period $period \
    exp/xvector_nnet_1a_tdnn_fbank/ exp/xvector_nnet_1a_tdnn_fbank/plda_model \
    data/displace_eval_fbank_seg/ exp/displace_diarization_nnet_1a_eval_fbank
fi



################################################################################
# Evaluate first-pass diarization.
################################################################################
if [ $stage -le 4 ]; then
  echo "$0: Scoring first-pass diarization on DEV..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/displace_diarization_nnet_1a_dev_fbank/scoring \
    $DISPLACE_DEV_DIR exp/displace_diarization_nnet_1a_dev_fbank/per_file_rttm
fi


if [ $stage -le -5 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm ]; then
  echo "$0: Scoring first-pass diarization on EVAL..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/displace_diarization_nnet_1a_eval_fbank/scoring \
    $DISPLACE_EVAL_DIR exp/displace_diarization_nnet_1a_eval_fbank/per_file_rttm
fi
################################################################################
# Refined first-pass diarization using VB-HMM resegmentation
################################################################################
dubm_model=exp/xvec_init_gauss_1024_ivec_400/model/diag_ubm.pkl
ie_model=exp/xvec_init_gauss_1024_ivec_400/model/ie.pkl

if [ $stage -le 6 ]; then
  echo "$0: Performing VB-HMM resegmentation of DEV..."
  statScale=10
  loop=0.45
  maxiters=1
  echo "statScale=$statScale loop=$loop maxiters=$maxiters" 
  local/resegment_vbhmm.sh \
      --nj $nj --stage $vb_hmm_stage --statscale $statScale --loop $loop --max-iters $maxiters \
      data/displace_dev_fbank exp/displace_diarization_nnet_1a_dev_fbank/rttm \
      $dubm_model $ie_model exp/displace_diarization_nnet_1a_vbhmm_dev/
fi


if [ $stage -le -7 ]; then
  echo "$0: Performing VB-HMM resegmentation of EVAL..."
  local/resegment_vbhmm.sh \
      --nj $nj --stage $vb_hmm_stage \
      data/displace_eval_fbank exp/displace_diarization_nnet_1a_eval_fbank/rttm \
      $dubm_model $ie_model exp/displace_diarization_nnet_1a_vbhmm_eval/
fi



################################################################################
# Evaluate VB-HMM resegmentation.
################################################################################
if [ $stage -le 8 ]; then
  echo "$0: Scoring VB-HMM resegmentation on DEV..."
  local/diarization/score_diarization.sh \
    --scores-dir exp/displace_diarization_nnet_1a_vbhmm_dev/scoring \
    $DISPLACE_DEV_DIR exp/displace_diarization_nnet_1a_vbhmm_dev/per_file_rttm
fi


if [ $stage -le -9 ] && [ -d $DISPLACE_EVAL_DIR/data/rttm ]; then
  if [ -d $DISPLACE_EVAL_DIR/data/rttm/ ]; then
    echo "$0: Scoring VB-HMM resegmentation on EVAL..."
    local/diarization/score_diarization.sh \
      --scores-dir exp/displace_diarization_nnet_1a_vbhmm_eval/scoring \
      $DISPLACE_EVAL_DIR exp/displace_diarization_nnet_1a_vbhmm_eval/per_file_rttm
  fi
fi
