#!/bin/bash

set -e  # Exit on error.

PYTHON=python  # Python to use; defaults to system Python.

################################################################################
# Configuration
################################################################################

stage=0
sad_train_stage=0
sad_decode_stage=0
diarization_stage=0
nj=4

exp_dir=exps/exp2
seg_dur=0.4
seg_shift=0.2
embs_path=$exp_dir/ecapa_tdnn_voxlingua_speechbrain_language_embeddings
subsegs_basename=subsegments_${seg_dur}seg_${seg_shift}shift
output_rttm_path=$exp_dir/rttm_outputs
clustering_mode=AHC



################################################################################
# Paths to DISPLACE 2023 releases
################################################################################

DISPLACE_DEV_AUDIO_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Release/
DISPLACE_DEV_LABELS_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Label_Release
DISPLACE_DEV_RTTM_DIR=$DISPLACE_DEV_LABELS_DIR/RTTM/Track-2\:LD

DISPLACE_EVAL_AUDIO_DIR=/home/shreyasr/zenodo_downloads/DISPLACE_2023_Dev-Part1_Release/

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

################################################################################
# Prepare data directories, SAD decoding
################################################################################

if [ $stage -le 0 ]; then
  echo "$0: Preparing data directories..."

  # dev
  if [ -d "$exp_dir" ]; then 
    mkdir -p $exp_dir
  fi


  if [ -d "$DISPLACE_DEV_AUDIO_DIR" ]; then 
  local/make_data_dir.py \
  --rttm-dir $DISPLACE_DEV_RTTM_DIR \
    data/displace_dev \
    $DISPLACE_DEV_AUDIO_DIR
  ./create_utt2spk_spk2utt.sh data/displace_dev $DISPLACE_DEV_AUDIO_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_dev/
  else
    echo "${DISPLACE_DEV_AUDIO_DIR} does not exist"
    exit 1
  fi
  # eval
  if [ -d "$DISPLACE_EVAL_AUDIO_DIR" ]; then 
  local/make_data_dir.py \
    data/displace_eval \
    $DISPLACE_EVAL_AUDIO_DIR/data/wav 
   ./create_utt2spk_spk2utt.sh data/displace_eval $DISPLACE_EVAL_AUDIO_DIR
  ./utils/validate_data_dir.sh \
    --no-text --no-feats data/displace_eval/
  else
    echo "${DISPLACE_EVAL_AUDIO_DIR} does not exist"
  fi
fi


#####################################
# SAD decoding.
#####################################
if [ $stage -le 1 ]; then
  echo "$0: Applying SAD model to DEV/EVAL..."
  for dset in dev; do
    echo local/segmentation/detect_speech_activity.sh \
      --nj $nj --stage $sad_decode_stage \
      data/displace_${dset} exp/dihard3_sad_tdnn_stats \
      mfcc exp/displace_sad_tdnn_stats_decode_${dset} \
      data/displace_${dset}_seg

    local/segmentation/detect_speech_activity.sh \
      --nj $nj --stage $sad_decode_stage \
      data/displace_${dset} exp/dihard3_sad_tdnn_stats \
      mfcc exp/displace_sad_tdnn_stats_decode_${dset} \
      data/displace_${dset}_seg
    done
fi

################################################################################
# Get subsegments
################################################################################

if [ $stage -le 2 ]; then
    if [ ! -d $exp_dir/subsegments ]; then
        mkdir -p $exp_dir/subsegments
    fi

    $PYTHON ./get_subsegments.py --max-segment-duration $seg_dur \
        --overlap-duration `echo $seg_dur-$seg_shift|bc` \
        --max-remaining-duration $seg_dur \
        --constant-duration true \
        --vad_segments_file data/displace_dev_seg/segments \
        --out_subsegments_file $exp_dir/subsegments/$subsegs_basename.txt

    mkdir -p $exp_dir/subsegments/$subsegs_basename

    for uttid in `awk '{print $1}' data/displace_dev_seg/wav.scp`;
        do 
            grep --color=never $uttid $exp_dir/subsegments/$subsegs_basename.txt > $exp_dir/subsegments/$subsegs_basename/$uttid.txt
        done



fi

################################################################################
# Extract Embeddings from Speechbrain ECAPA_TDNN language recognition model
# trained on voxlingua107 dataset
################################################################################

if [ $stage -le 3 ]; then

    $PYTHON ./compute_and_save_language_embeddings.py --input_wav_scp data/displace_dev_seg/wav.scp \
        --segments $exp_dir/subsegments/$subsegs_basename.txt \
        --out_path $embs_path \
        --batch_size 100 --use_gpu 1 --model ECAPA_speechbrain_voxlingua_pretrained 

fi

################################################################################
# Perform AHC to cluster languages
################################################################################



if [ $stage -le 4 ]; then

    if [ ! -f $exp_dir/embeddings_$subsegs_basename.tsv ]; then

        find $exp_dir/subsegments/$subsegs_basename/ -type f -name "*.txt" |sort -u > tmp_segs.txt
        find $embs_path/embeddings/$subsegs_basename/  -type f -name "*.npy" |sort -u > tmp_embs.txt
        if [ "$(wc -l < tmp_segs.txt)" -eq "$(wc -l < tmp_embs.txt)" ]; then 
            echo "Creating embeddings_segments file..."
            paste tmp_embs.txt tmp_segs.txt > $exp_dir/embeddings_$subsegs_basename.tsv
            rm -f tmp_embs.txt tmp_segs.txt
            echo "Done"
        else
            echo "Number of segment files is not matching with number of embedding files."
            exit 1;
        fi
    fi

    echo "Running clustering algorithm..."

    $PYTHON ./clustering.py $exp_dir/embeddings_$subsegs_basename.tsv $output_rttm_path $clustering_mode

    echo "Done"
    
fi

###############################################################################
# Evaluate Results
###############################################################################

if [ $stage -le 5 ]; then

    output_rttm_files=$output_rttm_path/$subsegs_basename/$clustering_mode/*.rttm
    ground_truth_rttm_files=$DISPLACE_DEV_RTTM_DIR/*.rttm

    echo "Evaluating output in $output_rttm_path/$subsegs_basename/$clustering_mode/..."
    $PYTHON ../tools/dscore/score.py --collar 0.0 --step 0.01 \
        -r $ground_truth_rttm_files \
        -s $output_rttm_files > $output_rttm_path/$subsegs_basename/$clustering_mode/results.txt #2>/dev/null

    echo "Results written to $output_rttm_path/$subsegs_basename/$clustering_mode/results.txt"
    echo ""
    cat $output_rttm_path/$subsegs_basename/$clustering_mode/results.txt
    
fi

echo ""
echo "Finished."