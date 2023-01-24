import pandas as pd
import argparse
from pathlib import Path

def get_speaker_id(rttm_file):
    df = pd.read_csv(rttm_file, sep=' ', names=['type','Meeting_ID_File','Channel_ID' ,'start', 'dur', 'ortho','stype','speaker',"conf",'slat'])
    print(df)
    speaker_ids = df['speaker'].unique()
    return speaker_ids

#speaker_ids = get_speaker_id('file.rttm')
#print(speaker_ids)


parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()



target_dir = Path(args.path)

if not target_dir.exists():
    print("The target directory doesn't exist")
    raise SystemExit(1)

get_speaker_id(target_dir)
