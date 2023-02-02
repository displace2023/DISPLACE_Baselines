#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import soundfile as sf
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from kaldi_io import read_mat_scp, write_vec_flt
from subprocess import Popen, PIPE
from librosa.core import resample
from tqdm import tqdm
import sys
from struct import unpack
import argparse
import os

from pdb import set_trace as bp
from models.ECAPA_speechbrain_voxlingua_pretrained import MainModel
    
# torch.backends.cudnn.benchmark = True

#################################################################################################################################
#%% Some utilities

def printf(format, *args):
    sys.stdout.write(format % args)
    
def buffer_reshape(x, hop_len, win_len):
    x = x.reshape(-1)
    y = []
    for i in range(1+len(x)//hop_len):
        if i*hop_len+win_len <= len(x):
            y.append(x[i*hop_len:i*hop_len+win_len])
    # typ = type(x)
    y = torch.stack(y, dim=0)
    return y


def wav_scp_read(cmd):
    cmd = cmd.rstrip("|") + " 2>/dev/null"
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.stdout.read()
    nchannels = np.frombuffer(out, dtype='H', count=1, offset=22)[0]
    sample_rate = np.frombuffer(out, dtype='uint32', count=1, offset=24)[0]
    #can assert if sample_rate == resample_rate  at this point
    audio = np.frombuffer(out, dtype=np.int16, count=-1, offset=44).astype('f8')
    audio /= 2**15  # assuming 16-bit
    return audio, sample_rate


class audio_scp_dataset(Dataset):
    def __init__(self, utt_id, audio_specifier, segments, sample_rate=16000):
        self.sample_rate = sample_rate
        segs = np.genfromtxt(segments,dtype=str)
        self.seg_ids = segs[:,0]
        self.utt_ids = segs[:,1]
        self.strt = (sample_rate*segs[:,2].astype(float)).astype(int)
        self.endd = (sample_rate*segs[:,3].astype(float)).astype(int)
        inds = self.utt_ids == utt_id
        self.seg_ids = self.seg_ids[inds]
        self.utt_ids = self.utt_ids[inds]
        self.strt = self.strt[inds]
        self.endd = self.endd[inds]
        self.max_samples = max(self.endd - self.strt)
        self.audio, fs = wav_scp_read(audio_specifier)
            
    def __getitem__(self, index):
        seg, utt, strt, endd = self.seg_ids[index], self.utt_ids[index], self.strt[index], self.endd[index]
        # bp()
        audio = self.audio[strt:endd]
        # to_pad = self.max_samples - (endd - strt)
        while len(audio) < self.max_samples:
            audio = np.concatenate((audio,audio), axis=0)
        return torch.FloatTensor(audio[:self.max_samples]), seg
    
    def __len__(self):
        return len(self.seg_ids)

#################################################################################################################################
#%% Code to extract the embeddings from pretrained model
parser = argparse.ArgumentParser(description='Embedding extractor')

parser.add_argument('--batch_size',     type=int,   default=100,     help='Batch size for embedding extraction');
parser.add_argument('--gpu',            type=int,   default=0,      help='GPU ID')
parser.add_argument('--use_gpu',        type=int,   default=0,      help='Use GPU? (pass 1 to use gpu)')


# %% Required args
parser.add_argument('--model',          type=str,   default="ECAPA_speechbrain_voxceleb_pretrained",     help='Name of model');
parser.add_argument('--input_wav_scp', help='wav.scp')
parser.add_argument('--segments', help='segments')
parser.add_argument('--out_path', help='output file path')

args = parser.parse_args()

if args.use_gpu:
    torch.cuda.set_device('cuda:{}'.format(args.gpu))



s = MainModel()
if args.use_gpu:
    s.cuda()
s.eval()

def SequenceGenerator(audio_scp_file_name, segments, out_path, batch_size, use_gpu):

    embs_path = os.path.join(out_path, 'embeddings')
    
    with open(audio_scp_file_name, 'r') as f:
        lines = f.readlines()
    
    audio_scp = {}
    for line in lines:
        uttid, cmd = line.rstrip('\n').split(' ', 1)
        audio_scp[uttid] = cmd

    
    for audioid, audiosp in audio_scp.items():
        audio_dataset = audio_scp_dataset(audioid, audiosp, segments=segments)
        audio_loader = DataLoader(audio_dataset, batch_size=batch_size, drop_last=False)


        with torch.no_grad():
            emb = []
            for audbatch, segids in tqdm(audio_loader):
                # bp()
                if use_gpu:
                    audbatch = audbatch.cuda()
                em = s(audbatch).detach()
                emb.append(em)
            emb = torch.cat(emb)

        emb = emb.cpu().numpy()
        
        emb_file = "{}/{}/{}.npy".format(embs_path,os.path.splitext(os.path.basename(segments))[0],audioid)
        
        if not os.path.isdir(os.path.dirname(emb_file)):
            os.makedirs(os.path.dirname(emb_file))

        np.save(emb_file, emb)

      
        
    

SequenceGenerator(args.input_wav_scp, args.segments, args.out_path, args.batch_size, args.use_gpu)
