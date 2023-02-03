import sys
import torch
import torch.nn as nn
import torchaudio
from pdb import set_trace as bp
import os

from speechbrain.pretrained import EncoderClassifier

class ECAPA_speechbrain_voxlingua(nn.Module):
        def __init__(self, gpu=0, nograd=False, **kwargs):
            super(ECAPA_speechbrain_voxlingua, self).__init__()
            savedir = "pretrained_models/lang-id-voxlingua107-ecapa"
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            self.language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=savedir)
            # self.language_id.cuda()
            if 'use_gpu' in kwargs:
                if kwargs['use_gpu']:
                    self.language_id.device = torch.device("cuda:{}".format(gpu))
            self.nograd = nograd
            if self.nograd:
                for param in self.language_id.parameters():
                    param.requires_grad = False
            
        def forward(self, x):
            if self.nograd:
                self.language_id.eval()
                with torch.no_grad():
                    emb = self.language_id.encode_batch(x)
            else:
                emb = self.language_id.encode_batch(x)
            emb = emb.squeeze(1)
            return emb
        
        def train(self, mode=True):
            super().train(mode)
            self.language_id.eval()


def MainModel(**kwargs):
    model = ECAPA_speechbrain_voxlingua(nograd=True, **kwargs)
    return model