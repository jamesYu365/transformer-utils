# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:54:10 2021

@author: James
"""

import torch
import torch.nn as nn
import pandas as pd
from utils import subsequent_mask
from model import Transformer

data_pkl=
train_path=
val_path=
opt={

     '-epoch':10,
     '-b':2048,
     'd_model':512,
     'd_ff':2048,
     'heads':8,
     'n_layers':6,
     'n_warmup_steps':4000,
     'lr_mul':2,
     'seed':10,
     'dropout_prob':0.1,
     'no_cuda':False,
     'label_smoothing':True
     }

opt['cuda'] = not opt['no_cuda']
opt['d_word_vec'] = opt['d_model']


# https://pytorch.org/docs/stable/notes/randomness.html
# For reproducibility
if opt['seed'] is not None:
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])


transformer=Transformer(d_model,d_ff,heads,n_vocab,max_len,bias,is_gated,
                        bias_gate,activation,dropout_prob,n_layers)

