# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:20:32 2021

@author: James
"""
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from datetime import datetime
from model import Transformer


path='./log/' + datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(path)
transformer=Transformer(512,2048,8,10)
writer.add_graph(transformer)