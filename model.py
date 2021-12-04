# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 18:43:44 2021

@author: James
"""
import torch
import torch.nn as nn
from layers import Encoder,Decoder,EmbeddingsWithPositionalEncoding,Generator



class Transformer(nn.Module):
    """
    ## Combined Encoder-Decoder
    """

    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 n_vocab: int,
                 max_len:int=5000,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ReLU(),
                 dropout_prob: float=0.1,
                 n_layers: int=6
                 ):
        
        super().__init__()
        self.encoder = Encoder(d_model,d_ff,heads,bias,is_gated,bias_gate,
                               activation,dropout_prob,n_layers)
        self.decoder = Decoder(d_model,d_ff,heads,bias,is_gated,bias_gate,
                               activation,dropout_prob,n_layers)
        self.src_embed = EmbeddingsWithPositionalEncoding(d_model,n_vocab,max_len)
        self.tgt_embed = EmbeddingsWithPositionalEncoding(d_model,n_vocab,max_len)
        self.generator = Generator(n_vocab,d_model)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor):
        
        # Run the source through encoder
        enc = self.encode(src, src_mask)
        # Run encodings and targets through decoder
        return self.decode(enc, src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

