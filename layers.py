# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 17:15:55 2021

@author: James
"""
import math
import torch
import torch.nn as nn

from utils import LabelSmoothingLoss,subsequent_mask,clone_module_list
from sublayers import FeedForward,MultiHeadAttention,get_positional_encoding


class EmbeddingsWithPositionalEncoding(nn.Module):
    """
    ## Embed tokens and add [fixed positional encoding](positional_encoding.html)
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super(EmbeddingsWithPositionalEncoding,self).__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len))

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].requires_grad_(False)
        return self.linear(x) * math.sqrt(self.d_model) + pe



class EmbeddingsWithLearnedPositionalEncoding(nn.Module):
    """
    <a id="EmbeddingsWithLearnedPositionalEncoding"></a>

    ## Embed tokens and add parameterized positional encodings
    """

    def __init__(self, d_model: int, n_vocab: int, max_len: int = 5000):
        super(EmbeddingsWithLearnedPositionalEncoding,self).__init__()
        self.linear = nn.Embedding(n_vocab, d_model)
        self.d_model = d_model
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]]
        return self.linear(x) * math.sqrt(self.d_model) + pe



class AttentionLayer(nn.Module):
    """

    ## Attention Layer

    This can act as an encoder layer or a decoder layer.

    Some implementations, including the paper seem to have differences
    in where the layer-normalization is done.
    Here we do a layer normalization before attention and feed-forward networks,
    and add the original residual vectors.
    Alternative is to do a layer normalization after adding the residuals.
    But we found this to be less stable when training.
    We found a detailed discussion about this in the paper
     [On Layer Normalization in the Transformer Architecture](https://papers.labml.ai/paper/2002.04745).
    """

    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ReLU(),
                 dropout_prob: float=0.1):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `src_attn` is the source attention module (when this is used in a decoder)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        
        super().__init__()
        self.size = d_model
        self.self_attn = MultiHeadAttention(heads,d_model,dropout_prob,bias)
        self.src_attn = MultiHeadAttention(heads,d_model,dropout_prob,bias)
        self.feed_forward = FeedForward(d_model,d_ff,dropout_prob,activation,
                                         is_gated,bias,bias_gate)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # If a source is provided, get results from attention to source.
        # This is when you have a decoder layer that pays attention to 
        # encoder outputs
        if src is not None:
            # Normalize vectors
            z = self.norm_src_attn(x)
            # Attention to source. i.e. keys and values are from source
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # Add the source attention results
            x = x + self.dropout(attn_src)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Save the input to the feed forward layer if specified
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x


class Encoder(nn.Module):
    """
    ## Transformer Encoder
    """

    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ReLU(),
                 dropout_prob: float=0.1,
                 n_layers: int=6):
        
        super(Encoder,self).__init__()
        
        layer=AttentionLayer(d_model,d_ff,heads,bias,is_gated,bias_gate,
                               activation,dropout_prob)
        # Make copies of the Attention layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each Attention layer
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Decoder(nn.Module):
    """
    ## Transformer Decoder
    """

    def __init__(self,
                 d_model: int,
                 d_ff:int,
                 heads:int,
                 bias:bool=True,
                 is_gated: bool = False,
                 bias_gate:bool=True,
                 activation=nn.ReLU(),
                 dropout_prob: float=0.1,
                 n_layers: int=6):
        
        super(Decoder,self).__init__()
        
        layer=AttentionLayer(d_model,d_ff,heads,bias,is_gated,bias_gate,
                               activation,dropout_prob)
        # Make copies of the Attention layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor):
        # Run through each Attention layer
        for layer in self.layers:
            x = layer(x=x, mask=tgt_mask, src=memory, src_mask=src_mask)
        # Finally, normalize the vectors
        return self.norm(x)


class Generator(nn.Module):
    """
    ## Generator

    This predicts the tokens and gives the lof softmax of those.
    You don't need this if you are using `nn.CrossEntropyLoss`.
    """

    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(d_model, n_vocab)

    def forward(self, x):
        return self.projection(x)


