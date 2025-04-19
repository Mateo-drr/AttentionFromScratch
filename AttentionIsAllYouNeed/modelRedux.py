# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:34:19 2025

@author: Mateo-drr
"""

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, dModel: int, vocabSize: int):
        
        super().__init__()
        self.dModel = dModel #size of each token
        self.vocabSize = vocabSize
        self.embedding = nn.Embedding(vocabSize, dModel)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dModel)

class PositionalEncoding(nn.Module):
    
    def __init__(self, dModel: int, seqLen: int, dropout: float):
        
        super().__init__()
        self.dModel = dModel
        self.seqLen = seqLen #size of sentence
        self.dropout = nn.Dropout(dropout)
        
        posEnc = torch.zeros(seqLen, dModel) 
        
        #the formula is a division of pos/denom
        pos = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
        #this denom formula is the same from the paper just with log for numstab
        denom = torch.exp(torch.arange(0, dModel, 2, dtype=torch.float) * (-math.log(10000.0) / dModel))
        
        #evens -> sin | odd -> cos
        posEnc[:,0::2] = torch.sin(pos*denom)
        posEnc[:,1::2] = torch.cos(pos*denom)
        
        posEnc = posEnc.unsqueeze(0) # [1, seqLen, dModel]
        
        #use this buffer thing to make the model save this
        self.register_buffer('posEnc', posEnc)
        
    def forward(self,x):
        x = x + (self.posEnc[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

    
class TransformerRedux(nn.Module) :
    
    def __init__(self,srcVsize: int, tgtVsize: int, srcSeqLen: int, tgtSeqLen: int,
                 dModel: int=512, layers: int=6, numheads: int=8, dropout: float=0.1,
                 hidSize: int=2048):
        super().__init__()
        
        self.srcEmb = InputEmbeddings(dModel, srcVsize)
        self.tgtEmb = InputEmbeddings(dModel, tgtVsize)
        self.srcPos = PositionalEncoding(dModel, srcSeqLen, dropout)
        self.tgtPos = PositionalEncoding(dModel, tgtSeqLen, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
                                                   d_model=dModel,
                                                   nhead=numheads,
                                                   dim_feedforward=hidSize,
                                                   dropout=dropout,
                                                   batch_first=True
                                                   )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=dModel,
                                                   nhead=numheads,
                                                   dim_feedforward=hidSize,
                                                   dropout=dropout,
                                                   batch_first=True
                                                   )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        
        self.lin = nn.Linear(dModel, tgtVsize)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for param in self.parameters():
            if param.dim() > 1:  # Apply Xavier initialization to weights with >1 dimension
                nn.init.xavier_uniform_(param)
        
    def encode(self,x, srcMask):
        x = self.srcEmb(x)
        x = self.srcPos(x)
        x = self.encoder(src=x, src_key_padding_mask=srcMask)
        return x
        
    def decode(self, x, encOut, srcMask, tgtMaskCau, tgtMaskPad):
        x = self.tgtEmb(x)
        x = self.tgtPos(x)
        x = self.decoder(tgt=x,
                         memory=encOut,
                         
                         #shape has to be either [tgtSeqLen, srcSeqLen]
                         # or [b * numheads, tgtSeqLen, srcSeqLen]
                         tgt_mask=tgtMaskCau, 
                         
                         #Shape has to be [b, seqLen]
                         tgt_key_padding_mask= tgtMaskPad,
                         
                         #Shape has to be [b, seqLen]
                         memory_key_padding_mask=srcMask,
                         tgt_is_causal=True) #idk
        return x
        
    def lastLL(self, x):
        return torch.log_softmax(self.lin(x), dim=-1)











