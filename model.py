# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:41:53 2025

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
        self.dropout = dropout
        
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
        
class LayerNorm(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        
class FeedForwardBlock(nn.Module):
    
    def __init__(self, dModel: int, dFf: int, dropout: float):
        super().__init__()
        self.lin1 = nn.Linear(dModel, dFf)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dFf, dModel)
        self.actfunc = nn.ReLU(inplace=True)
        
    def forward(self,x):
        #[b,seqlen,dmodel]
        x = self.lin1(x)    
        x = self.dropout(self.actfunc(x))
        x = self.lin2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, dModel: int, numheads: int, dropout: float):
        super().__init__()
        assert dModel % numheads == 0, 'dModel is not divisible by numheads'
        
        self.dModel = dModel
        self.numheads = numheads
        #dk is dmodel / numheads aka h
        self.dk = dModel//numheads
        
        self.wq = nn.Linear(dModel, dModel) #query mux
        self.wk = nn.Linear(dModel, dModel) #key mux
        self.wv = nn.Linear(dModel, dModel) #value mux
        
        self.wo = nn.Linear(dModel, dModel) #wo
        self.dropout = nn.Dropout(dropout)
    
    def attentionCalc(self, query, key, value, mask):
        dk = query.shape[-1]
        
        #swap the last two dimensions
        attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        #the output of [b, numheads, seqlen, dk] @ [b, numheads, dk, seqlen]
        #is [b,numheads,seqlen,seqlen]
        
        #apply mask if available
        if mask is not None:
            attentionScores.masked_fill_(mask == 0, -1e9) #mask with -1e9 so softmax will put as zero
        
        attentionScores = attentionScores.softmax(dim = -1)
        if self.dropout is not None:
            attentionScores = self.dropout(attentionScores)
            
        #we go back to the og shape [b,numheads,seqlen,dk]
        return (attentionScores @ value), attentionScores
    
    def forward(self,q,k,v,mask):
        #all these dont change shape [b, seqlen, dmodel]
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)
        
        #reshape into [b, seqlen, numheads, dk] and then into [b, numheads, seqlen, dk]
        query = query.view(query.shape[0], query.shape[1], self.numheads, self.dk)
        query = query.transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.numheads, self.dk)
        key = key.transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.numheads, self.dk)
        value = value.transpose(1,2)
        
        x, self.attentionScores = self.attentionCalc(query, key, value, mask)
        
        x = x.transpose(1,2) #go to shape [b,seqlen,numheads,dk]
        x = x.view(x.shape[0], -1, self.h * self.dk) # now to [b, seqlen, dmodel] 
        
        x = self.wo(x) #no change in shape
        
        return x
        
class Residual(nn.Module):
    
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout()
        self.norm = LayerNorm()
        
    def forward(self, x, prevLayer):
        return x + self.dropout(self.norm(prevLayer(x))) #his implementation puts norm first
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        