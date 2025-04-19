# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:41:59 2025

@author: mateo
"""

import torch
import torch.nn as nn
import math

def best_patch_split(H, W, num_patches):
    best_pair = None
    best_ratio_diff = float('inf')
    
    for h in range(1, num_patches + 1):
        if num_patches % h != 0:
            continue
        w = num_patches // h
        
        # Check if this configuration produces integer-sized patches
        patch_H = H / h
        patch_W = W / w
        
        # Skip configurations that don't produce integer dimensions
        if not (patch_H.is_integer() and patch_W.is_integer()):
            continue
        
        # Calculate how far the ratio is from 1:1 (perfect square)
        ratio = patch_W / patch_H
        ratio_diff = abs(ratio - 1)
        
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_pair = (h, w)
    
    assert best_pair is not None, 'image cant be chunked into given number of patches'
    return best_pair

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
        
        x = x.transpose(1,2) #from [b,numheads,seqlen,dk] to shape [b,seqlen,numheads,dk]
        x = x.reshape(x.shape[0], -1, self.numheads * self.dk) # now to [b, seqlen, dmodel] 
        #view has some memory issue here so using reshape
        
        x = self.wo(x) #no change in shape
        
        return x
    
class tinyViT(nn.Module):
    
    def __init__(self, dModel, numheads, dropout=0.1, numChunks=9):
        super().__init__()
        
        self.hstep,self.wstep=None,None
        self.numChunks=numChunks
        
        self.attention = MultiHeadAttentionBlock(dModel, numheads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm()
        self.ff = nn.Linear(dModel)
        
    def forward(self, x):
        
        B,C,H,W = x.shape
        
        #TODO add training check
        if self.hstep is None and self.wstep is None:
            self.hstep,self.wstep = best_patch_split(H, W, self.numChunks)
        
        x = x.unfold(dim=2, )
        
        x1 = self.attention(q=x, k=x, v=x, mask=None)
        
        x = x + self.dropout(self.norm(x1))
    
        x1 = self.ff(x)
        
        x = x + self.dropout(self.norm(x1))
    
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    