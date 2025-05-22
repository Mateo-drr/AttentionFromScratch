# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:41:59 2025

@author: mateo
"""

import torch
import torch.nn as nn
import math

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
    
class ViTBlock(nn.Module):
    def __init__(self, dModel, numheads, dropout, hidSize):
        super().__init__()
        
        self.attention = MultiHeadAttentionBlock(dModel, numheads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dModel)
        self.ff = nn.Sequential(nn.Linear(dModel, hidSize),
                                nn.Mish(inplace=True),
                                nn.Linear(hidSize, dModel)
                                )
        self.m = nn.Mish(inplace=True)
        
    def forward(self,x):
        
        x1 = self.attention(q=x, k=x, v=x, mask=None)
        x = x + self.dropout(self.norm(x1))
        x1 = self.ff(x)
        x = x + self.dropout(self.norm(x1))
        
        return x
    
class tinyViT(nn.Module):
    
    def __init__(self, dModel, numheads, hidSize=1024, dropout=0.1, layers=1,
                 numChunks=16, chunkSize=7, stepSize=4, numChan=1, numClasses=10):
        super().__init__()
        
        self.hstep,self.wstep=stepSize,stepSize
        self.numChunks=numChunks
        self.chunkSize=chunkSize
        self.dModel=dModel
        
        self.chunkProj = nn.Linear(numChan*chunkSize*chunkSize, dModel)
        self.posEnc = PositionalEncoding(dModel, self.numChunks+1, dropout)
        
        self.attention = MultiHeadAttentionBlock(dModel, numheads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dModel)
        self.ff = nn.Sequential(nn.Linear(dModel, hidSize),
                                nn.Mish(inplace=True),
                                nn.Linear(hidSize, dModel)
                                )
        self.m = nn.Mish(inplace=True)
        
        self.decoders = nn.ModuleList([
            ViTBlock(dModel, numheads, dropout, hidSize) for _ in range(layers)])
        
        self.classifier = nn.Linear(dModel, numClasses) 
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dModel))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
                # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, x):
        
        B,C,H,W = x.shape
        #chop first the columns and then the rows
        patches = x.unfold(2, self.chunkSize, self.chunkSize) 
        #[b,c, stepsize, w, chunkSize]
        patches = patches.unfold(3, self.chunkSize, self.chunkSize) 
        #[b, c, stepsize, stepsize, chunksize, chunksize]
        
        #now to reshape we need to reorder the tensor
        patches = patches.permute(0,2,3,1,4,5)
        patches = patches.contiguous().view(B, self.numChunks, C * self.chunkSize * self.chunkSize)
        #[b, numchunks, c * chunksize * chunksize]
        
        x = self.chunkProj(patches)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dModel]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, numChunks + 1, dModel]
        
        x = self.posEnc(x)
        
        # x1 = self.attention(q=x, k=x, v=x, mask=None)
        
        # x = x + self.dropout(self.norm(x1))
    
        # x1 = self.ff(x)
        
        # x = x + self.dropout(self.norm(x1))
        
        for decoder in self.decoders:
            x = decoder(x)
        
        # x = self.m(x)
        x = self.classifier(x[:,0,:]) #only cls
    
        return x
    
    
class pxVIT(nn.Module):
    
    def __init__(self, dModel, numheads, hidSize=1024, dropout=0.1, layers=1,
                 numChunks=16, chunkSize=7, stepSize=4, numChan=1, numClasses=10):
        super().__init__()
        
        self.hstep,self.wstep=stepSize,stepSize
        self.numChunks=numChunks
        
        self.dModel=dModel
        
        self.px = nn.PixelUnshuffle(chunkSize)
        
        self.chunkProj = nn.Linear(numChan*chunkSize*chunkSize, dModel)
        self.posEnc = PositionalEncoding(dModel, numChunks+1, dropout)
        
        self.attention = MultiHeadAttentionBlock(dModel, numheads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dModel)
        self.ff = nn.Sequential(nn.Linear(dModel, hidSize),
                                nn.Mish(inplace=True),
                                nn.Linear(hidSize, dModel)
                                )
        self.m = nn.Mish(inplace=True)
        
        self.decoders = nn.ModuleList([
            ViTBlock(dModel, numheads, dropout, hidSize) for _ in range(layers)])
        
        self.classifier = nn.Linear(dModel, numClasses) 
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dModel))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        print('Weights initialized')
        for param in self.parameters():
            if param.dim() > 1:  
                nn.init.xavier_normal_(param)
                # nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, x):
        
        B,C,H,W = x.shape
        
        x = self.px(x)
        
        #go from [b,c*size^2, h/size, w/size] to [b,c*size^2, h/size * w/size] and then 
        #flip channels with the data
        xflat = x.flatten(2).transpose(1,2) 
        
        x = self.chunkProj(xflat)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, dModel]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, numChunks + 1, dModel]
        
        x = self.posEnc(x)
                
        # x1 = self.attention(q=x, k=x, v=x, mask=None)
        
        # x = x + self.dropout(self.norm(x1))
    
        # x1 = self.ff(x)
        
        # x = x + self.dropout(self.norm(x1))
        
        for decoder in self.decoders:
            x = decoder(x)
        
        # x = self.m(x)
        x = self.classifier(x[:,0,:]) #only cls
    
        return x    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    