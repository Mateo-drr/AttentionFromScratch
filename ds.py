# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:48:59 2025

@author: Mateo-drr
"""

from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):

    def __init__(self, data, tokenizerSrc, tokenizerTgt, srcLang, tgtLang, seqLen):
        super().__init__()
        self.data = data
        self.tokenizerSrc
        self.tokenizerTgt
        self.srcLang
        self.tgtLang
        self.seqLen
        
        self.bosToken = torch.Tensor([tokenizerSrc.token_to_id(['[BOS]'])], dtype=torch.int64)
        self.eosToken = torch.Tensor([tokenizerSrc.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.padToken = torch.Tensor([tokenizerSrc.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
    #JUST THE LENGTH OF THE DATASET
        return len(self.data)

    def __getitem__(self, idx):    
    #TAKE ONE ITEM FROM THE DATASET
        data = self.data[idx]
        #split both languages
        srcTxt = data['translation'][self.srcLang]
        tgtTxt = data['translation'][self.tgtLang]
        
        encInput = self.tokenizerSrc.encode(srcTxt).ids
        decInput = self.tokenizerTgt.encode(tgtTxt).ids

        encPad = self.seqLen - len(encInput) - 2
        decPad = self.seqLen - len(decInput) - 2

        #check if seq len is enough
        if encPad < 0 or decPad < 0:
            raise ValueError('Not enough tokens: sentece is longer than limit')
            
        #add special tokens
        encInput = torch.cat([self.bosToken, torch.tensor(encInput,dtype=torch.int64),
                              self.eosToken, torch.tensor([self.padToken] * encPad, dtype=torch.int64)])
        
        decInput = torch.cat([self.bosToken, torch.tensor(decInput,dtype=torch.int64),
                              torch.tensor([self.padToken] * decPad, dtype=torch.int64)])

        label = torch.cat([torch.tensor(decInput,dtype=torch.int64),
                           self.eosToken, torch.tensor([self.padToken] * decPad, dtype=torch.int64)])

        #all these 3 should have the seqLen
        
        #the encoder mask only needs to mask pad tokens
        #the decoder mask needs to mask pad and future words
        decMask = (decInput != self.padToken).unsqueeze(0).unsqueeze(0).int()
        decMask = decMask & causalMask(decInput.size(0))

        return {'encInput': encInput, #[seqLen]
                'decInput': decInput, #[seqLen]
                'encMask': (encInput != self.padToken).unsqueeze(0).unsqueeze(0).int(),
                'decMask': decMask,
                'label':label,
                'srcTxt': srcTxt,
                'tgtTxt': tgtTxt
                }
            
def causalMask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0








