# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:40:49 2025

@author: Mateo-drr
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
import wandb

from ds import CustomDataset, causalMask
from model import initTransformer


def getSentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def makeTokenizer(config, ds, lang):
    tokenizerPath = Path(config.tokenizer_file.format(lang))
    if not Path.exists(tokenizerPath):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[BOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(getSentences(ds,lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(str(tokenizerPath))
        
    return tokenizer

def getDs(config):
    dsRaw = load_dataset('opus_books', f'{config.lang_src}-{config.lang_tgt}', split='train')
    
    #buildTKNZ
    tokenizerSrc = makeTokenizer(config, dsRaw, config.lang_src)
    tokenizerTgt = makeTokenizer(config, dsRaw, config.lang_tgt)
    
    train_ds_size = int(0.9 * len(dsRaw))
    valid_ds_size = len(dsRaw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(dsRaw, [train_ds_size, valid_ds_size])
    
    train_ds = CustomDataset(train_ds_raw, tokenizerSrc, tokenizerTgt,
                             config.lang_src, config.lang_tgt, config.seq_len)   
    valid_ds = CustomDataset(valid_ds_raw, tokenizerSrc, tokenizerTgt,
                             config.lang_src, config.lang_tgt, config.seq_len)   
    
    srcMax, tgtMax = 0,0
    
    for item in dsRaw:
        srcId = tokenizerSrc.encode(item['translation'][config.lang_src]).ids
        tgtId = tokenizerTgt.encode(item['translation'][config.lang_tgt]).ids
        srcMax = max(srcMax,len(srcId))
        tgtMax = max(tgtMax,len(tgtId))
        
    print('max lens', srcMax, tgtMax)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, pin_memory=True, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=config.batch_size, pin_memory=True, shuffle=False)
    
    return train_dl, valid_dl, tokenizerSrc, tokenizerTgt


configD = {
    'batch_size': 16,
    'num_epochs': 10,
    'lr': 1e-4,
    'seq_len': 350,
    'd_model': 512,
    'lang_src': 'en',
    'lang_tgt': 'it',
    'tokenizer_file': 'tokenizer_{0}.json',
    'device': 'cuda',
    'wb':True
    }
config = SimpleNamespace(**configD)

torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

#get the dataset
train_dl, valid_dl, tokenizerSrc, tokenizerTgt = getDs(config)
srcVsize,tgtVsize = tokenizerSrc.get_vocab_size(), tokenizerTgt.get_vocab_size()

# Instantiate the model
model = initTransformer(srcVsize, tgtVsize, config.seq_len, config.seq_len,
                        config.d_model).to(config.device)
# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizerSrc.token_to_id('[PAD]'),
                                label_smoothing=0.1).to(config.device)
optimizer = optim.AdamW(model.parameters(), lr=config.lr)

#initialize wb
if config.wb:
    wandb.init(project="AttentionFromScratch",
               config=configD)

for epoch in range(config.num_epochs):    
    
    model.train()
    trainLoss=0
    for data in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        
        encInput = data['encInput'].to(config.device) #[b, seqLen]
        decInput = data['decInput'].to(config.device) #[b. seqLen]
        encMask = data['encMask'].to(config.device) #[b, 1, 1, seqLen]
        decMask = data['decMask'].to(config.device) #[b, 1, seqLen, seqLen]
        label = data['label'].to(config.device) #[b, seqlen]
        
        #run the model
        encOut = model.encode(encInput, encMask) #[b , seqlen, dmodel]
        decOut = model.decode(decInput, encOut, encMask, decMask) # ''
        out = model.lastLL(decOut) #[b, seqlen, tgtVsize]
        
        #loss and param update
        optimizer.zero_grad()  
        #for some reason he changes shape of out to [b * seqlen, tgtVsize]
        out = out.view(-1, tokenizerTgt.get_vocab_size())
        label = label.view(-1) #and [b * seqlen]
        loss = criterion(out, label)  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()        
        
        if config.wb:
            wandb.log({"TLoss": loss})

        trainLoss += loss.item()
        
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')    
    
    model.eval()
    with torch.no_grad():
        validLoss=0
        for data in tqdm(valid_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            
            encInput = data['encInput'].to(config.device) #[b, seqLen]
            decInput = data['decInput'].to(config.device) #[b. seqLen]
            encMask = data['encMask'].to(config.device) #[b, 1, 1, seqLen]
            decMask = data['decMask'].to(config.device) #[b, 1, seqLen, seqLen]
            label = data['label'].to(config.device) #[b, seqlen]
            
            #run the model
            encOut = model.encode(encInput, encMask) #[b , seqlen, dmodel]
            decOut = model.decode(decInput, encOut, encMask, decMask) # ''
            out = model.lastLL(decOut) #[b, seqlen, tgtVsize]
            
            #for some reason he changes shape of out to [b * seqlen, tgtVsize]
            out = out.view(-1, tokenizerTgt.get_vocab_size())
            label = label.view(-1) #and [b * seqlen]
            loss = criterion(out, label)  # Compute loss 
    
            validLoss += loss.item()
            
        avg_lossV = validLoss / len(valid_dl)    
        print(f'Epoch {epoch+1}, Loss: {avg_lossV}') 
    
    if config.wb:
        wandb.log({"Validation Loss": avg_lossV, "Training Loss": avg_loss})
    
if config.wb:
    wandb.finish()    
