# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:41:59 2025

@author: mateo
"""

from dataset import CustomDataset
from torch.utils.data import DataLoader
from model import SampleNet
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import torch
import wandb
import copy
from pathlib import Path

#PARAMS
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

configD = {'lr':1e-2,
           'num_epochs': 12,
           'batch':64,
           'wb':False,
           'project_name': 'Sample',
           'basePath': Path(__file__).resolve().parent, #base dir
           'modelDir': Path(__file__).resolve().parent / 'weights',
           }
config = SimpleNamespace(**configD)

data=[]
vdat=[]

train_ds = CustomDataset(data)
valid_ds = CustomDataset(vdat)
train_dl = DataLoader(train_ds, batch_size=config.batch, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=config.batch, pin_memory=True)


# Instantiate the model
model = SampleNet()
# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#init weights & biases
if config.wb:
    wandb.init(project=config.project_name,
               config=configD)


bestTloss=1e9
bestVloss=1e9
for epoch in range(config.num_epochs):    
    
    model.train()
    trainLoss=0
    for sample in tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
        
        lbl,inp = sample
        out = model(sample)
        
        optimizer.zero_grad()    
        loss = criterion(out, lbl)  # Compute loss
        loss.backward()             # Backward pass
        optimizer.step()        

        trainLoss += loss.item()
        
        if config.wb:
            wandb.log({"TLoss": loss,
                       'Learning Rate': optimizer.param_groups[0]['lr']})
            
    avg_loss = trainLoss / len(train_dl)    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    model.eval()
    validLoss=0
    with torch.no_grad():
        for sample in tqdm(valid_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            
            lbl,inp = sample
            out = model(sample)
        
            loss = criterion(out, lbl)  # Compute loss
            
            validLoss += loss.item()
            
        avg_lossV = validLoss / len(valid_dl)    
        print(f'Epoch {epoch+1}, Loss: {avg_lossV}') 


    if config.wb:
        wandb.log({"Validation Loss": avg_lossV, "Training Loss": avg_loss})
    
    if avg_loss <= bestTloss and avg_lossV <= bestVloss:
        bestModel = copy.deepcopy(model)
        bestTloss = avg_loss
        bestVloss = avg_lossV
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'losstv': (avg_loss,avg_lossV),
        'config':config,
        }, config.modelDir / 'best.pth')
    
if config.wb:
    wandb.finish()    










