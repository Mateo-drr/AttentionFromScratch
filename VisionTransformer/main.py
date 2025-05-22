# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 16:41:59 2025

@author: mateo
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ViT import tinyViT, pxVIT
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import torch
import wandb
import copy
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bitsandbytes.optim import Lion

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix using seaborn heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

# PARAMS
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark = True

configD = {
    'runs':1,
    'lr': 1e-4,
    'ds2use': 1,
    'datasets': ['mnist', 'cifar100'],
    'num_epochs': 12,
    'batch': 1024,
    'dModel': 512,
    'hidSize': 1024, 
    'numheads':8,
    'layers':2,
    'numChunks':16, 
    'device':'cuda',
    'wb': False,
    'project_name': 'SampleMNIST',
    'basePath': Path(__file__).resolve().parent,  # base dir
    'modelDir': Path(__file__).resolve().parent / 'weights',
}
config = SimpleNamespace(**configD)

# Make sure modelDir exists
config.modelDir.mkdir(parents=True, exist_ok=True)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(),
    #transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

if config.datasets[config.ds2use] == 'mnist':
    # Load MNIST datasets
    train_ds = datasets.MNIST(root=config.basePath / 'data', train=True, download=True, transform=transform)
    valid_ds = datasets.MNIST(root=config.basePath / 'data', train=False, download=True, transform=transform)
elif config.datasets[config.ds2use] == 'cifar100':
    #Load cifar100
    train_ds = datasets.CIFAR10(root=config.basePath / 'data', train=True, download=True, transform=transform)
    valid_ds = datasets.CIFAR10(root=config.basePath / 'data', train=False, download=True, transform=transform)
    
train_dl = DataLoader(train_ds, batch_size=config.batch, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=config.batch, pin_memory=True)        
    
    
def runModel(model):
    
    metrics = [0]
    
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    # Print the total number of parameters
    print(f'Total number of parameters: {total_params}')
    
    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(config.device)  # For classification (MNIST = 10 classes)
    # optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    optimizer = Lion(model.parameters(), lr=config.lr)
    
    # Initialize Weights & Biases
    if config.wb:
        wandb.init(project=config.project_name, config=configD)
    
    bestTloss = float('inf')
    bestVloss = float('inf')
    
    scaler = torch.amp.GradScaler(device='cuda')
    
    for epoch in range(config.num_epochs):
        model.train()
        trainLoss = 0
    
        for images, labels in train_dl:#tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            images, labels = images.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
    
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
            
            scaler.scale(loss).backward()  # Backward pass
            scaler.step(optimizer)         # Optimizer step
            scaler.update()                # Update scaler for next iteration
    
            trainLoss += loss.item()
            if config.wb:
                wandb.log({"TLoss": loss.item(), 'Learning Rate': optimizer.param_groups[0]['lr']})
    
    
        avg_loss = trainLoss / len(train_dl)
        print(f'Epoch {epoch+1}, Training Loss: {avg_loss}')
    
        model.eval()
        validLoss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in valid_dl:#tqdm(valid_dl, desc=f"Validation {epoch+1}/{config.num_epochs}"):
                images, labels = images.to(config.device), labels.to(config.device)
        
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
        
                validLoss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics Calculation
        avg_lossV = validLoss / len(valid_dl)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_lossV}')
        
        conf_mat = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Validation Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        if config.wb:
            wandb.log({
                "Validation Loss": avg_lossV,
                "Training Loss": avg_loss,
                "Validation Accuracy": acc,
                "Validation Precision": prec,
                "Validation Recall": recall,
                "Validation F1": f1
            })
    
        if avg_loss <= bestTloss and avg_lossV <= bestVloss:
            bestModel = copy.deepcopy(model)
            bestTloss = avg_loss
            bestVloss = avg_lossV
            # torch.save({
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'epoch': epoch,
            #     'losstv': (avg_loss, avg_lossV),
            #     'config': config,
            # }, config.modelDir / 'best.pth')
            
            metrics[0] = (conf_mat,acc,prec,recall,f1)
            
            # plot_confusion_matrix(all_labels, all_preds, class_names=[str(i) for i in range(10)])
    
    
    if config.wb:
        wandb.finish()
        
    return metrics


def summarize_results(results, class_names=[str(i) for i in range(10)], normalize=True):
    accs = [r[0][1] for r in results]
    precs = [r[0][2] for r in results]
    recalls = [r[0][3] for r in results]
    f1s = [r[0][4] for r in results]

    print(f"Average Accuracy: {np.mean(accs):.4f}")
    print(f"Average Precision: {np.mean(precs):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1 Score: {np.mean(f1s):.4f}")

    # Average Confusion Matrix
    conf_mats = [r[0][0] for r in results]
    avg_conf_mat = sum(conf_mats) / len(conf_mats)

    if normalize:
        avg_conf_mat = avg_conf_mat / avg_conf_mat.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_conf_mat, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Average Confusion Matrix')
    plt.tight_layout()
    plt.show()

'''
Run models
'''

#calculate patch variables assuming square images
C,H,W = train_ds[0][0].shape
numClasses = len(train_ds.classes)

stepSize = np.sqrt(config.numChunks)
assert stepSize.is_integer(), 'Invalid number of chunks'
stepSize = int(stepSize)

chunkSize = H/stepSize
assert chunkSize.is_integer(), 'Invalid chunk size'
chunkSize = int(chunkSize)

print('Images shape',C,'x', H,'x',W,)
print('numChunks',config.numChunks,'stepSize',stepSize,'chunkSize',chunkSize)

resA = []
for i in tqdm(range(0,config.runs)):
    model = tinyViT(config.dModel,
                    config.numheads,
                    layers=config.layers,
                    hidSize=config.hidSize,
                    stepSize=stepSize,
                    numChunks=config.numChunks,
                    chunkSize=chunkSize,
                    numChan=C,
                    numClasses=numClasses).to(config.device)
    resA.append(runModel(model))
    

resB = []
for i in tqdm(range(0,config.runs)):
    model = pxVIT(config.dModel,
                  config.numheads,
                  layers=config.layers,
                  hidSize=config.hidSize,
                  stepSize=stepSize,
                  numChunks=config.numChunks,
                  chunkSize=chunkSize,
                  numChan=C,
                  numClasses=numClasses).to(config.device)
    resB.append(runModel(model))

summarize_results(resA)
summarize_results(resB)