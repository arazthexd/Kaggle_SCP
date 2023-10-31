import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn as nn

from tqdm import tqdm

def loss_fn(y_pred, y_true):

    loss = (y_true - y_pred) ** 2
    loss = loss.mean(dim=1)
    loss = torch.sqrt(loss)
    loss = loss.sum(dim=0)

    return loss

def inference(val_loader, model, loss_fn, transforms_mol, transforms_cell, device):

    with torch.no_grad():
        loss = 0

        for i, batch in enumerate(val_loader):
            cell_types, sm_names, expressions = batch
            
            fps = list(sm_names).copy()
            for t in transforms_mol:
                fps = t(fps) # TODO: List of transforms, also as input
            fps = fps.to(device)

            for t in transforms_cell:
                cell_types = t(cell_types)
            cell_types = cell_types.to(device)

            expressions = expressions.to(device)
            
            y_pred = model(fps, cell_types)
            loss = loss_fn(y_pred, expressions)
        
        return loss

def train(train_loader, val_loader, model, args, loss_fn, optimizer, scheduler):
    
    lr = args['lr']
    epochs = args['epochs']
    transforms_mol = args['mol_transform']
    transforms_cell = args['cell_transform']

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # print(device)
    else:
        device = torch.device("cpu")

    model.to(device)

    train_losses, val_losses = [], []
    for epoch in tqdm(range(epochs)):
        model.train()

        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()

            cell_types, sm_names, expressions = batch

            fps = list(sm_names)
            for transform in transforms_mol:
                fps = transform(fps)
            fps = fps.to(device)

            for transform in transforms_cell:
                cell_types = transform(cell_types)
            cell_types = cell_types.to(device)

            expressions = expressions.to(device)

            y_pred = model(fps, cell_types)
            loss = loss_fn(y_pred, expressions)
            loss.backward()
            optimizer.step()
        
        model.eval()
        train_losses.append(loss.cpu().detach().numpy())
        val_losses.append(inference(val_loader, model, loss_fn, transforms_mol, 
                                    transforms_cell, device).cpu().detach().numpy())
        scheduler.step(val_losses[-1])
    
    return train_losses, val_losses

if __name__ == "__main__":
    pass