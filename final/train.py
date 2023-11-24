### IMPORTS ###
from argparse import ArgumentParser

from tqdm import tqdm

import torch

### FUNCTIONS ###
def loss_mrrmse(y_pred, y_true, mask=None, device="cuda:0"):

    if mask == None:
        mask = torch.ones(y_pred.shape).to(device)
    loss = mask * ((y_true - y_pred) ** 2)
    n = mask.sum(1)
    loss = loss.sum(dim=1)[n!=0] / n[n!=0]
    loss = torch.sqrt(loss)
    loss = loss.mean(dim=0)

    return loss

def train_one_epoch(model, train_loader, process_batch, 
                    optimizer, device):

    # Send model to device
    model.to(device)

    # Set model to train mode
    model.train()

    # Iterate over batches and take optimization steps
    losses = []
    for batch in train_loader:
        
        loss = process_batch(batch)
        # x_batch, (y_batch, mask_batch) = batch
        # y_batch = y_batch.to(device)
        # mask_batch = mask_batch.to(device)
        # y_pred = model(*x_batch, device) # TODO: Send to device the x in model?

        # loss = loss_fn(y_pred, y_batch, mask_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss)
     
    return losses

def infer_model(model, data_loader, process_batch, 
                metrics: dict, device, calculate_loss=False):

    data_len = len(data_loader)
    
    # Send model to device
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Create Output Dict
    metric_values = dict()
    for metric_name in metrics:
        metric_values[metric_name] = 0

    with torch.no_grad():
        
        losses = []
        for batch in data_loader:

            loss = process_batch(batch)
            # x_batch, (y_batch, mask_batch) = batch
            # y_batch = y_batch.to(device)
            # mask_batch = mask_batch.to(device)
            # # print(x_batch)
            # y_pred = model(*x_batch, device)

            # if calculate_loss:
            #     loss = loss_fn(y_pred, y_batch, mask_batch)
            #     losses.append(loss)

            # for metric_name in metrics:
            #     metric_value = metrics[metric_name](y_pred, y_batch)
            #     metric_values[metric_name] += (metric_value / data_len)

            losses.append(loss)

    if calculate_loss:
        return losses, metric_values
    else:
        return metric_values
        
def train_many_epochs(model, train_loader, val_loader, epochs,
                      process_batch, optimizer, scheduler=None, 
                      metrics=[], writer=None, device="cpu"):
    
    for epoch in tqdm(range(epochs)):

        # Train model for one epoch and calculate metrics for the 
        # resulting model...
        train_b_losses = train_one_epoch(model, train_loader, process_batch, 
                                         optimizer, device)
        train_b_metrics = infer_model(model, train_loader, process_batch, 
                                      metrics, device)
        val_b_losses, val_b_metrics = infer_model(model, val_loader, process_batch, 
                                    metrics, device, calculate_loss=True)
        
        epoch_train_loss = sum(train_b_losses) / len(train_b_losses)
        epoch_val_loss = sum(val_b_losses) / len(val_b_losses)
        if writer:
            writer.add_scalar("Training Loss", epoch_train_loss, global_step=epoch)
            writer.add_scalar("Validation Loss", epoch_val_loss, global_step=epoch)
        if scheduler:
            scheduler.step(epoch_train_loss)



if __name__ == "__main__":

    # Parse input arguments:
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str)
    ### TODO: FINISH





