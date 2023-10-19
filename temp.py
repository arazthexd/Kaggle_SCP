def train_one_epoch(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device):

    # Send model to device
    model.to(device)

    # Set model to train mode
    model.train()

    # Iterate over batches and take optimization steps
    for batch in train_loader:

        x_batch, y_batch = batch
        y_batch = y_batch.to(device)
        y_pred = model(x_batch, device)

        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
    return loss





