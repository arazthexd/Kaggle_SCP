def runTraining(args, model, optimizer, loss_fn, train_loader, val_loader, transforms_x, transforms_y): # TODO: Dataloader code
    
    lr = args['lr']
    batch_size = args['batch_size']
    epochs = args['epochs']


    for epoch in range(epochs):

        model.train()
        for data_batch in train_loader:
            model.zero_grad()
            optimizer.zero_grad()
            x_batch = data_batch.copy()
            y_batch = data_batch.copy()
            for transform in transforms_x:
                x_batch = transform(x_batch)
            for transform in transforms_y:
                y_batch = transform(y_batch)
            
            y_pred = model(x_batch)

            loss = loss_fn(y_pred, y_batch)

            loss.backwards()
            optimizer.step()
        
        
        model.eval()




if __name__ == "__main__":
    pass