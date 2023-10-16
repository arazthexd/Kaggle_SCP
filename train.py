from data_utils import Sm2Smiles, Smiles2Mol, Mol2Morgan
import pandas as pd

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






def train(train_loader, val_loader, model, args, loss_fn, optimizer):
    
    lr = args['lr']
    epochs = args['epochs']

    sm_smiles_df = pd.read_csv("config/sm_smiles.csv")
    sm_smiles_dict = sm_smiles_df.set_index("sm_name").to_dict()["SMILES"]
    sm2smiles = Sm2Smiles(sm_smiles_dict)
    smiles2mol = Smiles2Mol()
    mol2morgan = Mol2Morgan()

    model.train()
    for epoch in epochs:
        model.zero_grad()
        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            cell_types, sm_names, expressions = batch
            fps = mol2morgan(smiles2mol(sm2smiles(sm_names)))
            y_pred = model(fpa)
            loss = loss_fn(y_pred, expressions)
            loss.backwards()
            optimizer.step()

        model.eval()
        
        #write inference

            





if __name__ == "__main__":
    pass