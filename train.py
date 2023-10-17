from data_utils import Sm2Smiles, Smiles2Mol, Mol2Morgan
import pandas as pd

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