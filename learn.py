import torch
import pandas as pd

class DFDataset(torch.utils.data.Dataset):
    def __init__(self, path, data_type):
        self.path = path

        if data_type == "csv":
            df = pd.read_csv(path)
        elif data_type == "parquet":
            df = pd.read_parquet(path)
        
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        x = list(self.df[index]["sm_name"]), list(self.df[index]["cell_type"])
        y = self.df.loc[:, 5:] # TODO hello!

        return x, y
    
df1 = DFDataset()
