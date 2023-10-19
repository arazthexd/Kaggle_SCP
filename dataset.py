import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class RawDEDataset(Dataset):
    
    def __init__(self, de, mode="path"):

        super(RawDEDataset, self).__init__()
        
        if mode == "path":
            self.de_df = pd.read_parquet(de, "pyarrow")
        elif mode == "df":
            self.de_df = de
        else:
            print("WRONG OUTPUT TYPE")
            return
    
    def __len__(self):
        return len(self.cell_types)
    
    def __getitem__(self, index):
        
        out_df = self.de_df.iloc[index]
        out_dataset = RawDEDataset(out_df, mode="df")
        return out_dataset
        


def stratified_split(stratify, test_size, seed):
    sample_size = len(stratify)
    idx = np.arange(sample_size)
    idx_train, idx_val = train_test_split(idx, test_size=test_size, 
                         random_state=seed, stratify=stratify)
    return idx_train, idx_val