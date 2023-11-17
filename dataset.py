import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from features import Type2OneHot, Sm2Smiles, Smiles2Mol, Mol2Morgan

class DataFrameDataset(Dataset):

    # TODO: implement batch loading for large datasets
    
    def __init__(self, path_or_df, mode="path_parquet"):

        super(DataFrameDataset, self).__init__()
        
        self.mode = mode
        if mode == "path_parquet":
            self.df = pd.read_parquet(path_or_df, "pyarrow")
        elif mode == "path_csv":
            self.df = pd.read_csv(path_or_df)
        elif mode == "df":
            self.df = path_or_df
        else:
            print("WRONG OUTPUT TYPE")
            return
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        if self.mode in ["df", "path_csv", "path_parquet"]:
            return self.df.iloc[index]
        else:
            print("NOT IMPLEMENTED")
            return
    
class DEDataset(Dataset):
    
    def __init__(self, df_dataset, sm_transforms={}, cell_transforms={}, contains_y=True, mode="whole"):
        super(DEDataset, self).__init__()

        meta_columns = ["cell_type", "sm_name", "sm_lincs_id", "SMILES", "control"]

        if mode == "whole":
            input_df = df_dataset[:]
            if contains_y:
                self.expressions = torch.tensor(
                    input_df.drop(meta_columns, axis=1).values
                ).float()

            self.sm_names = input_df["sm_name"].to_list()
            self.cell_names = input_df["cell_type"].to_list()

            self.sm_feats = dict()
            self.sm_types = set(self.sm_names)
            for sm_transform_name in sm_transforms:
                self.sm_feats[sm_transform_name] = sm_transforms[sm_transform_name](self.sm_names)
        
            self.cell_feats = dict()
            self.cell_types = set(self.cell_names)
            for cell_transform_name in cell_transforms:
                self.cell_feats[cell_transform_name] = cell_transforms[cell_transform_name](self.cell_names)

        else:
            print("NOT IMPLEMENTED")

        # DEFAULT CONFIG
        self.sm_out_feature = self.sm_names
        self.cell_out_feature = self.cell_names
        self.return_y = True
        self.ae_mode = False


    def __len__(self):
        return len(self.sm_names)

    def configure(self, sm_out_feature=None, cell_out_feature=None, return_y=None, ae_mode=False):
        if sm_out_feature:
            if sm_out_feature == "sm_name":
                self.sm_out_feature = self.sm_names
            else:
                self.sm_out_feature = self.sm_feats[sm_out_feature]
        if cell_out_feature:
            if cell_out_feature == "cell_name":
                self.cell_out_feature = self.cell_names
            else:
                self.cell_out_feature = self.cell_feats[cell_out_feature]
        if return_y != None:
            self.return_y = return_y
        
        self.ae_mode = ae_mode
    
    def __getitem__(self, index):
        
        if self.ae_mode == "sm":
            x = (self.sm_out_feature[index], )
            y = self.sm_out_feature[index]
            return x, y
        elif self.ae_mode == "cell":
            x = (self.cell_out_feature[index], )
            y = self.cell_out_feature[index]
            return x, y
        elif self.ae_mode == "de":
            x = (self.expressions[index, :], )
            y = self.expressions[index, :]
            return x, y

        x = (self.sm_out_feature[index], self.cell_out_feature[index])
        if self.return_y == True:
            y = self.expressions[index, :]
            return x, y
        else:
            return x


def stratified_split(stratify, test_size, seed):
    sample_size = len(stratify)
    idx = np.arange(sample_size)
    idx_train, idx_val = train_test_split(idx, test_size=test_size, 
                         random_state=seed, stratify=stratify)
    return idx_train, idx_val