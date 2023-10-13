import opendatasets as od
import pandas as pd # TODO: We might need to use dask

import os

import torch
from torch.utils.data import Dataset

def download_data(url, redownload=False):
    if not os.path.exists("./data"):
        os.mkdir("./data")
    od.download(url, "./data", force=redownload)



class SCPDataSet(Dataset): # TODO: Totally incomplete

    def __init__(self, root_dir, 
                 de_train_file="de_train.parquet",
                 gene_list="all") -> None:
        super().__init__()

        self.meta_columns = 
        self.root_dir = root_dir
        self.gene_list = gene_list
        self.de_train = pd.read_parquet(f"{root_dir}/{de_train_file}")
        
    def __len__(self):
        return len(self.de_train)
    
    def __getitem__(self, index) -> torch.tensor:
        
        samples = self.de_train.iloc[index].drop([""])
        


        

        



