from typing import Any
import numpy as np
import dask.dataframe as dd

import opendatasets as od

import pandas as pd # TODO: We might need to use dask
from sklearn.model_selection import train_test_split

from rdkit import Chem
from rdkit.Chem import AllChem

import os

import torch
from torch.utils.data import Dataset

def download_data(url, redownload=False):
    if not os.path.exists("./data"):
        os.mkdir("./data")
    od.download(url, "./data", force=redownload)

# def split(data, test_size, seed=42): # OLD VERSION
#     expression = data[:]['expression']
#     cell_type = data[:]['cell_type']
#     sm_name = data[:]['sm_name']
#     smiles = data[:]['smiles']
#     expression_train, expression_val, cell_type_train, cell_type_val, \
#         sm_name_train, sm_name_val, smiles_train, smiles_val = train_test_split(
#         expression, cell_type, sm_name, smiles, test_size=test_size, 
#         stratify=cell_type, random_state=seed)
#     out_dic_train = {
#         'expression': expression_train,
#         'cell_type': cell_type_train,
#         'sm_name': sm_name_train,
#         'smiles': smiles_train
#     }
#     out_dic_val = {
#         'expression': expression_val,
#         'cell_type': cell_type_val,
#         'sm_name': sm_name_val,
#         'smiles': smiles_val
#     }
#     return out_dic_train, out_dic_val

def stratified_split(stratify, test_size, seed):
    sample_size = len(stratify)
    idx = np.arange(sample_size)
    idx_train, idx_val = train_test_split(idx, test_size=test_size, 
                         random_state=seed, stratify=stratify)
    return idx_train, idx_val


class SCPDataSet(Dataset): # TODO: Totally incomplete

    def __init__(self, root_dir=".\data", 
                 file_names=None,
                 genes_list=None) -> None:
        super().__init__()

        # Define filenames, update in case of input, and define root_dir
        file_names_default = { # TODO: Other file types, if needed...
            "diff_expression": "de_train.parquet",
            "multiome": "multiome_train.parquet"
        }
        self.root_dir = root_dir
        if file_names:
            self.file_names = file_names_default.update(file_names)
        else:
            self.file_names = file_names_default

        self.diff_exps_df = dd.read_parquet(f"{self.root_dir}/{self.file_names['diff_expression']}")
        self.multiome_df = dd.read_parquet(f"{self.root_dir}/{self.file_names['multiome']}", n_)

        # Determine the list of the gene names used in the dataset:
        if not genes_list:
            print("Expression columns not given, looking for file...")
            if not os.path.exists("config/genes_list.txt"): # TODO: Decide the format
                print("No file found for expression columns, dropping default meta columns...")
                self.genes_list = self.diff_exps_df.drop(["cell_type", "sm_name", "sm_lincs_id", 
                                                  "SMILES", "control"], axis=1).columns
            else:
                with open("config/genes_list.txt", "r") as f:
                    self.genes_list = f.readlines()
        else:
            self.genes_list = genes_list
        
    def __len__(self):
        return len(self.diff_exps_df)
    
    def __getitem__(self, index) -> torch.tensor:
        
        samples = self.diff_exps_df.iloc[index]
        expression = torch.from_numpy(samples[self.genes_list].values.astype('float32'))
        cell_type = samples["cell_type"].to_list() # TODO: For single index --> error
        sm_name = samples["sm_name"].to_list()
        smiles = samples["SMILES"].to_list()
        out_dic = {
            "expression": expression,
            "cell_type": cell_type,
            "sm_name": sm_name,
            "smiles": smiles
        }
        return out_dic

class SelectExpressions(object):
    def __init__(self, gene_ids="all") -> None:
        self.genes_ids = gene_ids

    def __call__(self, sample_dic) -> torch.tensor:
        out = sample_dic["expression"]
        if self.genes_ids != "all":
            out = out.index_select(dim=1, index=self.genes_ids)
        return out
    
class Smiles2Mol(object):
    def __init__(self) -> None:
        pass

    def __call__(self, smiles_list) -> list:
        mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        return mol_list

class Mol2Morgan(object):
    def __init__(self, vec_bit=2048, radius=2) -> None:
        self.vec_bit = vec_bit
        self.radius = radius

    def __call__(self, mol_list) -> list:
        fps_list = [AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.vec_bit) 
                    for mol in mol_list]
        out_tensor = torch.tensor(fps_list)
        return out_tensor



        

        



