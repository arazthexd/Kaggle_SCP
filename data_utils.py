from typing import Any

import opendatasets as od

import pandas as pd # TODO: We might need to use dask

from rdkit import Chem
from rdkit.Chem import AllChem

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
                 genes_list=None) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.de_train = pd.read_parquet(f"{root_dir}/{de_train_file}")

        # Determine the list of the gene names used in the dataset:
        if not genes_list:
            print("Expression columns not given, looking for file...")
            if not os.path.exists("config/genes_list.txt"): # TODO: Decide the format
                print("No file found for expression columns, dropping default meta columns...")
                self.genes_list = self.de_train.drop(["cell_type", "sm_name", "sm_lincs_id", 
                                                  "SMILES", "control"]).columns
            else:
                with open("config/genes_list.txt", "r") as f:
                    self.genes_list = f.readlines()
        else:
            self.genes_list = genes_list
        
    def __len__(self):
        return len(self.de_train)
    
    def __getitem__(self, index) -> torch.tensor:
        
        samples = self.de_train.iloc[index]
        expression = torch.from_numpy(samples[self.genes_list].value)
        cell_type = samples["cell_type"].to_list()
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
        mol_list = [Chem.GetMorganFingerprintAsVect(mol) for mol in mol_list]
        out_tensor = torch.stack(mol_list, dim=0)
        return out_tensor



        

        



