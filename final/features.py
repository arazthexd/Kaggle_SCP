from typing import Any
import copy

import numpy as np
import dask.dataframe as dd

import opendatasets as od

import pandas as pd # TODO: We might need to use dask

from sklearn.preprocessing import OneHotEncoder

from rdkit import Chem
from rdkit.Chem import AllChem

from operator import itemgetter

import torch

class TransformList(object):

    def __init__(self, transform_list):
        self.transform_list = transform_list
    
    def __call__(self, x):
        
        y = x.copy()
        for transform in self.transform_list:
            y = transform(y)
        return y
    

class Sm2Smiles(object):
    def __init__(self, sm_dict_or_path, mode="dict") -> None:
        if mode == "dict":
            self.sm_dict = sm_dict_or_path
        elif mode == "path":
            df = pd.read_csv(sm_dict_or_path)
            self.sm_dict = df.set_index("sm_name").to_dict()["SMILES"]

    def __call__(self, sm_names) -> list:
        smiles = list(itemgetter(*sm_names)(self.sm_dict))
        return smiles
    
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
        return out_tensor.float()

class Type2OneHot(object):
    def __init__(self, types) -> None:
        self.oh_encoder = OneHotEncoder(sparse_output=False)

        if type(types) != np.array:
            types = np.array(types)
        self.oh_encoder.fit(types.reshape(-1, 1))
        
    def __call__(self, types) -> list:
        if type(types) != np.array:
            types = np.array(types)
        encoded = self.oh_encoder.transform(types.reshape(-1, 1))
        encoded = torch.tensor(encoded)

        return encoded.float()