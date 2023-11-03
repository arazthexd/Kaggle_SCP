import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import opendatasets
import torch
import rdkit
from torch.utils.data import Dataset
from dataset import *
from features import *

#get train and val x and y
de_df = pd.read_parquet("data/de_train.parquet")

from dataset import stratified_split
train_index, val_index = stratified_split(de_df["cell_type"], 0.2, 45)
from dataset import DataFrameDataset
de_df_dataset_train = DataFrameDataset(de_df.iloc[train_index], mode="df")
de_df_dataset_val = DataFrameDataset(de_df.iloc[val_index], mode="df")



mtypes = list(set(de_df["sm_name"].to_list()))
mol_transforms = {
    "morgan2_fp": TransformList([Sm2Smiles("config/sm_smiles.csv", mode="path"), Smiles2Mol(), Mol2Morgan(2048, 2)]),
    "morgan3_fp": TransformList([Sm2Smiles("config/sm_smiles.csv", mode="path"), Smiles2Mol(), Mol2Morgan(2048, 3)]),
    "one_hot": TransformList([Type2OneHot(mtypes)])
}

ctypes = list(set(de_df["cell_type"].to_list()))

file_names = ["data/temp/"+name.replace(" ", "_").replace("+", "")+"_control_mean.csv"
              for name in ctypes]
gene_num = len(pd.read_csv(file_names[0]))

cell_transforms = {
    "one_hot": TransformList([Type2OneHot(ctypes)]),
    "gene_exp": TransformList([CType2CSVEncoding(ctypes, file_names)])
    # "gene_exp": TransformList([CType2CSVEncoding(ctypes, file_names), NormCount2CPM()])
}


de_dataset_train = DEDataset(de_df_dataset_train, mol_transforms, cell_transforms)
de_dataset_val = DEDataset(de_df_dataset_val, mol_transforms, cell_transforms)


de_dataset_train.configure(sm_out_feature="morgan2_fp", cell_out_feature="gene_exp", return_y=True, ae_mode=False)
de_dataset_val.configure(sm_out_feature="morgan2_fp", cell_out_feature="gene_exp", return_y=True, ae_mode=False)

X_train, y_train = de_dataset_train[:]
X_val, y_val = de_dataset_val[:]

mols = np.array(X_train[0])
cells = np.array(X_train[1])
X_train = np.concatenate([mols, cells], axis=1)
val_mols = np.array(X_val[0])
val_cells = np.array(X_val[1])
X_val = np.concatenate([val_mols, val_cells], axis=1)

from sklearn.multioutput import MultiOutputRegressor
regressor = SVR(kernel="poly")
regressor = MultiOutputRegressor(regressor, n_jobs=-1)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_val)
loss = loss_mrrmse(y_val, y_pred)
print(loss)