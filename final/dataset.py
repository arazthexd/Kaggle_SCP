import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
    
class C2PDataset(Dataset):
    
    def __init__(self, data_array, meta_dict, sm_transforms={}, contains_y=True):
        super(C2PDataset, self).__init__()
        
        data_initial = data_array[:, meta_dict["mols"].index("Dimethyl Sulfoxide"), :].squeeze()
        gene_idx = (data_initial>0).sum(1)==6
        control_array = data_initial[gene_idx, :] # 16k x 6
        self.control_val, self.control_train = control_array[:, 4:], control_array[:, :4]
        self.gene_names_control = np.array(meta_dict["gene_names"])[gene_idx]
        data_pert = np.delete(data_array, meta_dict["mols"].index("Dimethyl Sulfoxide"), axis=1)
        self.data_val, self.data_train = (data_pert[gene_idx, :, 4:] - np.expand_dims(self.control_val, axis=1), 
                                          data_pert[gene_idx, :, :4] - np.expand_dims(self.control_train, axis=1))
        self.mask_val, self.mask_train = self.data_val>0, self.data_train>0

        self.meta_dict = meta_dict

        self.sm_feats = dict()
        self.sm_types = meta_dict["mols"].copy()
        self.sm_types.remove("Dimethyl Sulfoxide")
        for sm_transform_name in sm_transforms:
            self.sm_feats[sm_transform_name] = sm_transforms[sm_transform_name](self.sm_types)
        
        self.cell_types = meta_dict["cell_types"]

        self.idx_dict_train = {
            i: (i % self.data_train.shape[2], i % self.data_train.shape[2]) \
                for i in range(self.data_train.shape[2]*self.data_train.shape[1])
        }
        self.idx_dict_val = {
            i: (i % self.data_val.shape[2], i % self.data_val.shape[2]) \
                for i in range(self.data_val.shape[2]*self.data_val.shape[1])
        }

        # DEFAULT CONFIG
        self.sm_out_feature = self.sm_types
        self.control_array = self.control_train
        self.idx_dict = self.idx_dict_train
        self.pert_array = self.data_train
        self.mask_array = self.mask_train
        self.mode = "train"
        self.return_y = True
        self.ae_mode = False

    def __len__(self):

        if self.mode == "train":
            return len(self.sm_types) * self.data_train.shape[2]
        elif self.mode == "validation":
            return len(self.sm_types) * self.data_val.shape[2]

    def configure(self, sm_out_feature=None, mode=None, return_y=None, ae_mode=False):
        if sm_out_feature:
            if sm_out_feature == "sm_name":
                self.sm_out_feature = self.sm_types
            else:
                self.sm_out_feature = self.sm_feats[sm_out_feature]

        if mode:
            self.mode = mode
            if mode == "train":
                self.control_array = self.control_train
                self.idx_dict = self.idx_dict_train
                self.pert_array = self.data_train
                self.mask_array = self.mask_train
            elif mode == "validation":
                self.control_array = self.control_val
                self.idx_dict = self.idx_dict_val
                self.pert_array = self.data_val
                self.mask_array = self.mask_val

        if return_y != None:
            self.return_y = return_y
        
        self.ae_mode = ae_mode
    
    def __getitem__(self, index):
        
        # if self.ae_mode == "sm":
        #     x = (self.sm_out_feature[index], )
        #     y = self.sm_out_feature[index]
        #     return x, y
        # if self.ae_mode == "cell":
        #     x = (self.cell_out_feature[index], )
        #     y = self.cell_out_feature[index]
        #     return x, y
        # elif self.ae_mode == "de":
        #     x = (self.expressions[index, :], )
        #     y = self.expressions[index, :]
        #     return x, y

        if self.ae_mode == "initial_expr":
            x = (torch.tensor(self.control_array[:, cell_ind]), )
            return x, (x, None)
        
        if type(index) == int:
            cell_ind, mol_ind = self.idx_dict[index]
            x = (torch.tensor(self.control_array[:, cell_ind]), self.sm_out_feature[mol_ind])
            if self.return_y == True:
                y = self.pert_array[:, mol_ind, cell_ind].squeeze()
                y_mask = self.mask_array[:, mol_ind, cell_ind].squeeze()
                return x, (y, y_mask)
            else:
                return x
        else:
            x_sm, x_cell, y, y_mask = [], [], [], []
            for i in index:
                cell_ind, mol_ind = self.idx_dict[i]
                x_sm.append(self.sm_out_feature[mol_ind])
                x_cell.append(torch.tensor(self.control_array[cell_ind]))
                if self.return_y == True:
                    y.append(self.pert_array[mol_ind, cell_ind].squeeze())
                    y_mask.append(self.mask_array[mol_ind, cell_ind].squeeze())
            x_sm = torch.stack(x_sm, dim=0)
            x_cell = torch.stack(x_cell, dim=0)
            if self.return_y == True:
                y = torch.stack(y, dim=0)
                y_mask = torch.stack(y_mask, dim=0)
                return (x_cell.float(), x_sm.float()), (y.float(), y_mask.float())
            else:
                return x_cell.float(), x_sm.float()

class CP2DEDataset(Dataset):

    def __init__(self, data_array, c2p_dataset) -> None:
        super().__init__()

        self.data_array = data_array
        self.c2p_dataset = c2p_dataset

        self.idx_dict = {
            i: (i % self.data_array.shape[2], i % self.data_array.shape[2]) \
                for i in range(self.data_array.shape[2]*self.data_array.shape[1])
        }
    
    def __len__(self):
        return self.data_array.shape[1] * self.data_array.shape[2]
    
    def __getitem__(self, index):
        
        if type(index) == int:
            cell_ind, mol_ind = self.idx_dict[index]
            out = (self.c2p_dataset[index],
                   torch.tensor(self.data_array[:, mol_ind, cell_ind].squeeze()))
            return out
        
        out = []
        for i in index:
            cell_ind, mol_ind = self.idx_dict[i]
            out.append(torch.tensor(self.data_array[:, mol_ind, cell_ind].squeeze()))
        return self.c2p_dataset[index], torch.stack(out, dim=0)
        


def stratified_split(stratify, test_size, seed):
    sample_size = len(stratify)
    idx = np.arange(sample_size)
    idx_train, idx_val = train_test_split(idx, test_size=test_size, 
                         random_state=seed, stratify=stratify)
    return idx_train, idx_val