import torch
from torch import nn

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pyarrow.parquet import ParquetFile, read_table
import numpy as np

from tqdm import tqdm

class ContPert2DE(nn.Module):

    def __init__(self, c2p_model, layers_sizes, out_dim) -> None:
        
        self.c2p_model = c2p_model
        self.layers = nn.ModuleList()
        for layers_sizes


class Control2Pert(nn.Module):

    def __init__(self, in_dim, bottle_dim, pert_dim, out_dim) -> None:
        super().__init__()

        self.enc_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, 2000),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2000, bottle_dim),
            nn.Dropout(0.1),
            nn.ReLU()
        )
        self.dec_layer = nn.Sequential(
            nn.Linear(bottle_dim+pert_dim, 3000),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(3000, out_dim)
        )

        self.mol_drop = nn.Dropout(0.3)

        self.float()

        self.mode = "normal"
    
    def encode(self, x):
        x = self.enc_layer(x)
        return x
    
    def decode(self, x, pert):
        pert = self.mol_drop(pert)
        x = torch.concat([x, pert], dim=1)
        x = self.dec_layer(x)
        return x
    
    def forward(self, x, pert, device):
        x = x.to(device).float()
        pert = pert.to(device).float()
        x = self.encode(x)
        x = self.decode(x, pert)
        return x

        