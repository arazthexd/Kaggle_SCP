import pandas as pd

import torch
import torch.nn as nn
from torch_geometric.data import Data

from features import *

class BaselineModel(nn.Module):

  def __init__(self, cell_in, mol_in, out_size) -> None: # TODO: Arch Path...
    super().__init__()

    self.cell_in = cell_in
    self.mol_in = mol_in
    
    self.module_list = nn.ModuleList([
      nn.Linear(cell_in+mol_in, 1000),
      nn.Dropout(0.3),
      nn.Tanh(),
      nn.Linear(1000, 1000),
      nn.Dropout(0.3),
      nn.Tanh(),
      nn.Linear(1000, 1000),
      nn.Dropout(0.3),
      nn.Tanh(),
      nn.Linear(1000, out_size)
    ])
    
  def forward(self, x_cell, x_mol, device="cpu"):

    x = torch.concat((x_cell, x_mol), dim=1)
    x = x.to(device)
    for module in self.module_list:
      x = module(x)
    
    return x

class DEAutoEncoder(nn.Module):

  def __init__(self, de_in, bottleneck, noise=0.3) -> None: # TODO: Arch Path...
    super().__init__()

    self.encoder = nn.Sequential(
      nn.Dropout(noise),
      nn.Linear(de_in, 7000),
      nn.Tanh(),
      nn.Linear(7000, 4000),
      nn.Tanh(),
      nn.Linear(4000, 2000),
      nn.Tanh(),
      nn.Linear(2000, bottleneck)
    )
    self.decoder = nn.Sequential(
      nn.Linear(bottleneck, 2000),
      nn.Tanh(),
      nn.Linear(2000, 4000),
      nn.Tanh(),
      nn.Linear(4000, 7000),
      nn.Tanh(),
      nn.Linear(7000, de_in)
    )
    
  def forward(self, x, device="cpu"):
    
    x = x.to(device)
    x = self.encoder(x)
    x = self.decoder(x)
    
    return x



class GeneTransformer(nn.Module):
    def __init__(self, d_model, n_heads, gene_num, ff_dim) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_size = gene_num
        self.n_heads = n_heads
        self.emb = nn.Embedding(self.emb_size, self.d_model)
        self.enc_1 = nn.TransformerEncoderLayer(self.d_model, nhead=self.n_heads, dim_feedforward=ff_dim)
        self.enc_2 = nn.TransformerEncoderLayer(self.d_model, nhead=self.n_heads, dim_feedforward=ff_dim)
        self.enc_3 = nn.TransformerEncoderLayer(self.d_model, nhead=self.n_heads, dim_feedforward=ff_dim)
        self.dec_1 = nn.TransformerDecoderLayer(self.d_model, nhead=self.n_heads, dim_feedforward=ff_dim)
        self.dec_2 = nn.TransformerDecoderLayer(self.d_model, nhead=self.n_heads, dim_feedforward=ff_dim)
        self.mha_custom = nn.MultiheadAttention(self.d_model, num_heads=self.n_heads)

    def forward(self, x_enc, x_dec, x_value):
        
        x_enc = self.encode(x_enc)
        x_dec = self.decode(x_dec, x_enc)
        x = self.mha(query=x_dec, key=x_enc, value=x_value)
        return x
    
    def encode(self, x_enc):
       x_enc = self.emb(x_enc)
       x_enc = self.enc_3(self.enc_2(self.enc_1(x_enc)))
       return x_enc
    
    def decode(self, x_dec, x_enc):
       x_dec = self.emb(x_dec)
       x_dec = self.dec_2(self.dec_1(x_dec, x_enc), x_enc)
       return x_dec