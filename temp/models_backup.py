class NNRegressor(nn.Module):
  pass # TODO: Write!

class CombinerModel(nn.Module):

  def __init__(self, mol_encoder, cell_encoder, regressor):
    super(CombinerModel, self).__init__()

    self.mol_encoder = mol_encoder
    self.cell_encoder = cell_encoder
    self.regressor = regressor
    self.float()

  def forward(self, x_mol, x_cell, device):
    
    x_mol, x_cell = x_mol.to(device), x_cell.to(device)
    x_mol_enc = self.mol_encoder(x_mol)
    x_cell_enc = self.cell_encoder(x_cell)
    y = self.regressor(torch.concat((x_mol_enc, x_cell_enc), dim=1))

    return y

class VecEncoder(nn.Module):

  def __init__(self, layer_sizes: list[int], drop_rate=0):
    super(VecEncoder, self).__init__()

    self.layers = nn.ModuleList()
    for i in range(len(layer_sizes)-1):
      in_size, out_size = layer_sizes[i], layer_sizes[i+1]
      layer = nn.Linear(in_size, out_size)
      self.layers.append(layer)
    
    self.activation = nn.Tanh()
    self.drop_out = nn.Dropout(drop_rate)

  def forward(self, x):

    if len(self.layers) == 0:
      return x
    
    if len(self.layers) == 1:
      x = self.layers[0](x)
      return x
    
    x = self.layers[0](x)
    x = self.activation(x)
    for layer in self.layers[1:-1]:
      x = layer(x)
      x = self.activation(x)
      x = self.drop_out(x)
      
    x = self.layers[-1](x)
    
    
    return x
  
class VecAutoEncoder(nn.Module):

  def __init__(self, layer_sizes: list[int]):
    super().__init__()
    
    self.encoder = nn.ModuleList()
    for s1, s2 in zip(layer_sizes[:-2], layer_sizes[1:-1]):
      self.encoder.append(
        nn.Sequential(
          nn.Linear(s1, s2),
          nn.ReLU()
        )
      )
    self.encoder.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
    
    self.decoder = nn.ModuleList()
    for s1, s2 in zip(layer_sizes[-1:1:-1], layer_sizes[-2:0:-1]):
      self.decoder.append(
        nn.Sequential(
          nn.Linear(s1, s2),
          nn.ReLU()
        )
      )
    self.decoder.append(nn.Linear(layer_sizes[1], layer_sizes[0]))

    self.float()
    
  def forward(self, x, device):

    x = x.to(device)
    x = self.encode(x, device)
    x = self.decode(x, device)

    return x.float()
  
  def encode(self, x, device="cpu"):

    x = x.to(device)
    for enc_layer in self.encoder:
      x = enc_layer(x)
    return x.float()

  def decode(self, x, device="cpu"):

    x = x.to(device)
    for dec_layer in self.decoder:
      x = dec_layer(x)
    return x.float()

class VecAEEncoder(nn.Module):

  def __init__(self, trained_ae, grads=False, device="cpu"):
    super().__init__()

    self.ae = trained_ae
    self.calc_grads = grads
    self.device = device

    self.float()

  def forward(self, x):

    x = x.to(self.device)
    if self.calc_grads == True:
      x = self.ae.encode(x, self.device)
    else:
      with torch.no_grad():
        x = self.ae.encode(x, self.device)
    
    return x.float()



class ArazTest(nn.Module):

  def __init__(self):
    
    dataData()

  def forward(mol_emb, cell_emb, ):
    pass


class BasicModel(nn.Module):

  def __init__(self, input_size:int, layers_size:list, output_size:int) -> None:
    
    super(BasicModel, self).__init__()
    
    self.layers = nn.ModuleList()
    all_layers = [input_size] + layers_size + [output_size]
    for i in range(len(all_layers)-1):
      layer_i = nn.Linear(in_features=all_layers[i], out_features=all_layers[i+1])
      self.layers.append(layer_i)
    
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(p=0.5)

    self.float()

  def forward(self, x):
    
    x = self.layers[0](x)
    for layer in self.layers[1:]:
      x = self.activation(x)
      x = self.dropout(x)
      x = layer(x)

    return x
  
class BasicModel2(nn.Module):

  def __init__(self, input_size:int, layers_size:list, output_size:int, cell_classes=6) -> None:
    
    super(BasicModel2, self).__init__()
    
    self.layers = nn.ModuleList()
    mol_layers = [input_size] + layers_size
    for i in range(len(mol_layers)-1):
      layer_i = nn.Linear(in_features=mol_layers[i], out_features=mol_layers[i+1])
      self.layers.append(layer_i)
    
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(p=0.2)

    self.final_layer = nn.Linear(in_features=layers_size[-1] + cell_classes, 
                                 out_features=output_size)

    self.float()

  def forward(self, x_mol, x_cell):
    
    x_mol = self.layers[0](x_mol)
    for layer in self.layers[1:]:
      x_mol = self.activation(x_mol)
      x_mol = self.dropout(x_mol)
      x_mol = layer(x_mol)

    x = torch.concat([x_mol, x_cell], dim=1)
    x = self.final_layer(x)
    
    return x
  
class BasicModel3(nn.Module):

  def __init__(self, input_size:int, layers_size:list, output_size:int, cell_emb_size=10) -> None:
    
    super(BasicModel2, self).__init__()
    
    self.layers = nn.ModuleList()
    mol_layers = [input_size] + layers_size
    for i in range(len(mol_layers)-1):
      layer_i = nn.Linear(in_features=mol_layers[i], out_features=mol_layers[i+1])
      self.layers.append(layer_i)
    
    self.emb = torch.nn.Embedding()
    
    self.activation = nn.ReLU()
    self.dropout = nn.Dropout(p=0.5)

    self.final_layer = nn.Linear(in_features=layers_size[-1] + cell_emb_size, 
                                 out_features=output_size)

    self.float()

  def forward(self, x_mol, x_cell):
    
    x_mol = self.layers[0](x_mol)
    for layer in self.layers[1:]:
      x_mol = self.activation(x_mol)
      x_mol = self.dropout(x_mol)
      x_mol = layer(x_mol)

    x = torch.concat([x_mol, x_cell], dim=1)
    x = self.final_layer(x)
    
    return x