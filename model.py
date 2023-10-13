import torch
import torch.nn as nn

class BasicModel(nn.Module):

  def __init__(self, input_size:int, layers_size:list, output_size:int) -> None:
    
    super(BasicModel, self).__init__()
    
    self.layers = []
    all_layers = [input_size] + layers_size + [output_size]
    for i in range(len(all_layers)-1):
      layer_i = nn.Linear(in_features=all_layers[i], out_features=all_layers[i+1])
      self.layers.append(layer_i)
    
    self.activation = nn.ReLU()

  def forward(self, x):
    
    x = self.layers[0](x)
    for layer in self.layers[1:]:
      x = self.activation(x)
      x = layer(x)

    return x