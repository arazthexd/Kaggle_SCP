import numpy as np
import torch
import torch.nn as nn
import torch.functional as Func


class GCNLayer(nn.Module):

    def __init__(self, in_dim, out_dim, activation="relu"):
        super(GCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        self.linear = nn.Linear(in_dim, out_dim)
        if activation=="relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
    
    def forward(self, X):
        out = self.linear(X)
        if not self.activation:
            return out
        elif self.activation=="relu":
            return self.activation(out)
        
class GCN(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_genes):
        super(GCN, self).__init__()

        self.gcn_layer1 = GCNLayer(in_dim, hidden_dim)
        self.gcn_layer2 = GCNLayer(hidden_dim, num_genes, activation=False)
        
    def forward(self, A, X):
        A = torch.from_numpy(A).float()
        F = torch.mm(A, X)
        F = gcn_layer1(F)
        F = torch.mm(A, F)
        output = gcn_layer2(F)
        return output

def loss_mrrmse(y_pred, y_true):

    loss = (y_true - y_pred) ** 2
    loss = loss.mean(dim=1)
    loss = torch.sqrt(loss)
    loss = loss.mean(dim=0)

    return loss


