import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter

from einops import rearrange, repeat, reduce, pack, unpack

import math
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.
    """
    def __init__(
            self, 
            in_dims: int, 
            out_dims: int, 
            bias: bool =True
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dims, out_dims, bias=bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Graph Convolution layer.

        Args:
            x: Node features
            adj: Adjacency matrix
        """
        support = self.fc1(x)
        output = torch.sparse.mm(adj, support)
        return output


class GCN(nn.Module):
    """
    Multi-layer GCN with dropout and log-softmax output.
    """
    def __init__(
        self,
        num_layers: int,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        layer_sizes = [in_dims] + [hidden_dims] * num_layers + [out_dims]

        self.layers = nn.ModuleList([
            GraphConvolution(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN.

        Args:
            x: Node features
            adj: Adjacency matrix
        """
        for i, layer in enumerate(self.layers[:-1]):
            x = self.dropout(F.relu(layer(x, adj)))
            
        x = self.layers[-1](x, adj)
        return self.logsoftmax(x)


class LSTMGCN(nn.Module):
    """
    LSTM-GCN hybrid model as described in https://arxiv.org/abs/2408.05659v1.
    
    Architecture:
    - LSTM (64 units, tanh)
    - Dense Layer 1 (32 units, tanh)
    - Dense Layer 2 (16 units, tanh)
    - GCN (12 graphs, tanh)
    - Output Layer (1 unit, linear)
    """
    def __init__(
            self, 
            input_size: int, 
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=64, 
            batch_first=True
        )

        self.dense1 = nn.Linear(64, 32, bias=True)
        self.dense2 = nn.Linear(32, 12, bias=True)

        self.gcn = GCN(
            num_layers=1,
            in_dims=12,
            hidden_dims=12,
            out_dims=12,
            dropout=0.0
        )

        self.output = nn.Linear(12, 1) 
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSTM-GCN model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            adj: Adjacency matrix of shape (num_nodes, num_nodes)
            
        Returns:
            Tensor of shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)

        x = torch.tanh(self.dense1(lstm_out[:, -1, :]))  
        dense2_out = torch.tanh(self.dense2(x))

        x = self.gcn(dense2_out, adj)

        x = self.output(x) + dense2_out 

        return x
