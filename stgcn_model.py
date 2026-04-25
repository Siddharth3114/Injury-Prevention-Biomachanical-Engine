import torch
import torch.nn as nn
from graph import Graph

class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance=1):
        super().__init__()
        
        # In a full ST-GCN, max_graph_distance denotes the spatial partition (e.g. 3).
        # We are using 1 for our simplified normalized matrix.
        self.conv = nn.Conv2d(in_channels, out_channels * max_graph_distance, kernel_size=1)
        
    def forward(self, x, A):
        """
        x: (N, C, T, V)
        A: (max_graph_distance, V, V)
        """
        # x -> (N, C, T, V)
        x = self.conv(x)
        # x -> (N, out_channels * max_graph_distance, T, V)
        
        N, C, T, V = x.size()
        
        # In our case max_graph_distance = 1, so out_channels == C
        # We multiply our features by the Adjacency matrix A.
        # x reshaped: (N, C, T, V) -> (N, C*T, V)
        x = x.view(N, C * T, V)
        
        # A[0]: (V, V)
        # BMM: (N, C*T, V) x (V, V) -> (N, C*T, V)
        # To apply A to the last dimension (vertices)
        
        # Convert A to tensor
        A_tensor = torch.from_numpy(A[0]).float().to(x.device)
        
        # Matmul: x @ A_tensor. 
        # (N, C*T, V) @ (V, V) -> (N, C*T, V)
        x = torch.matmul(x, A_tensor)
        
        # Reshape back to (N, C, T, V)
        x = x.view(N, C, T, V)
        
        return x

class STGCN_Block(nn.Module):
    """
    Applies Spatial Graph Convolution followed by Temporal Convolution.
    """
    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        
        self.sgc = SpatialGraphConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Temporal Convolution (1D over time dimension, implemented via 2D Conv with kernel_size=(Kt, 1))
        # t_kernel_size = 9 as per original ST-GCN paper
        self.tc = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0))
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, A):
        # Apply Spatial Graph Convolution
        res = self.residual(x)
        x = self.sgc(x, A)
        x = self.relu(x)
        
        # Apply Temporal Convolution
        x = self.tc(x)
        x = x + res
        x = self.relu(x)
        
        return x

class Model(nn.Module):
    """
    Base ST-GCN Architecture (Proof of Concept)
    """
    def __init__(self, in_channels, num_class, graph_args={}, edge_importance_weighting=True):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        # Shape: (1, 33, 33)
        A = self.graph.A 
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32, requires_grad=False))
        
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for i in range(2)
        ])

        # build networks
        # We start by mapping the input channels (e.g. 4 for x,y,z,vis) to 64
        self.data_bn = nn.BatchNorm1d(in_channels * self.A.size(1))
        
        # ST-GCN Blocks
        self.st_gcn_networks = nn.ModuleList((
            STGCN_Block(in_channels, 64, residual=False),
            STGCN_Block(64, 64)
        ))

    def forward(self, x):
        """
        x: Tensor of shape (N, C, T, V, M)
        N: Batch Size
        C: Channels (4)
        T: Frames (Time)
        V: Vertices (33)
        M: Persons (1)
        """
        N, C, T, V, M = x.size()
        
        # The logic usually handles multi-person by stacking them into the batch dimension.
        # Shape becomes: (N * M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        # Normalization
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous() # (N*M, C, T, V)
        
        # Forward through ST-GCN Blocks
        for i, gcn in enumerate(self.st_gcn_networks):
            # Scale adjacency matrix by edge importance weight
            importance_A = self.A * self.edge_importance[i]
            x = gcn(x, importance_A.detach().numpy()) # passed as numpy for our basic sgc implementation
            
        # Returning the intermediate feature representation
        # Shape: (N * M, Out_C, T_out, V)
        
        return x
