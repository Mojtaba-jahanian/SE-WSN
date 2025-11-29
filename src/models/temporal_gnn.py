import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class TemporalGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.gcn = GCNConv(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.node_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, graph_seq):
        spatial_embeddings = []
        for G in graph_seq:
            x, edge_index = G.x, G.edge_index
            h = torch.relu(self.gcn(x, edge_index))
            spatial_embeddings.append(h)

        H = torch.stack(spatial_embeddings, dim=1)  # [N, T, hidden]
        H_temporal, _ = self.gru(H)
        h_final = H_temporal[:, -1, :]
        node_risk = torch.sigmoid(self.node_head(h_final))
        return node_risk, h_final
