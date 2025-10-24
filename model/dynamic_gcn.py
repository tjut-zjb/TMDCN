import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse, to_dense_batch
from torch_geometric.nn import ChebConv


class DynamicChebNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, K):

        super(DynamicChebNet, self).__init__()

        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.convs.append(ChebConv(hidden_dim, output_dim, K=K))

    def forward(self, X, A):

        batch_size, num_sensors, time_steps, embed_dim = X.shape

        # Flatten temporal and embedding dimensions into node features: (B, N, T*D)
        X = X.reshape(batch_size, num_sensors, time_steps * embed_dim)

        # Convert dense adjacency to sparse (edge_index, edge_weight)
        edge_data = [dense_to_sparse(A[b]) for b in range(batch_size)]
        edge_indices, edge_weights = zip(*edge_data)

        # Build batched PyG graph
        data_list = [
            Data(x=X[b], edge_index=edge_indices[b], edge_weight=edge_weights[b])
            for b in range(batch_size)
        ]
        batch = Batch.from_data_list(data_list)

        # Apply ChebConv layers
        for i, conv in enumerate(self.convs):
            batch.x = conv(batch.x, batch.edge_index, batch.edge_weight)
            if i < self.num_layers - 1:
                batch.x = F.relu(batch.x)
                batch.x = self.dropout(batch.x)

        # (B, N, output_dim)
        output, _ = to_dense_batch(batch.x, batch.batch)

        return output
