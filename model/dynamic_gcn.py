import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse
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

        batch_size, num_sensors, time_steps, embed_dim,  = X.shape
        X = X.reshape(batch_size, num_sensors, time_steps * embed_dim)

        graphs = [dense_to_sparse(A[b]) for b in range(batch_size)]
        edge_indices, edge_weights = zip(*graphs)

        batch = Batch.from_data_list(
            [Data(x=X[b], edge_index=edge_indices[b]) for b in range(batch_size)])

        for i, conv in enumerate(self.convs):
            batch.x = conv(batch.x, batch.edge_index)

            if i < self.num_layers - 1:
                batch.x = F.relu(batch.x)
                batch.x = self.dropout(batch.x)

        return batch.x.view(batch_size, num_sensors, -1)
