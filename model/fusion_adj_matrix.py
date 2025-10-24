import torch
from torch import nn


class FusionAdjacencyMatrix(nn.Module):

    def __init__(self, cost_adj_matrix: torch.Tensor, similarity_adj_matrix: torch.Tensor):

        super().__init__()

        self.num_nodes = cost_adj_matrix.shape[0]

        self.cost_linear = nn.Linear(self.num_nodes, self.num_nodes)
        self.similarity_linear = nn.Linear(self.num_nodes, self.num_nodes)

        self.register_buffer("cost_adj", cost_adj_matrix.unsqueeze(0))
        self.register_buffer("similar_adj", similarity_adj_matrix.unsqueeze(0))

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Controls sharpness of the sigmoid (smaller = more binary output)
        self.temperature = nn.Parameter(torch.tensor(0.01))
        # Filters out weak connections
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, dynamic_adj):

        batch_size = dynamic_adj.shape[0]

        cost_adj = self.cost_linear(self.cost_adj)
        cost_adj = cost_adj.expand(batch_size, -1, -1)

        similar_adj = self.similarity_linear(self.similar_adj)
        similar_adj = similar_adj.expand(batch_size, -1, -1)

        combined = torch.stack([cost_adj, dynamic_adj, similar_adj], dim=1)

        weights = self.attention(combined)

        # Fuse matrices using attention weights → Shape: [B, N, N]
        fusion_adj = torch.sum(weights * combined, dim=1)
        fusion_adj = (fusion_adj + fusion_adj.transpose(1, 2)) / 2
        fusion_adj = torch.sigmoid((fusion_adj - self.threshold) * self.temperature)

        return fusion_adj
