import torch
from torch import nn


class FusionAdjacencyMatrix(nn.Module):

    def __init__(self, cost_adj_matrix, similarity_adj_matrix):

        super(FusionAdjacencyMatrix, self).__init__()

        self.register_buffer("cost_adj_matrix", cost_adj_matrix.unsqueeze(0))
        self.register_buffer("similarity_adj_matrix", similarity_adj_matrix.unsqueeze(0))

        self.a = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.c = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, dynamic_adj_matrix):

        batch_size = dynamic_adj_matrix.shape[0]

        fusion_adj_matrix = (self.a * self.cost_adj_matrix.expand(batch_size, -1, -1) +
                             self.b * dynamic_adj_matrix +
                             self.c * self.similarity_adj_matrix.expand(batch_size, -1, -1))

        # (B, N, N)
        return fusion_adj_matrix
