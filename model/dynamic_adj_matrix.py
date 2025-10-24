import torch
import torch.nn.functional as F
from torch import nn


class DynamicAdjMatrix(nn.Module):

    def __init__(self, out_channels):

        super(DynamicAdjMatrix, self).__init__()

        self.embedding = nn.Conv2d(1, out_channels, kernel_size=1)

    def forward(self, X_flow):

        batch_size, num_sensors, time_steps = X_flow.shape

        # (B, 1, T, N)
        X_flow = X_flow.permute(0, 2, 1).unsqueeze(1)

        # (B, C, T, N)
        phi_X = self.embedding(X_flow)

        # (B, C * T, N)
        phi_X = phi_X.reshape(batch_size, -1, num_sensors)

        # (B, N, N)
        A_d = torch.bmm(phi_X.transpose(1, 2), phi_X)

        # matrix symmetry
        A_d = (A_d + A_d.transpose(1, 2)) / 2

        A_d = F.relu(A_d)
        A_d = F.softmax(A_d, dim=-1)

        return A_d
