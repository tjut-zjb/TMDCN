import torch
import torch.nn as nn


class GatedFusion(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128):

        super(GatedFusion, self).__init__()

        self.fc_TA = nn.Linear(input_dim, output_dim, bias=True)
        self.fc_MC = nn.Linear(input_dim, output_dim, bias=False)

        self.fc_output = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feature_matrix_TA, feature_matrix_MC):

        feature_matrix_TA = self.fc_TA(feature_matrix_TA)
        feature_matrix_MC = self.fc_MC(feature_matrix_MC)

        gate = torch.sigmoid(feature_matrix_TA + feature_matrix_MC)
        output = gate * feature_matrix_TA + (1 - gate) * feature_matrix_MC

        output = self.fc_output(output)

        return output
