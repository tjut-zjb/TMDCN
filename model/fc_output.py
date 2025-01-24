import torch.nn as nn


class FCOutputLayer(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim):

        super(FCOutputLayer, self).__init__()

        self.norm_layer = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.ReLU())
            layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, X):

        X = self.norm_layer(X)
        X = self.fc_layers(X)

        return X
