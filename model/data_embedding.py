import math
import torch
from torch import nn


class TokenEmbedding(nn.Module):

    def __init__(self, input_dim, embed_dim):

        super().__init__()

        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        x = self.token_embed(x)
        x = self.layer_norm(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=100):

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        # different offsets for each set of 12 time steps
        offsets = [2016, 288, 12]

        for i in range(3):
            pe[i * 12:(i + 1) * 12, 0::2] = torch.sin((position[i * 12:(i + 1) * 12] + offsets[i]) * div_term)
            pe[i * 12:(i + 1) * 12, 1::2] = torch.cos((position[i * 12:(i + 1) * 12] + offsets[i]) * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class DataEmbedding(nn.Module):

    def __init__(self, input_dim, embed_dim, dropout):

        super().__init__()

        self.embedding = TokenEmbedding(input_dim, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):

        # (B, N, T, F)
        X = X.permute(0, 1, 3, 2)

        # (B, N, T, embed_dim)
        X = self.embedding(X)

        # (B, T, N, embed_dim)
        X = X.permute(0, 2, 1, 3)

        X = X + self.position_encoding(X)

        X = self.dropout(X)

        # (B, N, T, embed_dim)
        return X.permute(0, 2, 1, 3)
