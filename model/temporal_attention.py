import torch
import torch.nn as nn
import math


class TemporalAttentionLayer(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout):

        super(TemporalAttentionLayer, self).__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, X):

        batch_size, num_sensors, time_steps, embed_dim = X.shape

        # (B, N, T, embed_dim)
        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        # (B, N, T, T)
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(embed_dim)
        attention_weights = self.attention_dropout(self.softmax(attention_scores))

        # (B, N, T, embed_dim)
        attention_output = torch.matmul(attention_weights, V)

        X = self.layer_norm1(attention_output + X)

        feed_forward_output = self.feed_forward(X)
        X = self.layer_norm2(feed_forward_output + X)

        # (B, N, T, embed_dim)
        return X


class TemporalAttention(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dropout, num_layers):

        super(TemporalAttention, self).__init__()

        self.layers = nn.ModuleList([
            TemporalAttentionLayer(embed_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X):

        for layer in self.layers:
            X = layer(X)

        return X
