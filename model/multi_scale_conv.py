from torch import nn


class MultiScaleConvTemporalModel(nn.Module):

    def __init__(self, embed_dim, kernel_sizes):

        super(MultiScaleConvTemporalModel, self).__init__()

        self.convs = nn.ModuleList()

        for i in range(len(kernel_sizes)):
            self.convs.append(
                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                          kernel_size=(kernel_sizes[i], 1), padding=(kernel_sizes[i] // 2, 0))
            )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, X):

        # (B, embed_dim, T, N)
        out = X.permute(0, 3, 2, 1)

        for conv in self.convs:
            out = conv(out)

        # (B, N, T, embed_dim)
        out = out.permute(0, 3, 2, 1)
        output = self.layer_norm(out + X)

        return output
