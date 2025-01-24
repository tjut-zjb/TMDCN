from torch import nn


class MultiScaleConvTemporalModel(nn.Module):

    def __init__(self, embed_dim, kernel_sizes):

        super(MultiScaleConvTemporalModel, self).__init__()

        self.convs = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(embed_dim)

        for i in range(len(kernel_sizes)):
            self.convs.append(
                nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim,
                          kernel_size=(kernel_sizes[i], 1), padding=(kernel_sizes[i] // 2, 0))
            )

    def forward(self, X):

        conv_out = 0
        for conv in self.convs:
            conv_out = conv(X.permute(0, 3, 2, 1))

        output = self.layer_norm(conv_out.permute(0, 3, 2, 1) + X)
        # (B, N, T, embed_dim)
        return output
