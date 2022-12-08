from torch import nn


class Embed(nn.Module):
    def __init__(self, in_channels, dim, patch_size, padding):
        self.embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=padding
        )
    
    # def forward(self, ):