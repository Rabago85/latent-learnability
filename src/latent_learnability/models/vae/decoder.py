class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, upsample_channels: Optional[int] = None):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.res2 = ResidualBlock(in_channels)
        self.attn = AttnBlock(in_channels)
        if upsample_channels is not None:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=in_channels, out_channels=upsample_channels, kernel_size=3, padding=1, stride=1),
            )
        else:
            self.upsample = None

    def forward(self, x: torch.Tensor):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attn(x)
        if self.upsample:
            x = self.upsample(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: list[int]):
        super().__init__()

        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        blocks = []
        for in_c, out_c in zip(ch_in, ch_out):
            blocks.append(DecoderBlock(in_c, out_c))
        self.blocks = nn.ModuleList(blocks)

        x_dim = hidden_channels[-1]
        self.x_mean = nn.Sequential(
            nn.GroupNorm(1, x_dim),
            nn.Conv2d(in_channels=x_dim, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )

        self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)
        x = self.x_mean(x)
        return x, self.logvar