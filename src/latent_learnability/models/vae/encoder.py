



class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, downsample_channels: Optional[int] = None):
        super().__init__()
        self.downsample_channels = downsample_channels
        self.resid_1 = ResidualBlock(in_channels)
        self.resid_2 = ResidualBlock(in_channels)
        self.attention = AttnBlock(in_channels)
        if downsample_channels:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=downsample_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor):
        x = self.resid_1(x)
        x = self.resid_2(x)
        x = self.attention(x)
        return x if not self.downsample_channels else self.downsample(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list[int]):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=3, padding=1, stride=1)

        ch_in = hidden_channels
        ch_out = hidden_channels[1:] + [None]
        blocks = []
        for in_c, out_c in zip(ch_in, ch_out):
            blocks.append(EncoderBlock(in_c, out_c))
        self.blocks = nn.ModuleList(blocks)

        z_dim = hidden_channels[-1]
        self.z_mean = nn.Sequential(
            nn.GroupNorm(1, z_dim),
            nn.Conv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=1, stride=1, padding=0),
        )
        # self.z_mean = nn.Conv2d(in_channels=z_dim, out_channels=z_dim, kernel_size=1, stride=1, padding=0)
        # Weight scale chosen so ||z_mean|| ~ 1 regardless of z_dim (see README)
        nn.init.normal_(
            self.z_mean[1].weight,
            mean=0.0,
            std=1.0 / z_dim,
        )
        nn.init.zeros_(self.z_mean[1].bias)
        # nn.init.normal_(
        #     self.z_mean.weight,
        #     mean=0.0,
        #     std=1.0 / z_dim,
        # )
        # nn.init.zeros_(self.z_mean.bias)

        # # logvar init matched to GMPrior's log_var target (2/(9*z_dim)) -
        # # previously zero-init, which caused a severe scale mismatch (see README)
        self.logvar = nn.Parameter(torch.tensor(math.log(2.0 / (9.0 * z_dim))))
        #self.logvar = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor):
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.z_mean(x)
        return x, self.logvar