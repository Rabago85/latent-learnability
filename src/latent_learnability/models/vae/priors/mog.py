import torch
from torch import nn
import numpy as np
import math

class MOGPrior(nn.Module):
    def __init__(self, k: int, dim: int):
        super().__init__()

        assert dim % 2 == 0
        self.k = k
        self.dim = dim
        self.half_dim = dim // 2

        self.weights = nn.Parameter(torch.randn(k, self.half_dim))

        self.register_buffer("pi", torch.ones(k) / k)

        self.raw_magnitudes = nn.Parameter(
            torch.full((k, ), math.log(math.e - 1)) # + 0.01 * torch.randn(k)  # softplus(this) centers near 1
        )

        self.log_var = nn.Parameter(
            torch.full((k, dim), math.log(2.0 / (9.0 * dim)))
        )

    @property
    def magnitudes(self) -> torch.Tensor:
        return nn.functional.softplus(self.raw_magnitudes)

    @property
    def unit_circle(self) -> torch.Tensor:
        f_i = 2 * torch.pi * self.weights
        unit_circle = torch.cat(
            [torch.cos(f_i), torch.sin(f_i)], dim=1
        ) / np.sqrt(self.half_dim)
        return unit_circle

    @property
    def means(self) -> torch.Tensor:
        return self.unit_circle * self.magnitudes.view(-1, 1)
