import torch
import torch.distributions as D


class GaussianPrior:
    """Standard isotropic Gaussian latent prior."""

    def __init__(self, latent_channels: int) -> None:
        self.latent_channels = latent_channels

    def distribution(self, device: torch.device | None = None) -> D.Normal:
        mean = torch.zeros(self.latent_channels, device=device)
        std = torch.ones(self.latent_channels, device=device)

        return D.Normal(mean, std)

    def sample(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        distribution = self.distribution(device=device)

        return distribution.sample((batch_size,))