import torch

from latent_learnability.models.vae.priors.gaussian import GaussianPrior


def test_gaussian_prior_sample_shape() -> None:
    prior = GaussianPrior(latent_channels=128)

    samples = prior.sample(batch_size=16)

    assert samples.shape == (16, 128)


def test_gaussian_prior_samples_are_finite() -> None:
    prior = GaussianPrior(latent_channels=128)

    samples = prior.sample(batch_size=16)

    assert torch.isfinite(samples).all()