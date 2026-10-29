import torch
import numpy as np

def log_mean_exp(input: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(input, dim) - np.log(input.size(dim))


def log_normal(input: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    log_prob = (torch.distributions
                .Normal(loc=means, scale=torch.sqrt(variances))
                .log_prob(input)
                .sum(-1))
    return log_prob

def log_normal_mixture(input: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    input = input.unsqueeze(1)
    log_prob_mixtures = log_normal(input, means, variances)
    log_prob = log_mean_exp(log_prob_mixtures, dim=1)
    return log_prob

def flatten(input: torch.Tensor) -> torch.Tensor:
    bs, c, h, w = input.shape
    flattened_input = input.permute(0, 2, 3, 1).reshape(bs * h * w, c)
    return flattened_input


def unflatten(input: torch.Tensor, bs: int):
    spatial_dim_size = input.shape[0] // bs
    unflattened_input = input.reshape(-1, spatial_dim_size)
    return unflattened_input

def raw_kl_unit(z, z_mean, z_logvar, z_prior_mean, z_prior_logvar):
        kl = 0.5 * (z_mean.pow(2) + torch.exp(z_logvar) - z_logvar - 1)
        return kl.sum(dim=1).mean()


def raw_kl_mog(z: torch.Tensor, 
               z_mean: torch.Tensor, 
               z_logvar: torch.Tensor, 
               z_prior_mean: torch.Tensor, 
               z_prior_logvar: torch.Tensor):
    bs = z.shape[0]
    dim = z.shape[1]
    z_flattened = flatten(z)
    z_mean_flattened = flatten(z_mean)
    ln = log_normal(z_flattened, z_mean_flattened, torch.exp(z_logvar))
    lnm = log_normal_mixture(z_flattened, z_prior_mean, torch.exp(z_prior_logvar))
    ln_unflattened = unflatten(ln, bs)
    lnm_unflattened = unflatten(lnm, bs)
    return (ln_unflattened - lnm_unflattened).sum(-1).mean()
