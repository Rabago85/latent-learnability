import torch
import numpy as np

def log_mean_exp(x, dim):
    return log_sum_exp(x, dim) - np.log(x.size(dim))


def log_sum_exp(x, dim=0):
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def log_normal(x, m, v):
    log_prob = (torch.distributions
                .Normal(loc=m, scale=torch.sqrt(v))
                .log_prob(x)
                .sum(-1))
    return log_prob


def log_normal_mixture(z, m, v):
    z = z.unsqueeze(1)
    log_prob = log_normal(z, m, v)
    log_prob = log_mean_exp(log_prob, dim=1)
    return log_prob

