import torch
from latent_learnability.models.vae.priors.mog import MOGPrior
import math
import pytest

@pytest.fixture
def prior():
    torch.manual_seed(42)
    return MOGPrior(k=4, dim=8)


# 1. Shapes and initialization

def test_parameter_shapes(prior):
    assert prior.weights.shape == (4, 4)
    assert prior.raw_magnitudes.shape == (4,)
    assert prior.magnitudes.shape == (4,)
    assert prior.unit_circle.shape == (4, 8)
    assert prior.means.shape == (4, 8)
    assert prior.log_var.shape == (4, 8)
    assert prior.pi.shape == (4,)


def test_initial_magnitudes_are_one(prior):
    torch.testing.assert_close(
        prior.magnitudes,
        torch.ones(4),
    )


def test_initial_mixture_probabilities(prior):
    torch.testing.assert_close(
        prior.pi,
        torch.full((4,), 1 / 4),
    )
    torch.testing.assert_close(prior.pi.sum(), torch.tensor(1.0))


def test_initial_log_variance(prior):
    expected = math.log(2.0 / (9.0 * prior.dim))

    torch.testing.assert_close(
        prior.log_var,
        torch.full_like(prior.log_var, expected),
    )


# 2. Geometry

def test_unit_circle_vectors_have_unit_norm(prior):
    norms = torch.linalg.vector_norm(prior.unit_circle, dim=1)

    torch.testing.assert_close(
        norms,
        torch.ones(prior.k),
    )


def test_mean_norms_equal_magnitudes(prior):
    norms = torch.linalg.vector_norm(prior.means, dim=1)

    torch.testing.assert_close(norms, prior.magnitudes)


def test_unit_circle_uses_polar_parameterization(prior):
    angles = 2 * torch.pi * prior.weights
    scale = math.sqrt(prior.half_dim)

    expected = torch.cat(
        [torch.cos(angles), torch.sin(angles)],
        dim=1,
    ) / scale

    torch.testing.assert_close(prior.unit_circle, expected)


# 3. Constraints and parameter registration

def test_magnitudes_are_positive(prior):
    assert torch.all(prior.magnitudes > 0)


def test_pi_is_a_buffer_not_a_parameter(prior):
    assert "pi" in dict(prior.named_buffers())
    assert "pi" not in dict(prior.named_parameters())


def test_learnable_parameters_are_registered(prior):
    parameters = dict(prior.named_parameters())

    assert "weights" in parameters
    assert "raw_magnitudes" in parameters
    assert "log_var" in parameters


# 4. Gradients and optimization

def test_parameters_receive_gradients(prior):
    loss = (
        prior.means.square().sum()
        + prior.log_var.square().sum()
    )
    loss.backward()

    for parameter in (
        prior.weights,
        prior.raw_magnitudes,
        prior.log_var,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_optimizer_updates_means(prior):
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-2)

    old_weights = prior.weights.detach().clone()

    optimizer.zero_grad()
    loss = prior.means[:, 0].sum()    
    loss.backward()
    optimizer.step()

    assert prior.weights.grad is not None
    assert torch.isfinite(prior.weights.grad).all()
    assert prior.weights.grad.abs().sum() > 0
    assert not torch.equal(prior.weights, old_weights)


def test_optimizer_updates_log_var(prior):
    optimizer = torch.optim.Adam(prior.parameters(), lr=1e-2)

    old_log_var = prior.log_var.detach().clone()

    optimizer.zero_grad()
    loss = prior.log_var.square().sum()
    loss.backward()
    optimizer.step()

    assert prior.log_var.grad is not None
    assert torch.isfinite(prior.log_var.grad).all()
    assert prior.log_var.grad.abs().sum() > 0
    assert not torch.equal(prior.log_var, old_log_var)


# 5. Input validation

def test_odd_dimension_is_rejected():
    with pytest.raises(AssertionError):
        MOGPrior(k=3, dim=7)

# Tests to add later
# Once you integrate GMPrior into the VAE, I'd add just three more tests:
# Variance validity: torch.exp(prior.log_var) is finite and strictly positive.
# State persistence: saving and loading the module's state_dict reproduces its means, 
# magnitudes, mixture probabilities, and log variances.
# KL integration: for k=1, compare the prior's mean and variance against your standard-normal KL reference; 
# for k>1, verify the mixture KL is finite and gradients propagate through the prior parameters.