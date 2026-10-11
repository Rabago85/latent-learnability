import pytest
import torch

from latent_learnability.models.vae.priors.probability import (
    raw_kl_unit,
    raw_kl_mog,
    log_normal,
    log_normal_mixture,
    flatten,
)


@pytest.fixture
def data():
    torch.manual_seed(42)

    bs, c, h, w = 16, 128, 4, 4

    z_mean = torch.randn(bs, c, h, w)
    z_logvar = torch.randn(1) * 0.1
    z = z_mean + torch.exp(0.5 * z_logvar) * torch.randn_like(z_mean)

    return {
        "z": z,
        "mu": z_mean,
        "logvar": z_logvar,
        "bs": bs,
        "c": c,
        "h": h,
        "w": w,
    }


def unit_prior(c):
    return torch.zeros(1, c), torch.zeros(1, c)


def mog_kl(data, prior_mean, prior_logvar):
    """Return KL per spatial grid point, matching raw_kl_unit."""
    return raw_kl_mog(
        data["z"],
        data["mu"],
        data["logvar"],
        prior_mean,
        prior_logvar,
    ) / (data["h"] * data["w"])


def analytical_kl(data):
    return raw_kl_unit(
        data["z"],
        data["mu"],
        data["logvar"],
        torch.zeros(1, data["c"]),
        torch.zeros(1, data["c"]),
    )


# -------------------------------------------------------
# 1. K=1: statistical equivalence to analytical KL
# -------------------------------------------------------

def test_k1_matches_analytical_kl_with_mc_tolerance(data):
    c = data["c"]
    prior_mean, prior_logvar = unit_prior(c)

    kl_exact = analytical_kl(data)
    kl_mc = mog_kl(data, prior_mean, prior_logvar)

    assert kl_exact.ndim == 0
    assert kl_mc.ndim == 0
    assert torch.isfinite(kl_exact)
    assert torch.isfinite(kl_mc)

    # Estimate the Monte Carlo standard error using the
    # per-grid residuals relative to the analytical KL.
    z_flat = flatten(data["z"])
    mu_flat = flatten(data["mu"])

    log_q = log_normal(
        z_flat, mu_flat, torch.exp(data["logvar"])
    )
    log_p = log_normal_mixture(
        z_flat, prior_mean, torch.exp(prior_logvar)
    )

    pointwise_exact = 0.5 * (
        data["mu"].pow(2)
        + torch.exp(data["logvar"])
        - data["logvar"]
        - 1
    ).sum(dim=1)

    residual = (
        (log_q - log_p).reshape(data["bs"], -1)
        - pointwise_exact.reshape(data["bs"], -1)
    ).reshape(-1)

    standard_error = (
        residual.std(unbiased=True)
        / residual.numel() ** 0.5
    )

    # Five standard errors gives a generous statistical bound.
    tolerance = 5 * standard_error + 1e-5

    assert (kl_mc - kl_exact).abs() <= tolerance


def test_k1_analytical_kl_matches_closed_form(data):
    expected = 0.5 * (
        data["mu"].pow(2)
        + torch.exp(data["logvar"])
        - data["logvar"]
        - 1
    ).sum(dim=1).mean()

    torch.testing.assert_close(
        analytical_kl(data),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


# -------------------------------------------------------
# 2. K>1: identical components must reproduce K=1
# -------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_identical_components_match_k1(data, k):
    prior_mean, prior_logvar = unit_prior(data["c"])

    mixture_mean = prior_mean.repeat(k, 1)
    mixture_logvar = prior_logvar.repeat(k, 1)

    kl_single = mog_kl(data, prior_mean, prior_logvar)
    kl_mixture = mog_kl(data, mixture_mean, mixture_logvar)

    torch.testing.assert_close(
        kl_mixture,
        kl_single,
        rtol=1e-5,
        atol=1e-5,
    )


# -------------------------------------------------------
# 3. Mixture density is invariant to component ordering
# -------------------------------------------------------

def test_mixture_density_permutation_invariance():
    torch.manual_seed(123)

    z = torch.randn(32, 8)
    means = torch.randn(3, 8)
    variances = torch.exp(torch.randn(3, 8) * 0.2)

    original = log_normal_mixture(z, means, variances)

    perm = torch.tensor([2, 0, 1])
    permuted = log_normal_mixture(
        z, means[perm], variances[perm]
    )

    torch.testing.assert_close(
        original, permuted, rtol=1e-5, atol=1e-6
    )


# -------------------------------------------------------
# 4. K>1: distinct components produce finite KL
# -------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 5])
def test_distinct_components_produce_finite_kl(data, k):
    torch.manual_seed(123)

    prior_mean = torch.randn(k, data["c"])
    prior_logvar = torch.randn(k, data["c"]) * 0.2

    kl = mog_kl(data, prior_mean, prior_logvar)

    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_different_prior_changes_kl(data):
    prior_mean, prior_logvar = unit_prior(data["c"])

    kl_standard = mog_kl(data, prior_mean, prior_logvar)

    # Two well-separated Gaussian components.
    mixture_mean = torch.stack([
        torch.full((data["c"],), -2.0),
        torch.full((data["c"],), 2.0),
    ])
    mixture_logvar = torch.zeros_like(mixture_mean)

    kl_mixture = mog_kl(data, mixture_mean, mixture_logvar)

    assert torch.isfinite(kl_mixture)
    assert not torch.isclose(
        kl_standard, kl_mixture, rtol=1e-3, atol=1e-3
    )


# -------------------------------------------------------
# 5. Output shape and numerical stability
# -------------------------------------------------------

def test_kl_outputs_are_finite_scalars(data):
    prior_mean, prior_logvar = unit_prior(data["c"])

    kl_unit = analytical_kl(data)
    kl_mixture = raw_kl_mog(
        data["z"],
        data["mu"],
        data["logvar"],
        prior_mean,
        prior_logvar,
    )

    assert kl_unit.ndim == 0
    assert kl_mixture.ndim == 0
    assert torch.isfinite(kl_unit)
    assert torch.isfinite(kl_mixture)


# -------------------------------------------------------
# 6. Analytical KL does not depend on sampled z
# -------------------------------------------------------

def test_unit_kl_independent_of_z(data):
    original = analytical_kl(data)

    altered = raw_kl_unit(
        torch.randn_like(data["z"]),
        data["mu"],
        data["logvar"],
        torch.zeros(1, data["c"]),
        torch.zeros(1, data["c"]),
    )

    torch.testing.assert_close(original, altered)