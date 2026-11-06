import pytest
import torch
from unittest.mock import patch
from torch.utils.data import TensorDataset
from latent_learnability.data_sources.mnist import MNISTSampler


@pytest.fixture
def sampler():
    images = torch.randn(10, 1, 32, 32)
    labels = torch.arange(10, dtype=torch.int64) % 10

    dataset = TensorDataset(images, labels)

    with patch("latent_learnability.data_sources.mnist.datasets.MNIST", return_value=dataset):
        return MNISTSampler()


def test_sample_returns_correct_shapes_and_types(sampler):
    samples, labels = sampler.sample(4)

    assert samples.shape == (4, 1, 32, 32)
    assert labels.shape == (4,)
    assert samples.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_sample_returns_samples_without_replacement(sampler):
    samples, labels = sampler.sample(10)

    # Each dataset item has a unique label.
    assert len(torch.unique(labels)) == 10


def test_sample_rejects_num_samples_exceeding_dataset_size(sampler):
    with pytest.raises(ValueError, match="num_samples exceeds dataset size"):
        sampler.sample(11)