from latent_learnability.config import load_config


def test_load_config() -> None:
    config = load_config("configs/experiments/smoke_test.yaml")

    assert config["experiment"]["name"] == "smoke_test"
    assert config["experiment"]["seed"] == 42
    assert config["dataset"]["name"] == "mnist"
    assert config["prior"]["type"] == "gaussian"
    assert config["vae"]["latent_channels"] == 128