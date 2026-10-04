from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment configuration from YAML."""
    path = Path(path)

    with path.open("r") as file:
        config = yaml.safe_load(file)

    return config