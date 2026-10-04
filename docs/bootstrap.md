# Latent Learnability

Research investigating how structured mixture-of-Gaussians priors in variational autoencoders affect the learnability of the resulting latent space by flow-matching models.

> **Purpose of this document**
>
> This document is the bootstrap and recovery record for the repository. It records the exact process used to create the project, construct the local development environment, verify the installation, and establish the initial repository structure.
>
> If the local environment or repository needs to be rebuilt from scratch, this document should provide the source of truth.

---

# 1. Repository Creation

## Goal

Create the remote GitHub repository and establish the local Git repository.

## Commands

```bash

```

## Verification

```bash
git remote -v
```

## Notes

- 

---

# 2. Python and uv Setup

## Goal

Establish the Python version used by the project and configure `uv` to manage the environment and dependencies.

## Commands

```bash
uv --version
uv 0.5.20 (1c17662b3 2025-01-15)
uv python pin 3.11
uv run python --version
Python 3.11.9
```

## Verification

```bash

```

## Expected Result

```text

```

## Notes

- 

---

# 3. Initial Repository Structure

## Goal

Create the minimum package structure necessary to begin development.

## Commands

```bash

```

## Initial Structure

```text
mkdir -p configs/experiments
mkdir -p src/latent_learnability/models/vae/priors
mkdir -p tests

touch src/latent_learnability/__init__.py
touch src/latent_learnability/models/__init__.py
touch src/latent_learnability/models/vae/__init__.py
touch src/latent_learnability/models/vae/priors/__init__.py

touch README.md
touch NOTICE
```

## Verification

```bash
find . -maxdepth 5 -type f | sort
```

## Notes

- 

---

# 4. Project Configuration

## Goal

Create and configure `pyproject.toml`.

The project configuration defines:

- supported Python version
- runtime dependencies
- development dependencies
- package discovery
- build system
- pytest configuration
- Ruff configuration

## Commands / Changes

```bash
touch pyproject.toml
```

## Verification

```bash

```

## Notes

- 

---

# 5. Environment Creation

## Goal

Use `uv` to create the local virtual environment and resolve the project's dependencies.

## Commands

```bash
uv sync
```

## Files Created

```text

```

## Verification

```bash

```

## Notes

- 

---

# 6. Scientific Python Stack Verification

## Goal

Verify that the packages required by the original research notebook are available in the local environment.

This includes the core stack used by the VAE and flow-matching implementation.

## Commands

```bash
uv run python -c "
import torch
import torchvision
import numpy
import matplotlib
import seaborn
import einops
import yaml
from PIL import Image
from tqdm import tqdm

print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('numpy:', numpy.__version__)
print('All imports successful.')
"

uv run python -c "
import torch
import torch.nn as nn
import torch.distributions as D

from torch.nn import functional as F
from torch.func import vmap, jacrev
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from einops import rearrange
from einops.layers.torch import Rearrange

print('Advanced imports successful.')
"

uv run python -c "
import torch

print('PyTorch:', torch.__version__)
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
"

uv run python -c "
import latent_learnability

print(latent_learnability.__file__)
"
```

## Expected Result

```text

```

## Notes

- 

---

# 7. PyTorch Device Verification

## Goal

Verify the local PyTorch installation and identify available compute devices.

Local development does not require CUDA. GPU experiments are expected to run in Google Colab.

## Commands

```bash

```

## Results

```text

```

## Notes

- 

---

# 8. Python Package Verification

## Goal

Verify that the `src/` package layout is installed correctly and that `latent_learnability` can be imported without modifying `PYTHONPATH` or `sys.path`.

## Commands

```bash

```

## Expected Result

```text

```

## Notes

- 

---

# 9. Configuration Smoke Test

## Goal

Create the first YAML experiment configuration and verify that it can be loaded through the project's configuration module.

## Files Created

```text

```

## Commands

```bash

```

## Expected Result

```text

```

## Notes

- 

---

# 10. First Automated Test

## Goal

Verify that the project's test infrastructure works and that experiment configurations can be tested automatically.

## Files Created

```text

```

## Commands

```bash

```

## Results

```text

```

## Notes

- 

---

# 11. First PyTorch Component

## Goal

Create a minimal scientific component inside the package and verify that PyTorch code can be imported and executed through the package architecture.

## Files Created

```text

```

## Commands

```bash

```

## Notes

- 

---

# 12. Scientific Component Tests

## Goal

Test the first PyTorch component through the public package import path.

## Commands

```bash

```

## Results

```text

```

## Notes

- 

---

# 13. Ruff Verification

## Goal

Verify linting and formatting before committing the initial repository.

## Commands

```bash

```

## Results

```text

```

## Notes

- 

---

# 14. `.gitignore` Setup

## Goal

Prevent local environments, caches, generated results, model checkpoints, and other non-source artifacts from entering Git.

## Changes

```gitignore

```

## Verification

```bash

```

## Notes

- 

---

# 15. Pre-Commit Verification

## Goal

Run the complete local validation suite before creating the initial commit.

## Commands

```bash

```

## Results

```text

```

## Notes

- 

---

# 16. Initial Git Commit

## Goal

Commit the verified project skeleton and push it to GitHub.

## Commands

```bash

```

## Verification

```bash

```

## Commit

```text

```

## Notes

- 

---

# 17. Clean Environment Reproduction Test

## Goal

Prove that the local virtual environment is disposable.

Delete the environment and reconstruct it exclusively from the repository's committed configuration and lockfile.

This is the first major reproducibility test for the project.

## Remove Existing Environment

```bash

```

## Reconstruct Environment

```bash

```

## Verification

```bash

```

## Results

```text

```

## Notes

- 

---

# 18. Google Colab Reproduction

## Goal

Verify that a clean Google Colab environment can clone the repository, construct the project environment, import the package, execute tests, and access a CUDA GPU.

## Repository Setup

```bash

```

## Environment Setup

```bash

```

## Package Verification

```bash

```

## Test Suite

```bash

```

## CUDA Verification

```bash

```

## Results

```text

```

## Notes

- 

---

# 19. Standard Local Development Workflow

## Goal

Record the normal workflow used after the initial bootstrap is complete.

### Pull

```bash

```

### Develop

```bash

```

### Validate

```bash

```

### Commit

```bash

```

### Push

```bash

```

## Notes

- 

---

# 20. Standard Colab Execution Workflow

## Goal

Record the normal process for updating the GPU execution environment after changes have been pushed from the local development environment.

### Pull Repository Changes

```bash

```

### Synchronize Environment

```bash

```

### Verify

```bash

```

### Run Experiment

```bash

```

## Notes

- 

---

# 21. Full Disaster-Recovery Procedure

Use this section as the shortest possible path for rebuilding the project on a completely new machine.

## Prerequisites

```text

```

## Clone

```bash

```

## Install / Select Python

```bash

```

## Reconstruct Environment

```bash

```

## Verify Package

```bash

```

## Run Tests

```bash

```

## Verify Scientific Stack

```bash

```

## Expected Final State

```text

```

## Notes

- 

---

# 22. Current Repository Structure

Update this section whenever the high-level architecture materially changes.

```text

```

---

# 23. Environment Record

## Python

```text

```

## uv

```text

```

## PyTorch

```text

```

## torchvision

```text

```

## NumPy

```text

```

## Platform

```text

```

## GPU Execution Environment

```text

```

---

# 24. Reproducibility Principles

The project follows several basic rules:

1. `pyproject.toml` declares the project's required dependencies.
2. `uv.lock` records the exact resolved dependency environment and is committed to Git.
3. `.python-version` records the intended Python version and is committed to Git.
4. `.venv/` is disposable and is never committed.
5. Research code lives in `src/latent_learnability/`.
6. Tests import `latent_learnability` through its installed package interface.
7. Experiment behavior is controlled by committed YAML configurations.
8. Notebooks are exploratory tools and are not required to reproduce reported results.
9. Generated datasets, checkpoints, and large experiment outputs are not treated as source code.
10. A clean environment must be reconstructable from the repository.
11. Local development and Colab execution should run the same package code.
12. Results reported by the research should ultimately be traceable to an experiment configuration and Git commit.

---

# 25. Attribution Record

This project builds upon educational code and course materials from MIT 6.S184/6.S975, *Introduction to Flow Matching and Diffusion Models* (2026).

The baseline VAE, Diffusion Transformer, and flow-matching implementations used by this research originate in or are adapted from those materials.

The research developed in this repository extends that baseline through work including:

- structured mixture-of-Gaussians latent priors
- modifications to the training objective
- new evaluation methodology
- latent-space analysis
- flow-matching experiments
- empirical comparison of latent representations

Detailed source-level attribution and licensing information is maintained in `NOTICE`.

---

# 26. Recovery Log

Use this section whenever the environment actually has to be rebuilt or repaired.

## Recovery — YYYY-MM-DD

### Reason

```text

```

### Starting State

```text

```

### Actions

```bash

```

### Problems Encountered

```text

```

### Resolution

```text

```

### Final Verification

```bash

```

### Lessons / Changes to This Runbook

- 
