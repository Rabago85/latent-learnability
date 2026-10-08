latent-learnability/
│
├── README.md
├── LICENSE
├── NOTICE                         # attribution for MIT course code
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── configs/
│   ├── experiments/              # one YAML = one experimental matrix cell
│   │   ├── mnist/
│   │   │   ├── gaussian_elbo.yaml
│   │   │   └── mog_elbo.yaml
│   │   ├── cifar10/
│   │   └── stl10/
│   │
│   └── sweeps/
│       └── main_matrix.yaml      # experiments comprising primary matrix
│
├── src/
│   └── latent_learnability/
│       ├── __init__.py
│       ├── cli.py                # command-line entry point
│       ├── config.py             # config schema + YAML loading
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── mnist.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   │
│       │   ├── vae/
│       │   │   ├── __init__.py
│       │   │   ├── model.py
│       │   │   ├── encoder.py
│       │   │   ├── decoder.py
│       │   │   │
│       │   │   └── priors/
│       │   │       ├── __init__.py
│       │   │       ├── gaussian.py
│       │   │       └── mog.py
│       │   │
│       │   └── flow/
│       │       ├── __init__.py
│       │       ├── model.py
│       │       └── dit.py
│       │
│       ├── objectives/
│       │   ├── __init__.py
│       │   ├── elbo.py
│       │   └── regularization.py
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── vae.py
│       │   └── flow.py
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── latent_structure.py
│       │   ├── likelihood.py
│       │   └── generation.py
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluate.py
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── latent_space.py
│       │   ├── trajectories.py
│       │   └── samples.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── checkpoint.py
│           └── seed.py
│
├── scripts/
│   └── run_matrix.py             # simple sequential matrix runner
│
├── experiments/
│   └── README.md                 # hypotheses + experiment index
│
├── results/                      # generated experimental output
│   ├── metrics/
│   ├── figures/
│   └── samples/
│
├── notebooks/
│   └── exploratory/              # exploration only; not reproduction path
│
├── tests/
│   ├── test_config.py
│   ├── test_priors.py
│   ├── test_objectives.py
│   └── test_metrics.py
│
└── docs/
    └── research/
        ├── 00_abstract.md
        ├── 01_background.md
        ├── 02_methodology.md
        ├── 03_math_appendix.md
        ├── 04_results.md
        ├── 05_discussion.md
        └── references.bib