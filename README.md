# Asymmetric Auto-Encoder Bottlenecks for Self-Supervised Latent Representation Learning

A controlled study of whether an asymmetric autoencoder (heavy nonlinear encoder, tight latent
bottleneck, lightweight multiscale decoder) learns more compact and noise-robust representations
than a conventional symmetric convolutional autoencoder under matched training conditions.

> **Status: Phase 1 (project foundation).** No models, training code or experiments exist yet.
> The architecture and hypotheses are *proposals*; see
> [`docs/phase_reports/phase_00_research_blueprint.md`](docs/phase_reports/phase_00_research_blueprint.md).

## Setup

Requires Python >= 3.10 (developed on 3.11).

```bash
git clone https://github.com/aman-18-choudhary/Asymmetric_Auto_Encoder.git
cd Asymmetric_Auto_Encoder
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Development

```bash
pytest              # run tests
ruff check .        # lint
ruff format --check .   # formatting check
```

Configuration lives in `configs/` (YAML, validated by `asymmetric_autoencoder.config`; unknown
keys are rejected). Seeding and device selection: `asymmetric_autoencoder.utils.reproducibility`.
Logging: `asymmetric_autoencoder.utils.logging`.

## Git workflow

`main` (stable) <- `develop` (integration) <- `feature/phase-N-*`. Conventional Commits
(`feat:`, `fix:`, `test:`, `docs:`, `chore:`, `exp:`). Datasets and checkpoints are never committed.

## Reproducibility notes

Seeds are set for Python, NumPy and PyTorch. Some MPS/CUDA kernels are non-deterministic;
`deterministic: true` requests deterministic algorithms and may raise for unsupported ops.
