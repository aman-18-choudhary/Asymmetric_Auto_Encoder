# Phase 1: Project Foundation

## A. Phase Summary
Establish the Python package, configuration, reproducibility and logging utilities, test/lint tooling and Git branching. No models, training or experiments (out of scope by design).

## B. Objectives
Package structure; packaging and dependencies; config management; reproducibility utilities; logging; pytest + ruff; `.gitignore`; README; branch structure; lightweight tests.

## C. Work Completed
- Cloned the empty GitHub repo (no new `git init`, no nested `.git`); remote and branch verified before changes.
- `pyproject.toml` (setuptools, `src/` layout, extras `dev`), `.gitignore`.
- `config.py`: dataclass config loaded from YAML, rejects unknown keys and invalid values, round-trips to YAML.
- `utils/reproducibility.py`: `seed_everything`, `seed_worker`, `make_generator`, `resolve_device`, `environment_info`.
- `utils/logging.py`: idempotent `setup_logging`, namespaced `get_logger`.
- `configs/base.yaml`, placeholder subpackages (`data`, `models`, `training`, `evaluation`) containing only docstrings.
- README with setup/dev instructions; approved Phase 0 report copied unmodified.

## D. Architecture and Technical Decisions
- Python 3.11 venv (`.venv`, ignored); `requires-python >=3.10`.
- Minimal runtime deps (torch, torchvision, numpy, pyyaml); sklearn/matplotlib deferred until needed.
- Unknown config keys are errors so typos cannot silently change an experiment.
- Explicit device requests never fall back silently; `auto` prefers CUDA, then MPS, then CPU.
- No LICENSE added: the choice is yours (see K).

## E. Files Added
`.gitignore`, `README.md`, `pyproject.toml`, `configs/base.yaml`, `data/.gitkeep`, `src/asymmetric_autoencoder/{__init__,config}.py`, `src/asymmetric_autoencoder/utils/{reproducibility,logging}.py`, subpackage `__init__.py` files, `tests/test_{config,reproducibility,logging,import}.py`, `docs/phase_reports/phase_00_...md` (copied, unmodified), this report.

## F. Commands Executed
```
git clone https://github.com/aman-18-choudhary/Asymmetric_Auto_Encoder.git
/opt/homebrew/bin/python3.11 -m venv .venv
pip install --dry-run torch torchvision numpy pyyaml pytest ruff   # compatibility check
pip install -e ".[dev]"; pip check
pytest -q; ruff check .; ruff format --check .
# fresh-venv install: python3.11 -m venv <tmp>/v && <tmp>/v/bin/pip install . ; import from outside the repo
git push -u origin main   (develop and feature branch pushed afterwards, see I)
```

## G. Verification Results
- `pytest`: 25 passed (1.6 s). An initial run had 1 failing test (my test's own import-scoping bug, `UnboundLocalError`); fixed in the test and rerun.
- `ruff check`: all checks passed; `ruff format --check`: all files formatted.
- `pip check`: no broken requirements.
- Fresh isolated venv: `pip install .` succeeded and `import asymmetric_autoencoder` worked from outside the repo (v0.1.0).
- Environment: Python 3.11.14, torch 2.14.1, numpy 2.4.6; MPS available, CUDA not.
- Secret scan (grep for key/token/password/private-key/absolute-home-path patterns) over added files: no matches. Tracked files contain no `.venv`, data or checkpoints.
- Forward-pass and tensor-shape checks: not applicable (no models yet).

## H. Experimental Results
None.

## I. GitHub Activity
- Auth: HTTPS with macOS keychain helper; no SSH key present; `gh` not installed (not required).
- `main` pushed (verified by push output): `6f2a8e9 chore: add .gitignore for Python, data and artifacts`.
- Feature branch commits (hashes from local log):
  - `d897fb0 feat: add package skeleton, config loading, seeding and logging utilities`
  - `01b5642 chore: add base experiment config and data directory placeholder`
  - `54164c2 test: add config, reproducibility, logging and import tests`
  - `583bf9c docs: add README setup instructions and approved Phase 0 blueprint`
- This report is committed afterwards on the feature branch; final push and remote verification of `develop` and `feature/phase-1-foundation` are stated in the completion message, not here.
- No force-push, no history rewrite. No pull request opened (no `gh`; open via web UI).

## J. Issues Encountered
- Installing torch exceeded the tool timeout; resolved by waiting for completion.
- One test bug (above), fixed.

## K. Outstanding Issues
- LICENSE not chosen (needs your decision).
- CI workflow (`.github/workflows/tests.yml`) not added; CI installing torch is heavy, deferred.
- PR for `feature/phase-1-foundation` -> `develop` must be opened/merged by you or after `gh` is installed.
- `main` contains only `.gitignore` until Phase 1 is merged.
- No `requirements.txt` lockfile; versions float within `pyproject` bounds.
- Python 3.10 and 3.12 untested (only 3.11).

## L. Acceptance Criteria
| Criterion | Status |
|---|---|
| Installs in a clean environment | PASS |
| Imports work | PASS |
| Basic tests pass | PASS |
| Configuration can be loaded | PASS |
| Random seeds configurable | PASS |
| Repository clean and documented | PARTIAL (no LICENSE, no CI, PR not merged) |

## M. Next Phase
Phase 2: dataset pipeline (MNIST/CIFAR-10, fixed split) and the baseline conv autoencoder with a tiny-batch overfit test. Not started.
