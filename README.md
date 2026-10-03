# physics-constrained-neural-operators

Code for *Only Linear Constraints Survive Coarse-Graining: Evaluating Physics-Constrained
Neural Operators on Stochastically Forced Turbulence* (Michael Groom and Rafael Oliveira, CSIRO),
NeurIPS 2026 Workshop on AI for Stochastic Dynamics. A Fourier neural operator (FNO) is trained
on two flows, resolved Kolmogorov flow (E1) and stochastically forced 2D turbulence truncated to
64² (E2), with the linear constraints (continuity, global momentum balance) enforced exactly by a
spectral projection on the output, with and without a projected global energy balance, and
compared against the PINO residual loss with and without test-time optimisation. The paper's E2 is
named `e3` in this code and in the dataset.

## Layout

- `experiments/` — training (`train_operator_AR_PINO_2d.py`), rollout evaluation
  (`test_operator_AR_2d.py`), the shared run protocol (`run_protocol.py`), seed aggregation
  (`aggregate_seeds.py`), free-running stability (`rollout_stability.py`, `stability_table.py`),
  the energy-closure fit (`fit_subgrid_source.py`), the step-448 spectra (`sixwindow_spectra.py`),
  data preparation (`data_utils/`) and the configurations of every reported run
  (`configs/e1_noise/` for E1, `configs/e3_final/` for E2).
- `models/` — the FNO (`fno.py`) and the constraint layers (`layers.py`).
- `utils/` — PDE residuals and the physics schema (`criterion.py`), diagnostics and helpers.
- `figures/` — the paper's figures and table bodies from the campaign summaries
  (`make_figures.py`, `figures.toml`, `make_snapshots.py`).
- `tests/` — unit tests for the constraint layers, the residuals, the protocol and the figures.

## Environment

Python 3.12, PyTorch 2.5.1, CUDA 12.2. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest
```

## Data

Every path in the configurations and in `figures/figures.toml` starts with the placeholder
`DATA_ROOT`; replace it with the directory that holds the data.

- **E1 (Kolmogorov flow).** `NS_fft_Re500_T4000.npy` from the
  [PINO repository](https://github.com/neuraloperator/physics_informed), converted to velocity
  and pressure with `experiments/data_utils/prepare_pino_kf.py`.
- **E2 (stochastically forced 2D turbulence, 64²).** The coarse dataset is on Zenodo,
  [doi:10.5281/zenodo.23113478](https://doi.org/10.5281/zenodo.23113478). It was generated at
  1024² with [NS2D](https://github.com/m-groom/NS2D) (Dedalus) and truncated with
  `experiments/data_utils/prepare_ns2d_stochastic.py`.

## Training and evaluation

Run the scripts from inside `experiments/`, one configuration and seed at a time; the paper uses
seeds 1 to 5.

```bash
cd experiments
uv run python train_operator_AR_PINO_2d.py --config_path configs/<name>.yaml --seed <s>
uv run python test_operator_AR_2d.py --config_path configs/<name>.yaml --seed <s>
```

The standard rollout is 64 steps; `--rollout_steps 448` evaluates E2's six longer windows.
`aggregate_seeds.py` summarises the seeds of a campaign, and the figures and table bodies are then
regenerated from the summaries with

```bash
uv run python figures/make_figures.py
```

## Licence and citation

Copyright 2026 CSIRO. Released under the Apache License 2.0 (see `LICENSE`).

Groom, M. and Oliveira, R. (2026). Only Linear Constraints Survive Coarse-Graining: Evaluating
Physics-Constrained Neural Operators on Stochastically Forced Turbulence. NeurIPS 2026 Workshop
on AI for Stochastic Dynamics.
