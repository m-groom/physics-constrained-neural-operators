"""Step-448 energy spectra of E2's six with-truth test windows, one arm at a time.

The archive this extends is
DATA_ROOT/e3_closed/figures_data/step448_spectra_sixwindows.npz,
whose JSON sidecar records the convention: the test split is 2,688 consecutive
coarse pairs, cut into six non-overlapping 448-step windows; each window is
rolled out autoregressively from its first input with the output constraint
applied at every step; the spectrum of the final state is the isotropic shell
sum of utils.compute_physical_statistics.compute_spectra, kept at shells 1..31.

Two modes:
  * the arm is absent from the archive: compute it for the given seeds and write
    a new npz (--out) holding every existing key plus pred_<arm>;
  * the arm is present: recompute the given seeds and report the maximum
    relative difference against the archived rows, as a check that this script
    reproduces the convention. Exit code 1 if the difference exceeds --tol.

Run from experiments/ so the local module imports resolve:
  python sixwindow_spectra.py --runs_dir <campaign runs dir> --arm FNO_proj_cont \
      --archive <npz> --out <npz>
"""

import json
import os
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_utils.datasets_dedalus import NSLoader2D
from run_protocol import seed_everything
from test_operator_AR_2d import (
    autoregressive_predict,
    check_split_available,
    denormalise,
    resolve_checkpoint_path,
)

from models.fno import FNO2d
from utils.compute_physical_statistics import compute_spectra
from utils.criterion import build_forcing_for_data, physics_from_config
from utils.utilities import torch2dgrid_2d

SHELLS = slice(1, 32)  # shells 1..31, as archived


def build_model(config, norm_mean, norm_std, device):
    """The FNO2d of test_operator_AR_2d.main(), for a velocity arm."""
    data_config = config["data"]
    model_cfg = config["model"]
    physics = physics_from_config(data_config)
    output_constraint_cfg = model_cfg.get("output_constraint", {})
    model = FNO2d(
        in_dim=model_cfg.get("in_dim", 3),
        out_dim=model_cfg.get("out_dim", 1),
        modes1=model_cfg["modes1"],
        modes2=model_cfg["modes2"],
        fc_dim=model_cfg["fc_dim"],
        layers=model_cfg["layers"],
        act=model_cfg["act"],
        output_constraint=output_constraint_cfg,
        physics=physics,
    ).to(device)
    if output_constraint_cfg.get("enabled", False):
        model.set_output_normalizer(norm_mean, norm_std)
        if output_constraint_cfg.get("energy_balance", {}).get("enabled", False):
            nx, ny = int(data_config["nx"]), int(data_config["ny"])
            forcing = build_forcing_for_data(physics, (nx, ny), device=device)
            model.set_energy_forcing(forcing.squeeze(-1))
    return model


def load_checkpoint(model, config, seed):
    """The strict-with-one-exception load of test_operator_AR_2d.main()."""
    train_cfg = config.get("train", {})
    ckpt_path = resolve_checkpoint_path(
        train_cfg.get("save_dir"), train_cfg.get("save_name"), seed, which="best"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = dict(ckpt["model"])
    allowed_missing = {"output_projector.forcing_hat"}
    model_state = model.state_dict()
    for key in allowed_missing:
        if key in model_state and key not in state:
            state[key] = model_state[key]
    result = model.load_state_dict(state, strict=False)
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    return ckpt_path, ckpt.get("epoch")


def window_spectra(config, seed, steps, device):
    """(n_windows, 31) step-``steps`` spectra for one seed of one arm."""
    seed_everything(seed)
    data_config = config["data"]
    check_split_available(data_config["datapath"], data_config["filename"], "test")
    test_set = NSLoader2D(
        datapath=data_config["datapath"],
        state="test",
        train=False,
        normalizer_path=data_config.get("normalizer_path", None),
        velocity_channels=data_config.get("velocity_channels", (0, 1)),
        filename=data_config["filename"],
    )
    norm_mean, norm_std = test_set.mean, test_set.std
    test_set.transform_rollout(T=steps)
    loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False, num_workers=0)

    model = build_model(config, norm_mean, norm_std, device)
    ckpt_path, epoch = load_checkpoint(model, config, seed)
    print(f"  loaded {ckpt_path} (epoch {epoch}); {len(test_set)} windows of {steps} steps")

    grid = torch2dgrid_2d(
        test_set.S[0],
        test_set.S[1],
        form=data_config["grid_form"],
        device=device,
        dtype=torch.float32,
    )
    constrained = bool(config["model"].get("output_constraint", {}).get("enabled", False))
    _, pred_seq, _ = autoregressive_predict(
        model,
        loader,
        device,
        grid,
        use_residual=config["model"].get("residual", False),
        constrain_output=constrained,
    )
    final = denormalise(pred_seq[..., -1:], norm_mean, norm_std, device)[..., 0]
    final = final.detach().cpu().numpy()  # (n_windows, S, S, C)

    length = float(config["data"]["domain_length"])
    rows = []
    for state in final:
        if not np.all(np.isfinite(state[..., :2])):
            rows.append(np.full(31, np.nan))
            continue
        _, ek, _ = compute_spectra(state[..., 0], state[..., 1], length, length)
        rows.append(ek[SHELLS])
    return np.stack(rows), ckpt_path, epoch


def main():
    """CLI entry point; see the module docstring."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True, help="campaign runs directory")
    parser.add_argument("--arm", required=True, help="config name, e.g. FNO_proj_cont")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=448)
    parser.add_argument("--archive", required=True, help="existing sixwindows npz")
    parser.add_argument("--out", default=None, help="merged npz to write (compute mode)")
    parser.add_argument("--tol", type=float, default=1e-4, help="verify-mode threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    archive = dict(np.load(args.archive))
    key = f"pred_{args.arm}"
    verify = key in archive

    rows, provenance = {}, {}
    for seed in args.seeds:
        run_dir = os.path.join(args.runs_dir, f"{args.arm}_seed{seed}")
        config_path = os.path.join(run_dir, f"{args.arm}.yaml")
        with open(config_path) as stream:
            config = yaml.load(stream, yaml.FullLoader)
        print(f"{args.arm} seed {seed}")
        rows[seed], ckpt_path, epoch = window_spectra(config, seed, args.steps, device)
        provenance[f"{args.arm}_seed{seed}"] = {
            "config": config_path,
            "checkpoint": ckpt_path,
            "checkpoint_epoch": epoch,
            "constrained": bool(config["model"].get("output_constraint", {}).get("enabled")),
        }

    if verify:
        worst = 0.0
        for seed in args.seeds:
            stored = archive[key][seed - 1]
            both = np.isfinite(stored) & np.isfinite(rows[seed])
            rel = np.abs(rows[seed][both] - stored[both]) / np.abs(stored[both])
            worst = max(worst, float(rel.max()))
            if not np.array_equal(np.isfinite(stored), np.isfinite(rows[seed])):
                print(f"  seed {seed}: NaN pattern differs from the archive")
                sys.exit(1)
        print(f"verify {key}: max relative difference {worst:.3e} (tol {args.tol:.0e})")
        sys.exit(0 if worst <= args.tol else 1)

    if args.out is None:
        sys.exit("--out is required when the arm is not yet archived")
    stacked = np.stack([rows[seed] for seed in sorted(rows)])
    archive[key] = stacked
    np.savez_compressed(args.out, **archive)

    sidecar = os.path.splitext(args.archive)[0] + ".json"
    with open(sidecar) as stream:
        meta = json.load(stream)
    meta["provenance"]["runs"].update(provenance)
    meta["provenance"]["extended"] = {
        "arm": args.arm,
        "script": "experiments/sixwindow_spectra.py",
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with open(os.path.splitext(args.out)[0] + ".json", "w") as stream:
        json.dump(meta, stream, indent=1)
    print(f"wrote {args.out} with {key} {stacked.shape}")


if __name__ == "__main__":
    main()
