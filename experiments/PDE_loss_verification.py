# verify the PDE loss computation
# use the ground truth of the data to see if the boundary condition, and PDE loss is satisfied

import os
import sys
from argparse import ArgumentParser

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import yaml

from utils.criterion import (
    PINO_loss3d_physics,
    build_forcing,
    physics_from_config,
    residual_scales_from_truth,
)


def load_ns_ground_truth(datapath, nt=63, device=None):
    """Load the ground truth data from the dataset
    return:
        u: (N, nc, nx, ny, nt+1)
        S_forcing: int
    """
    data = np.load(os.path.join(datapath, "kolmogorov_dataset.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    nc, nx, ny = X_train.shape[1:]
    X_train = X_train.reshape(-1, nt, nc, nx, ny)
    y_train = y_train.reshape(-1, nt, nc, nx, ny)

    u = np.concatenate(
        [X_train[:, :1], y_train], axis=1
    )  # (N, nt+1, nc, nx, ny) concatenate the initial condition and the trajectory
    u = torch.from_numpy(u).to(device)
    u = u.permute(0, 2, 3, 4, 1)  # (N, nt+1, 3, nx, ny) -> (N, 3, nx, ny, nt+1)
    S_forcing = nx
    return u, S_forcing


def verify_pde_loss(u, forcing, physics):
    """u: (N, nc, nx, ny, nt+1) from load_ns_ground_truth, compute PINO_loss to test if the PDE loss is zero
    forcing: the run's forcing, from build_forcing
    physics: the run's Physics, from physics_from_config
    return:
        total_loss_cont: float
        total_loss_ic: float
        total_loss_momx: float
        total_loss_momy: float
    """
    total_loss_cont = 0.0
    total_loss_ic = 0.0
    total_loss_momx = 0.0
    total_loss_momy = 0.0
    for i in range(len(u)):
        # print(f'Sample {i} shape: {data[i].shape}')
        u_i = u[i].unsqueeze(0)  # one realization (1, 3, nx, ny, nt+1)
        nt = u_i.shape[-1]
        t_interval = physics.dt * (nt - 1)
        # These are ground-truth trajectories, so the scales come from u_i itself.
        loss_ic, loss_cont, loss_momx, loss_momy, _, _, _ = PINO_loss3d_physics(
            u_i,
            forcing,
            physics,
            t_interval=t_interval,
            scales=residual_scales_from_truth(u_i, physics, t_interval),
        )
        print(
            f"Sample {i} PDE loss: {loss_cont.item()}, IC loss: {loss_ic.item()}, momx loss: {loss_momx.item()}, momy loss: {loss_momy.item()}"
        )
        total_loss_cont += loss_cont.item()
        total_loss_ic += loss_ic.item()
        total_loss_momx += loss_momx.item()
        total_loss_momy += loss_momy.item()
    print(f"average PDE loss: {total_loss_cont / len(u)}")
    print(f"average IC loss: {total_loss_ic / len(u)}")
    print(f"average momx loss: {total_loss_momx / len(u)}")
    print(f"average momy loss: {total_loss_momy / len(u)}")
    return (
        total_loss_cont / len(u),
        total_loss_ic / len(u),
        total_loss_momx / len(u),
        total_loss_momy / len(u),
    )


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = ArgumentParser(description="Verify the PDE loss on ground-truth trajectories")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path) as stream:
        config = yaml.load(stream, yaml.FullLoader)
    physics = physics_from_config(config["data"])

    u, S_forcing = load_ns_ground_truth(datapath=config["data"]["datapath"], device=device)
    forcing = build_forcing(physics, S_forcing, device=device)

    verify_pde_loss(u, forcing, physics)
