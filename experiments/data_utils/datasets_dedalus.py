import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from einops import rearrange


# todo: load all the data from the mat file in the folder:
# /data/large/pdearena/sw2d_pda/train, stack the needed variables and save it as a numpy array
def load_ns2d_data_split_and_save(load_dir, save_dir, nt=63):
    """Each realization has nt time steps"""
    data = np.load(os.path.join(load_dir, "kolmogorov_dataset.npz"))
    print(data)
    for key in data:
        print(key)
    # field, train_realisations, val_realisations, test_realisations, x_coords, y_coords, t_start, t_end, X_train, Y_train, times_train, realisation_ids_train, X_val, Y_val, times_val, realisation_ids_val, X_test, Y_test, times_test, realisation_ids_test

    X_train, X_val, X_test, y_train, y_val, y_test = (
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["Y_train"],
        data["Y_val"],
        data["Y_test"],
    )
    (
        times_train,
        realisation_ids_train,
        times_val,
        realisation_ids_val,
        times_test,
        realisation_ids_test,
    ) = (
        data["times_train"],
        data["realisation_ids_train"],
        data["times_val"],
        data["realisation_ids_val"],
        data["times_test"],
        data["realisation_ids_test"],
    )
    print("times_train shape: ", times_train.shape)
    print("realisation_ids_train shape: ", realisation_ids_train.shape)

    # X train shape (N*T, C, X, Y), reshape it and use all realisations
    nc, nx, ny = X_train.shape[1:]
    X_train = X_train.reshape(-1, nt, nc, nx, ny)
    X_val = X_val.reshape(-1, nt, nc, nx, ny)
    X_test = X_test.reshape(-1, nt, nc, nx, ny)
    y_train = y_train.reshape(-1, nt, nc, nx, ny)
    y_val = y_val.reshape(-1, nt, nc, nx, ny)
    y_test = y_test.reshape(-1, nt, nc, nx, ny)
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    X_train = X_train.reshape(n_train * nt, nc, nx, ny)
    X_val = X_val.reshape(n_val * nt, nc, nx, ny)
    X_test = X_test.reshape(n_test * nt, nc, nx, ny)
    y_train = y_train.reshape(n_train * nt, nc, nx, ny)
    y_val = y_val.reshape(n_val * nt, nc, nx, ny)
    y_test = y_test.reshape(n_test * nt, nc, nx, ny)

    # save this small prototype training/ val /test set in one npz file
    np.savez(
        os.path.join(save_dir, "kolmogorov_dataset.npz"),
        X_train=X_train,
        y_train=y_train,
        times_train=times_train,
        X_val=X_val,
        y_val=y_val,
        times_val=times_val,
        X_test=X_test,
        y_test=y_test,
        times_test=times_test,
    )
    return data


class NSLoader2D(Dataset):
    def __init__(
        self,
        datapath,
        state="train",
        train=True,
        normalizer_path=None,
        save_normalizer_path=None,
        velocity_channels=(0, 1),
        filename="kolmogorov_dataset.npz",
    ):
        """Load data from npz files (kolmogorov_dataset.npz)

        Args:
            datapath: path to directory containing the npz files
            state: 'train', 'val', or 'test'
            train: if True, data is for training (random sampling), else deterministic
            normalizer_path: path to saved normalizer file (required for val/test, optional for train)
            save_normalizer_path: path to save normalizer after computing from training set
            velocity_channels: channels that form the velocity vector (default: first two
                channels); None for data that carries no velocity vector, such as the
                one-channel vorticity state
            filename: name of the npz file inside datapath
        """
        self.train = train
        self.state = state
        self.velocity_channels = velocity_channels

        # Load data from npz file
        npz_path = os.path.join(datapath, filename)

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Dataset file not found: {npz_path}")

        data_dict = np.load(npz_path)
        # Keys match the state: X_train/y_train for train, X_val/y_val for val, X_test/y_test for test
        X_key = f"X_{state}"
        y_key = f"y_{state}"

        # Fallback: try to find any X/y keys if state-specific ones don't exist
        if X_key not in data_dict or y_key not in data_dict:
            keys = list(data_dict.keys())
            X_key = (
                next(k for k in keys if k.startswith("X"))
                if any(k.startswith("X") for k in keys)
                else None
            )
            y_key = (
                next(k for k in keys if k.startswith("y"))
                if any(k.startswith("y") for k in keys)
                else None
            )
            if X_key is None or y_key is None:
                raise KeyError(
                    f"Could not find X and y keys in {npz_path}. Expected X_{state}/y_{state}. Available keys: {keys}"
                )

        X_data = data_dict[X_key]
        y_data = data_dict[y_key]

        # X_data and y_data are (N, C, H, W) = (N, 3, 128, 128)
        # Stack X and y: (N, C, H, W) -> (N*2, C, H, W) if we want pairs, or just use X and y separately
        # For now, we'll use X as input and y as target
        self.X_data = torch.tensor(X_data, dtype=torch.float32)  # (N, C, H, W)
        self.y_data = torch.tensor(y_data, dtype=torch.float32)  # (N, C, H, W)

        self.S = (self.X_data.shape[-2], self.X_data.shape[-1])  # (H, W) = (128, 256)

        self.num_samples = self.X_data.shape[0]
        print(f"Loaded {state} dataset: {self.num_samples} samples, shape: {self.X_data.shape}")

        # Normalize the data
        self.normalize(normalizer_path, save_normalizer_path)
        print(f"Normalized data - mean shape: {self.mean.shape}, std shape: {self.std.shape}")

    def normalize(self, normalizer_path=None, save_normalizer_path=None):
        """Normalise data with global per-channel mean and scale of shape (C, 1, 1)
        For training set: compute and save normalizer
        For val/test sets: load saved normalizer
        """
        eps = 1e-8
        # Normalise velocity channels isotropically (shared scale) to preserve direction.
        vel_ch = self.velocity_channels
        if vel_ch is not None:
            if isinstance(vel_ch, (int, np.integer)):
                vel_ch = (int(vel_ch),)
            else:
                vel_ch = tuple(int(c) for c in vel_ch)

        if normalizer_path is not None and os.path.exists(normalizer_path):
            # Load saved normalizer
            normalizer = torch.load(normalizer_path)
            self.mean = normalizer["mean"]  # (C, 1, 1)
            self.std = normalizer["std"]  # (C, 1, 1)
            if (
                self.mean.ndim != 3
                or self.std.ndim != 3
                or self.mean.shape[1:] != (1, 1)
                or self.std.shape[1:] != (1, 1)
            ):
                raise ValueError(
                    f"Expected global per-channel normaliser stats of shape (C, 1, 1), "
                    f"got mean {tuple(self.mean.shape)} and std {tuple(self.std.shape)}. "
                    f"Please regenerate the normaliser file."
                )
            print(f"Loaded normalizer from {normalizer_path}")
        elif self.state == "train":
            # Compute global per-channel mean and RMS scale over the full training set.
            self.mean = self.X_data.mean(dim=(0, 2, 3), keepdim=True)  # (1, C, 1, 1)
            self.mean = self.mean.squeeze(0)  # (C, 1, 1)
            self.std = torch.zeros_like(self.mean)  # (C, 1, 1)
            if vel_ch is not None and len(vel_ch) >= 2:
                vel = self.X_data[:, vel_ch, :, :]  # (N, V, H, W)
                vel_mean = self.mean[list(vel_ch), :, :]  # (V, 1, 1)
                vel_dev = vel - vel_mean.unsqueeze(0)
                # Shared global RMS of velocity fluctuations -> isotropic scaling.
                vel_scale = torch.sqrt(torch.mean(vel_dev**2) + eps)  # scalar
                self.std[list(vel_ch), :, :] = vel_scale
            remaining_channels = [
                c for c in range(self.X_data.shape[1]) if vel_ch is None or c not in vel_ch
            ]
            for c in remaining_channels:
                dev = self.X_data[:, c : c + 1, :, :] - self.mean[c : c + 1].unsqueeze(0)
                self.std[c : c + 1] = torch.sqrt(torch.mean(dev**2) + eps)
            print(
                f"Computed normalizer from training data - mean shape: {self.mean.shape}, std shape: {self.std.shape}"
            )

            # Save normalizer if path provided
            if save_normalizer_path is not None:
                os.makedirs(
                    os.path.dirname(save_normalizer_path)
                    if os.path.dirname(save_normalizer_path)
                    else ".",
                    exist_ok=True,
                )
                torch.save(
                    {
                        "mean": self.mean,
                        "std": self.std,
                        "velocity_channels": vel_ch,
                        "normalization": "global_channel_rms",
                    },
                    save_normalizer_path,
                )
                print(f"Saved normalizer to {save_normalizer_path}")
        else:
            raise ValueError(f"normalizer_path must be provided for {self.state} set")

        # Enforce isotropic scaling for the chosen velocity channels.
        if vel_ch is not None and len(vel_ch) >= 2:
            vel_std = self.std[list(vel_ch), :, :]  # (V, 1, 1)
            std_shared = torch.sqrt((vel_std**2).mean() + eps)  # scalar
            self.std[list(vel_ch), :, :] = std_shared

        # Normalize the data: (N, C, H, W) with (C, 1, 1) mean/std
        self.X_data = (self.X_data - self.mean) / (self.std + eps)
        self.y_data = (self.y_data - self.mean) / (self.std + eps)

        # Permute data once here: (N, C, H, W) -> (N, H, W, C) for easier access in __getitem__
        self.X_data = self.X_data.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        self.y_data = self.y_data.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        print(f"Permuted data to (N, H, W, C) - final shape: {self.X_data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Returns input and target with shape (H, W, C)"""
        # Data is already in (N, H, W, C) format, so just return slices
        return self.X_data[idx], self.y_data[idx]  # (H, W, C), (H, W, C)

    def transform_rollout(self, T=63):
        """Reshape self.X_data and self.y_data from (N, H, W, C) to (N, T, H, W, C)"""
        assert self.X_data.shape[0] % T == 0, "Number of samples must be divisible by T"
        n_traj = self.X_data.shape[0] // T

        self.X_data = rearrange(self.X_data, "(n t) h w c -> n h w t c", t=T)
        self.y_data = rearrange(self.y_data, "(n t) h w c -> n h w t c", t=T)
        # self.X_data = rearrange(self.X_data,   '(t n) h w c -> n h w t c', t=T)
        # self.y_data = rearrange(self.y_data,   '(t n) h w c -> n h w t c', t=T)
        self.num_samples = n_traj
        # self.num_samples = 1
        self.T = T
        print(f"Reshaped data to (N, T, H, W, C) - final shape: {self.X_data.shape}")
        return self.X_data, self.y_data


def verify_time_steps(folder_path, state="test"):
    # Load data from npz file
    npz_filename = f"sw2d_{state}_dataset.npz"
    npz_path = os.path.join(folder_path, npz_filename)

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset file not found: {npz_path}")

    data_dict = np.load(npz_path)
    # Keys match the state: X_train/y_train for train, X_val/y_val for val, X_test/y_test for test
    X_key = f"X_{state}"
    y_key = f"y_{state}"
    print(data_dict.keys())
    X_data = data_dict[X_key]
    y_data = data_dict[y_key]
    realisation_names = data_dict["test_realisations"]
    times = data_dict["times_test"]
    print("realisation_names shape: ", realisation_names.shape)
    print("times shape: ", times.shape)
    print("realisation_names: ", realisation_names)
    print("times: ", times)
    return X_data, y_data


if __name__ == "__main__":
    load_ns2d_data_split_and_save(load_dir="DATA_ROOT/ns2d", save_dir="DATA_ROOT/Kolmogorov")
