import io
import math
import os
import time
from collections import OrderedDict
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
from einops import rearrange
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from utils.compute_diagnostics import streamfunction_to_velocity, velocity_from_vorticity
from utils.compute_physical_statistics import compute_spectra


class MultipleTensors(Sequence):
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def numel(self):
        return np.sum([x_.numel() for x_ in self.x])

    def __getitem__(self, item):
        return self.x[item]


def get_grid(data, n_dim, multi_channel):
    if n_dim == 1:
        grid = torch.meshgrid(torch.linspace(0, 1, data.shape[1]))
        grid = torch.unsqueeze(grid[0], dim=-1)
    elif n_dim == 2:
        grid = torch.meshgrid(
            torch.linspace(0, 1, data.shape[1]), torch.linspace(0, 1, data.shape[2])
        )
        grid = torch.stack(grid, dim=-1)
    elif n_dim == 3:
        grid = torch.meshgrid(
            torch.linspace(0, 1, data.shape[1]),
            torch.linspace(0, 1, data.shape[2]),
            torch.linspace(0, 1, data.shape[3]),
        )
        grid = torch.stack(grid, dim=-1)
    elif n_dim == 4:
        grid = torch.meshgrid(
            torch.linspace(0, 1, data.shape[1]),
            torch.linspace(0, 1, data.shape[2]),
            torch.linspace(0, 1, data.shape[3]),
            torch.linspace(0, 1, data.shape[4]),
        )
        grid = torch.stack(grid, dim=-1)
    else:
        raise NotImplementedError
    grid = grid.to(data.device)
    if multi_channel:
        grid = grid.unsqueeze(-2)
        data = torch.cat(
            [
                torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * n_dim + [data.shape[-2], 1]),
                data,
            ],
            dim=-1,
        )
    else:
        data = torch.cat(
            [torch.tile(grid.unsqueeze(0), [data.shape[0]] + [1] * n_dim + [1]), data], dim=-1
        )

    return data


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.5f} seconds")
        return result

    return wrapper


def count_parameters(model):
    total_params = 0
    for _name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = 2 * parameter.numel() if parameter.is_complex() else parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_model_from_checkpoint(model, model_state_dict):
    if next(iter(model_state_dict.keys())).startswith("module."):
        new_state_dict = OrderedDict()
        for key, item in model_state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = item
        del model_state_dict
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(model_state_dict)


def resume_training_from_checkpoint(model, saved_path, device, optimizer=None, scheduler=None):

    checkpoint = torch.load(saved_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    model.to(device)
    model.train()
    if os.path.exists(os.path.join(os.path.dirname(saved_path), "error_fft.pth")):
        error_fft = torch.load(os.path.join(os.path.dirname(saved_path), "error_fft.pth"))
        error_fft = error_fft.detach().cpu().numpy()[:-1, :-1]
        # the data is in [n_frequency, n_epoch]
        # cbar = plt.colorbar()

        # draw 3d lineplot for error_fft, x is frequency, y is epoch, z is error, so far error_fft is in [n_frequency, n_epoch]
        # so we need to transpose it to [n_epoch, n_frequency, error_fft]
        # Example data
        n_frequencies = error_fft.shape[0]
        n_epochs = error_fft.shape[1]
        epochs = np.arange(n_epochs)  # x
        frequencies = np.arange(n_frequencies)  # y

        # Create meshgrid for X (epoch) and Y (frequency)
        X, Y = np.meshgrid(epochs, frequencies)
        Z = error_fft

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Draw a separate line for each frequency, the color map is different shades of blue
        fre_list = [*list(range(0, n_frequencies, 5))[1:], 1]
        for i in fre_list:
            ax.plot(
                X[i],
                Z[i],
                label=f"Freq {frequencies[i]}",
                color=plt.cm.Blues((n_frequencies - i + 1) / n_frequencies),
            )  # label only first few

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        # ax.set_title('Plot of Error over Epochs and Frequencies')

        # Optional: show legend for first few lines
        ax.legend(loc="upper right", ncol=5)
        # plt.show()

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(saved_path), "error_fft.png"))
        plt.show()
    else:
        error_fft = None
    return model, optimizer, scheduler, start_epoch


def load_components_from_pretrained(model, state_dict, components="all"):
    """:model: the model
    :state_dict: state_dict of source model
    :components: 'all' or list from 'patch', 'pos', 'blocks','time_agg','cls_head', 'scale_feats', 'out'
    """
    if next(iter(state_dict.keys())).startswith("module."):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = item
        del state_dict
        state_dict = new_state_dict
    if (components == "all") or ("all" in components):
        model.load_state_dict(state_dict)
        return
    else:
        for name in components:
            if name == "patch_embed" and hasattr(model, "patch_embed"):
                model.patch_embed.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("patch_embed.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("patch_embed.")
                        }
                    )
                )
            elif name == "pos" and hasattr(model, "pos_embed"):
                # model.pos_embed.load_state_dict(OrderedDict(
                #     {k.replace('pos_embed.', ''): v for k, v in state_dict.items() if k.startswith('pos_embed.')}))
                model.pos_embed = nn.Parameter(state_dict["pos_embed"])
                # pos_embed_state = OrderedDict({k: v for k, v in state_dict.items() if k.startswith('pos_embed')})
                # if pos_embed_state:
                #     key, value = next(iter(pos_embed_state.items()))
                #     setattr(model, 'pos_embed', value)
            elif name == "blocks" and hasattr(model, "blocks"):
                for i, block in enumerate(model.blocks):
                    block_state_dict = OrderedDict(
                        {
                            k.replace(f"blocks.{i}.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith(f"blocks.{i}.")
                        }
                    )
                    block.load_state_dict(block_state_dict)
            elif name == "scale_feats" and hasattr(model, "scale_feats_mu"):
                model.scale_feats_mu.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("scale_feats_mu.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("scale_feats_mu.")
                        }
                    )
                )
                model.scale_feats_sigma.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("scale_feats_sigma.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("scale_feats_sigma.")
                        }
                    )
                )
            elif name == "cls_head" and hasattr(model, "cls_head"):
                model.cls_head.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("cls_head.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("cls_head.")
                        }
                    )
                )
            elif name == "time_agg" and hasattr(model, "time_agg_layer"):
                model.time_agg_layer.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("time_agg_layer.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("time_agg_layer.")
                        }
                    )
                )
            elif name == "out" and hasattr(model, "out_layer"):
                model.out_layer.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("out_layer.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("out_layer.")
                        }
                    )
                )
            else:
                print(f"Submodule does not exists:{name}")
        return


def load_3d_components_from_2d(model, state_dict, components="all"):
    """:model: the model
    :state_dict: state_dict of source model
    :components: 'all' or list from 'patch', 'pos', 'blocks','time_agg','cls_head', 'scale_feats', 'out'
    """
    if next(iter(state_dict.keys())).startswith("module."):
        new_state_dict = OrderedDict()
        for key, item in state_dict.items():
            new_key = key.replace("module.", "")
            new_state_dict[new_key] = item
        del state_dict
        state_dict = new_state_dict
    if (components == "all") or ("all" in components):
        model.load_state_dict(state_dict)
        return
    else:
        for name in components:
            if name == "blocks" and hasattr(model, "blocks"):
                for i, block in enumerate(model.blocks):
                    block_state_dict = OrderedDict(
                        {
                            k.replace(f"blocks.{i}.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith(f"blocks.{i}.")
                        }
                    )
                    ## reshape 2d conv param to 3d conv param
                    for k, v in block_state_dict.items():
                        if "mlp" in k and "weight" in k:
                            block_state_dict[k] = v.unsqueeze(-1)
                    block.load_state_dict(block_state_dict)

            elif name == "time_agg" and hasattr(model, "time_agg_layer"):
                model.time_agg_layer.load_state_dict(
                    OrderedDict(
                        {
                            k.replace("time_agg_layer.", ""): v
                            for k, v in state_dict.items()
                            if k.startswith("time_agg_layer.")
                        }
                    )
                )
            else:
                print(f"Submodule does not exists:{name}")
        return


def save_results_excel(filename, data_dict):
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for key, value in data_dict.items():
            df = pd.DataFrame(value)
            df.to_excel(writer, sheet_name=key, index=False)


# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def samples_fft(u):
    return scipy.fft.fftn(u, s=u.shape[2:], norm="forward", workers=-1)


def samples_ifft(u_hat):
    return scipy.fft.ifftn(u_hat, s=u_hat.shape[2:], norm="forward", workers=-1).real


def downsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f"Expected either real or complex valued array. Got {u.dtype}.")
    d = u_hat.ndim - 2
    u_hat_down = None
    if d == 2:
        u_hat_down = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_down[:, :, : N // 2 + 1, : N // 2 + 1] = u_hat[:, :, : N // 2 + 1, : N // 2 + 1]
        u_hat_down[:, :, : N // 2 + 1, -N // 2 :] = u_hat[:, :, : N // 2 + 1, -N // 2 :]
        u_hat_down[:, :, -N // 2 :, : N // 2 + 1] = u_hat[:, :, -N // 2 :, : N // 2 + 1]
        u_hat_down[:, :, -N // 2 :, -N // 2 :] = u_hat[:, :, -N // 2 :, -N // 2 :]
    else:
        raise ValueError(f"Invalid dimension {d}")
    if fourier:
        return u_hat_down
    u_down = samples_ifft(u_hat_down)
    return u_down


def upsample(u, N, fourier=False):
    if np.isrealobj(u):
        u_hat = samples_fft(u)
    elif np.iscomplexobj(u):
        u_hat = u
    else:
        raise TypeError(f"Expected either real or complex valued array. Got {u.dtype}.")
    d = u_hat.ndim - 2
    N_old = u_hat.shape[-2]
    u_hat_up = None
    if d == 2:
        u_hat_up = np.zeros((u_hat.shape[0], u_hat.shape[1], N, N), dtype=u_hat.dtype)
        u_hat_up[:, :, : N_old // 2 + 1, : N_old // 2 + 1] = u_hat[
            :, :, : N_old // 2 + 1, : N_old // 2 + 1
        ]
        u_hat_up[:, :, : N_old // 2 + 1, -N_old // 2 :] = u_hat[
            :, :, : N_old // 2 + 1, -N_old // 2 :
        ]
        u_hat_up[:, :, -N_old // 2 :, : N_old // 2 + 1] = u_hat[
            :, :, -N_old // 2 :, : N_old // 2 + 1
        ]
        u_hat_up[:, :, -N_old // 2 :, -N_old // 2 :] = u_hat[:, :, -N_old // 2 :, -N_old // 2 :]
    else:
        raise ValueError(f"Invalid dimension {d}")
    if fourier:
        return u_hat_up
    u_up = samples_ifft(u_hat_up)
    return u_up


## B, C, X, Y; B, X, Y, T, C (temporal)
def resize(x, out_size, permute=False, temporal=False):
    if temporal:
        T, C = x.shape[-2:]
        x = rearrange(x, "b x y t c -> b (t c) x y")
    if permute:
        x = x.permute(0, 3, 1, 2)

    f = torch.fft.rfft2(x, norm="backward")
    f_z = torch.zeros(
        (*x.shape[:-2], out_size[0], out_size[1] // 2 + 1), dtype=f.dtype, device=f.device
    )
    # 2k+1 -> (2k+1 + 1) // 2 = k+1 and (2k+1)//2 = k
    top_freqs1 = min((f.shape[-2] + 1) // 2, (out_size[0] + 1) // 2)
    top_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    # 2k -> (2k + 1) // 2 = k and (2k)//2 = k
    bot_freqs1 = min(f.shape[-2] // 2, out_size[0] // 2)
    bot_freqs2 = min(f.shape[-1], out_size[1] // 2 + 1)
    f_z[..., :top_freqs1, :top_freqs2] = f[..., :top_freqs1, :top_freqs2]
    f_z[..., -bot_freqs1:, :bot_freqs2] = f[..., -bot_freqs1:, :bot_freqs2]
    # x_z = torch.fft.ifft2(f_z, s=out_size).real
    x_z = torch.fft.irfft2(f_z, s=out_size).real
    x_z = x_z * (out_size[0] / x.shape[-2]) * (out_size[1] / x.shape[-1])

    # f_z[..., -f.shape[-2]//2:, :f.shape[-1]] = f[..., :f.shape[-2]//2+1, :]

    if temporal:
        x_z = rearrange(x_z, "b (t c) x y -> b x y t c", t=T, c=C)
    if permute:
        x_z = x_z.permute(0, 2, 3, 1)

    return x_z


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


def _to_rgb_minmax(image_2d: torch.Tensor) -> torch.Tensor:
    """Convert single-channel 2D field to 3-channel RGB with per-image min-max normalization.

    Args:
        image_2d: (H, W) tensor

    Returns:
        (3, H, W) tensor with RGB channels
    """
    img = image_2d.detach().float()
    min_val = torch.amin(img)
    max_val = torch.amax(img)
    if torch.isfinite(min_val) and torch.isfinite(max_val) and (max_val > min_val):
        img = (img - min_val) / (max_val - min_val)
    else:
        img = torch.zeros_like(img)
    return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)


def fig_to_tensorboard_image(fig: Figure) -> torch.Tensor:
    """Convert matplotlib figure to tensor for TensorBoard.

    Args:
        fig: matplotlib Figure object

    Returns:
        (3, H, W) tensor with RGB channels, values in [0, 1]
    """
    # Convert figure to image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)

    # Close figure to free memory
    plt.close(fig)

    # Convert to tensor and normalize to [0, 1]
    if len(img_array.shape) == 3:
        # RGB image
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        # Ensure 3 channels (RGB)
        if img_tensor.shape[0] == 4:
            img_tensor = img_tensor[:3]  # Remove alpha channel if present
        elif img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)  # Convert grayscale to RGB
    else:
        # Grayscale image
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)  # Convert to RGB

    return img_tensor


def log_tensorboard_images_and_spectra(
    writer,
    pred_denorm: torch.Tensor,
    target_denorm: torch.Tensor,
    epoch: int,
    form: str,
    model_name: str,
    Lx: float = 2 * np.pi,
    Ly: float = 2 * np.pi,
):
    """Log prediction, target, error images and energy/enstrophy spectra to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        pred_denorm: Denormalized predictions, shape (B, H, W, T, C)
        target_denorm: Denormalized targets, shape (B, H, W, T, C)
        epoch: Current epoch number
        form: Data form ('vorticity' or 'velocity')
        model_name: Name of the model (for plot labels)
        Lx: Domain size in x direction (default: 2*pi)
        Ly: Domain size in y direction (default: 2*pi)
    """
    B, H, W, T, C = pred_denorm.shape
    # print("pred_denorm shape:", pred_denorm.shape, "target_denorm shape:", target_denorm.shape)
    # Define channel names based on form
    if form == "vorticity":
        channel_names = ["vorticity", "streamfunction"]
    elif form == "velocity":
        channel_names = ["pressure", "velocity_x", "velocity_y"]
    else:
        channel_names = [f"channel_{i}" for i in range(C)]

    # Log images for all output channels
    for channel_idx in range(C):
        # Use descriptive name if available, otherwise use channel index
        if channel_idx < len(channel_names):
            channel_name = channel_names[channel_idx]
        else:
            channel_name = f"channel_{channel_idx}"

        # Extract first batch, first time step
        pred_img = _to_rgb_minmax(pred_denorm[0, :, :, 0, channel_idx])
        target_img = _to_rgb_minmax(target_denorm[0, :, :, 0, channel_idx])
        error_img = _to_rgb_minmax(
            pred_denorm[0, :, :, 0, channel_idx] - target_denorm[0, :, :, 0, channel_idx]
        )

        writer.add_image(f"pred/{channel_name}", pred_img, epoch)
        writer.add_image(f"target/{channel_name}", target_img, epoch)
        writer.add_image(f"error/{channel_name}", error_img, epoch)

    # Compute and plot energy and enstrophy spectra
    # Supports both vorticity and velocity forms
    try:
        # Extract data for pred and target
        # Shape: (B, H, W, T, C) -> extract first batch, first time step
        pred_batch = pred_denorm[0, :, :, 0, :].detach().cpu().numpy()  # (H, W, C)
        target_batch = target_denorm[0, :, :, 0, :].detach().cpu().numpy()  # (H, W, C)

        # Get velocity components based on form
        if form == "vorticity" and C >= 2:
            # For vorticity form: compute velocity from streamfunction
            # Channel 0: vorticity, Channel 1: streamfunction
            psi_pred = pred_batch[:, :, 1]  # streamfunction
            psi_target = target_batch[:, :, 1]  # streamfunction

            # Compute velocity from streamfunction
            ux_pred, uy_pred = streamfunction_to_velocity(psi_pred, Lx, Ly)
            ux_target, uy_target = streamfunction_to_velocity(psi_target, Lx, Ly)
        elif form == "velocity" and C >= 3:
            # For velocity form: use velocity components directly
            # Channel 1: velocity_x, Channel 2: velocity_y
            ux_pred = pred_batch[:, :, 1]  # velocity_x
            uy_pred = pred_batch[:, :, 2]  # velocity_y
            ux_target = target_batch[:, :, 1]  # velocity_x
            uy_target = target_batch[:, :, 2]  # velocity_y
        else:
            ux_pred, uy_pred = velocity_from_vorticity(torch.from_numpy(pred_batch[..., 0]))
            ux_target, uy_target = velocity_from_vorticity(torch.from_numpy(target_batch[..., 0]))

        # Compute spectra for prediction and target
        k_bins, Ek_pred, Zk_pred = compute_spectra(ux_pred, uy_pred, Lx, Ly)
        _, Ek_target, Zk_target = compute_spectra(ux_target, uy_target, Lx, Ly)

        # Create energy spectrum plot
        fig_energy, ax_energy = plt.subplots(figsize=(10, 6))
        k_nyquist = int((np.pi * H) // Lx)

        start_truth = 1
        ax_energy.loglog(
            k_bins[start_truth:k_nyquist],
            Ek_target[start_truth:k_nyquist],
            "X--",
            markersize=1,
            label="Ground Truth",
            linewidth=1,
            color="black",
        )
        ax_energy.loglog(
            k_bins[start_truth:k_nyquist],
            Ek_pred[start_truth:k_nyquist],
            "o-",
            markersize=1,
            label=f"{model_name} Prediction",
            linewidth=1,
            color="blue",
        )
        ax_energy.set_xlabel("Wavenumber", fontsize=14)
        ax_energy.set_ylabel("Energy", fontsize=14)
        ax_energy.set_title("Energy Spectrum", fontsize=14)
        ax_energy.legend(fontsize=12)
        ax_energy.grid(True)
        plt.tight_layout()

        # Convert to tensor and add to TensorBoard
        energy_img = fig_to_tensorboard_image(fig_energy)
        writer.add_image("spectra/energy_spectrum", energy_img, epoch)

        # Create enstrophy spectrum plot
        fig_enstrophy, ax_enstrophy = plt.subplots(figsize=(10, 6))
        ax_enstrophy.loglog(
            k_bins[start_truth:k_nyquist],
            Zk_target[start_truth:k_nyquist],
            "X-",
            markersize=2,
            label="Ground Truth",
            linewidth=2,
            color="black",
        )
        ax_enstrophy.loglog(
            k_bins[start_truth:k_nyquist],
            Zk_pred[start_truth:k_nyquist],
            "o-",
            markersize=2,
            label=f"{model_name} Prediction",
            linewidth=2,
            color="blue",
        )
        ax_enstrophy.set_xlabel("Wavenumber", fontsize=14)
        ax_enstrophy.set_ylabel("Enstrophy", fontsize=14)
        ax_enstrophy.set_title("Enstrophy Spectrum", fontsize=14)
        ax_enstrophy.legend(fontsize=12)
        ax_enstrophy.grid(True)
        plt.tight_layout()

        # Convert to tensor and add to TensorBoard
        enstrophy_img = fig_to_tensorboard_image(fig_enstrophy)
        writer.add_image("spectra/enstrophy_spectrum", enstrophy_img, epoch)
    except Exception as e:
        print(f"Warning: Failed to compute energy/enstrophy spectra: {e}")
        import traceback

        traceback.print_exc()


def save_checkpoint(path, name, model, epoch, optimizer=None, scheduler=None):
    if not torch.cuda.is_available():
        return
    ckpt_dir = path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    optim_dict = optimizer.state_dict() if optimizer is not None else None
    sched_dict = scheduler.state_dict() if scheduler is not None else None

    # os.path.join, not concatenation: a save_dir without a trailing slash used to
    # write the checkpoint beside the run directory rather than inside it, where
    # the evaluation script (which joins) could not find it.
    ckpt_path = os.path.join(ckpt_dir, name)
    torch.save(
        {"model": model_state_dict, "optim": optim_dict, "scheduler": sched_dict, "epoch": epoch},
        ckpt_path,
    )
    print(f"Checkpoint is saved at {ckpt_path}")


def torch2dgrid_2d(num_x, num_y, form="linear", device=None, dtype=None):
    """Args:
        num_x: number of grid points in x direction
        num_y: number of grid points in y direction
        form: 'linear' or 'periodic' (Return sin/cos positional grid: (H, W, 4) if form is 'periodic')
        device: device to place the grid
        dtype: dtype of the grid
    Returns:
        mesh: (num_x, num_y, 2 or 4) tensor
    """
    bot = (0, 0)
    top = (1, 1)
    x_bot, y_bot = bot
    x_top, y_top = top
    x_arr = torch.linspace(x_bot, x_top, steps=num_x, device=device, dtype=dtype)
    y_arr = torch.linspace(y_bot, y_top, steps=num_y, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x_arr, y_arr, indexing="ij")
    if form == "linear":
        mesh = torch.stack([xx, yy], dim=2)
    elif form == "periodic":
        two_pi = 2.0 * math.pi
        feats = [
            torch.sin(two_pi * xx),
            torch.cos(two_pi * xx),
            torch.sin(two_pi * yy),
            torch.cos(two_pi * yy),
        ]
        mesh = torch.stack(feats, dim=-1)
    elif form == "spherical":
        nlat, nlon = num_x, num_y
        # latitude: [-pi/2, pi/2], longitude: [0, 2pi)
        lat = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, nlat, device=device, dtype=dtype)
        lon = torch.linspace(0.0, 2.0 * math.pi, nlon, device=device, dtype=dtype)

        # meshgrid: lat varies along axis 0, lon along axis 1
        phi, lam = torch.meshgrid(lat, lon, indexing="ij")
        x = torch.cos(phi) * torch.cos(lam)
        y = torch.cos(phi) * torch.sin(lam)
        z = torch.sin(phi)
        mesh = torch.stack([x, y, z], dim=-1)  # (nlat, nlon, 3)
    else:
        raise ValueError(f"Invalid form: {form}")
    return mesh


# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # data_dict = {
    #     'a': [[1, 2], [3, 4]],
    #     'b': [[5, 6]]
    # }
    # save_results_excel('test.xlsx', data_dict)
    x = torch.rand(10, 2, 64, 64)
    y = resize(x, [32, 32], permute=False)
    print(y.shape)
