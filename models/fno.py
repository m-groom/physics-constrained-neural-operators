import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.layers import SpectralDivergenceFreeProjector2D, ZeroMeanProjector2D

################################################################
# fourier layer
################################################################


class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(2, in_channels, out_channels, self.modes1, self.modes2)
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        out_real = torch.einsum("bixy,ioxy->boxy", input.real, weights[0]) - torch.einsum(
            "bixy,ioxy->boxy", input.imag, weights[1]
        )
        out_imag = torch.einsum("bixy,ioxy->boxy", input.real, weights[1]) + torch.einsum(
            "bixy,ioxy->boxy", input.imag, weights[0]
        )
        return torch.view_as_complex(torch.stack([out_real, out_imag], dim=-1))
        # return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # FFT operations work best with float32 for numerical stability
        # Cast to float32 for FFT (if not already float32)
        x_float = x.to(torch.float32)

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x_float)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x_float.size(-2),
            x_float.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x_out = torch.fft.irfft2(out_ft, s=(x_float.size(-2), x_float.size(-1)))

        return x_out


class FNO2dDedalus(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        img_size=64,
        n_channels=1,
        in_timesteps=10,
        out_timesteps=1,
        n_layers=4,
        patch_size=1,
        use_ln=False,
        multi_channel=True,
        normalize=False,
        n_cls=0,
        meanstd=False,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.img_size = img_size
        self.use_ln = use_ln
        self.padding = 2  # pad the domain if input is non-periodic
        # self.patch_size = patch_size

        self.normalize = normalize
        self.n_cls = n_cls
        self.meanstd = meanstd
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.fc0 = nn.Linear(in_timesteps * n_channels + 2, self.width)

        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
                for _ in range(self.n_layers)
            ]
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)]
        )
        # self.ln_layers = nn.ModuleList([nn.LayerNorm([ self.in_shape[0], self.in_shape[1]]) for _ in range(self.n_layers)])
        if self.normalize:
            self.scale_feats = nn.Linear(2 * n_channels, width)
        if self.use_ln:
            self.ln_layers = nn.ModuleList(
                [nn.GroupNorm(4, self.width) for _ in range(self.n_layers)]
            )

        self.fc1 = nn.Linear(self.width, self.width)

        if self.meanstd:
            self.fc2 = nn.Linear(self.width, n_channels * out_timesteps * 2)
        else:
            self.fc2 = nn.Linear(self.width, n_channels * out_timesteps)

    def input_proj(self, x):
        T, C = x.shape[-2], x.shape[-1]
        if self.normalize:
            mu, sigma = (
                x.mean(dim=(1, 2, 3), keepdim=True),
                x.std(dim=(1, 2, 3), keepdim=True) + 1e-6,
            )  # B,1,1,1,C
            x = (x - mu) / sigma
            scale_feats = (
                self.scale_feats(torch.cat([mu, sigma], dim=-1)).squeeze(-2).contiguous()
            )  # B, 1, 1, C
        else:
            scale_feats = 0.0
        x = x.view(*x.shape[:-2], -1)  #### B, X, Y, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)  #### B, X, Y, T*C +2
        x = self.fc0(x) + scale_feats
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.normalize:
            return x, (mu, sigma)
        else:
            return x, (None, None)

    def forward(self, x):
        """x: (B, H, W, T, C)"""
        x, (mu, sigma) = self.input_proj(x)

        # print(x.shape, scale_feats.shape, self.normalize)
        # x = self.patch_embed(x) + scale_feats
        # x = x + scale_feats

        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.convs[i](x)
            x = x1 + x2
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x)

        # classification

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)  # mlp

        x = F.gelu(x)
        x = self.fc2(x)

        x = x.reshape(*x.shape[:3], self.out_timesteps, -1)

        if self.normalize:
            x = x * sigma + mu

        # return x, cls_pred
        return x

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_latent_by_index(self, x, start_block_index=0):
        x, (mu, sigma) = self.input_proj(x)  # encoder
        if start_block_index == 0:
            # (B, C, H, W) -> (B, H, W, 1, C)
            x = x.permute(0, 2, 3, 1).unsqueeze(-2)
            return x
        for i in range(start_block_index):
            x = self.spectral_convs[i](x)
            x = self.convs[i](x)
            x = F.gelu(x)
            if self.use_ln:
                x = self.ln_layers[i](x)  # (B, C, H, W)
        # (B, C, H, W) -> (B, H, W, 1, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(-2)
        return x

    def get_testing_block_by_index(self, index, x):
        # the input comes in shape (B, H, W, 1, C), change it to (B, C, H, W)
        x = x.squeeze(-2).permute(0, 3, 1, 2).contiguous()
        x = self.spectral_convs[index](x)
        x = self.convs[index](x)
        x = F.gelu(x)
        if self.use_ln:
            x = self.ln_layers[index](x)
        # (B, C, H, W) -> (B, H, W, 1, C)
        x = x.permute(0, 2, 3, 1).unsqueeze(-2)
        return x


def add_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = F.pad(x, (num_pad2[0], num_pad2[1], num_pad1[0], num_pad1[1]), "constant", 0.0)
    else:
        res = x
    return res


def remove_padding2(x, num_pad1, num_pad2):
    if max(num_pad1) > 0 or max(num_pad2) > 0:
        res = x[..., num_pad1[0] : -num_pad1[1], num_pad2[0] : -num_pad2[1]]
    else:
        res = x
    return res


def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    else:
        raise ValueError(f"{act} is not supported")
    return func


class FNO2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width=64,
        fc_dim=128,
        layers=None,
        in_dim=3,
        out_dim=1,
        act="gelu",
        pad_ratio=[0.0, 0.0],
        output_constraint=None,
        physics=None,
    ):
        super().__init__()
        """
        Args:
            - modes1: list of int, number of modes in first dimension in each layer
            - modes2: list of int, number of modes in second dimension in each layer
            - width: int, optional, if layers is None, it will be initialized as [width] * [len(modes1) + 1] 
            - in_dim: number of input channels
            - out_dim: number of output channels
            - act: activation function, {tanh, gelu, relu, leaky_relu}, default: gelu
            - pad_ratio: list of float, or float; portion of domain to be extended. If float, paddings are added to the right. 
            If list, paddings are added to both sides. pad_ratio[0] pads left, pad_ratio[1] pads right. 
            - physics: utils.criterion.Physics for the run; supplies the domain length, and the
            viscosity, friction and time step of the energy-balance projector.
        """
        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, "Cannot add padding in more than 2 directions"
        self.modes1 = modes1
        self.modes2 = modes2

        self.pad_ratio = pad_ratio
        # input channel is 3: (a(x, y), x, y)
        if layers is None:
            self.layers = [width] * (len(modes1) + 1)
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv2d_fast(in_size, out_size, mode1_num, mode2_num)
                for in_size, out_size, mode1_num, mode2_num in zip(
                    self.layers, self.layers[1:], self.modes1, self.modes2, strict=False
                )
            ]
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:], strict=False)
            ]
        )

        self.fc1 = nn.Linear(self.layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, self.layers[-1])
        self.fc3 = nn.Linear(self.layers[-1], out_dim)
        self.act = _get_act(act)

        self.output_constraint = dict(output_constraint or {})
        self.output_constraint_enabled = bool(self.output_constraint.get("enabled", False))
        self.constraint_velocity_channels = tuple(
            self.output_constraint.get("velocity_channels", [0, 1])
        )
        self._constraint_mean = None
        self._constraint_std = None
        self.output_projector = None

        if self.output_constraint_enabled:
            constraint_type = self.output_constraint.get("type")
            constraint_space = self.output_constraint.get("space")
            zero_mean = bool(self.output_constraint.get("zero_mean", False))

            if constraint_type is not None and constraint_type != "divergence_free":
                raise NotImplementedError(
                    f"Unsupported FNO output constraint type: {constraint_type}"
                )
            if constraint_space != "physical":
                raise NotImplementedError(
                    f"Unsupported FNO output constraint space: {constraint_space}"
                )
            if len(self.constraint_velocity_channels) != 2:
                raise ValueError("velocity_channels must contain exactly two channels")

            if constraint_type == "divergence_free":
                if "domain_lengths" in self.output_constraint:
                    domain_lengths = tuple(
                        float(length) for length in self.output_constraint["domain_lengths"]
                    )
                elif physics is not None:
                    domain_lengths = (float(physics.domain_length),) * 2
                else:
                    raise ValueError(
                        "the divergence-free constraint needs a domain: pass physics "
                        "(from config['data']) or output_constraint.domain_lengths"
                    )
                if len(domain_lengths) != 2:
                    raise ValueError("domain_lengths must contain exactly two entries")

                energy_cfg = self.output_constraint.get("energy_balance", {})
                energy_balance_enabled = bool(energy_cfg.get("enabled", False))
                energy_kwargs = {}
                if energy_balance_enabled:
                    # The physics of the energy balance comes from config['data'], never
                    # from this block: two sources of truth is how the friction of the
                    # projector once drifted 2.5e6x away from the friction of the data.
                    stale = [
                        key
                        for key in ("viscosity", "friction", "dt", "forcing")
                        if key in energy_cfg
                    ]
                    if stale:
                        raise ValueError(
                            f"output_constraint.energy_balance must not set {sorted(stale)}; "
                            "the energy balance reads nu, alpha, dt and the forcing "
                            "from config['data']"
                        )
                    if physics is None:
                        raise ValueError(
                            "the energy-balance constraint requires physics (from config['data'])"
                        )
                    energy_kwargs = {
                        "energy_balance": True,
                        "viscosity": float(physics.nu),
                        "friction": float(physics.alpha),
                        "dt": float(physics.dt),
                        "max_iterations": int(energy_cfg.get("max_iterations", 3)),
                        "residual_tol": float(energy_cfg.get("residual_tol", 1e-10)),
                        # The closure for the cross-cut-off inflow, fitted from the coarse
                        # training split. It is exempt from the guard above because it has
                        # no counterpart in config['data'] to drift from -- nu, alpha, dt
                        # and the forcing all do, which is why they are rejected here.
                        "subgrid_source": float(energy_cfg.get("subgrid_source", 0.0)),
                    }

                self.output_projector = SpectralDivergenceFreeProjector2D(
                    domain_lengths=domain_lengths,
                    zero_mean=zero_mean,
                    **energy_kwargs,
                )
            elif zero_mean:
                self.output_projector = ZeroMeanProjector2D()

    @staticmethod
    def _format_output_normalizer_stat(stat):
        if stat is None:
            raise ValueError("Output constraint normalizer statistics cannot be None")
        if stat.ndim != 3:
            raise ValueError("Normalizer statistics must have shape (C, 1, 1) or (1, 1, C)")
        if stat.shape[1:] == (1, 1):
            return stat.permute(1, 2, 0).detach().clone()
        if stat.shape[:2] == (1, 1):
            return stat.detach().clone()
        raise ValueError("Normalizer statistics must have shape (C, 1, 1) or (1, 1, C)")

    def set_output_normalizer(self, mean, std):
        if not self.output_constraint_enabled:
            return
        device = self.fc3.weight.device
        dtype = self.fc3.weight.dtype
        self._constraint_mean = self._format_output_normalizer_stat(mean).to(
            device=device, dtype=dtype
        )
        self._constraint_std = self._format_output_normalizer_stat(std).to(
            device=device, dtype=dtype
        )

    def set_energy_forcing(self, forcing):
        """Set the forcing field for the energy balance projector.

        Args:
            forcing: ``torch.Tensor`` of shape ``(2, Nx, Ny)`` or
                ``(1, 2, Nx, Ny)``, built from the run's forcing configuration.
        """
        if self.output_projector is None or not getattr(
            self.output_projector, "energy_balance", False
        ):
            return
        if not isinstance(forcing, torch.Tensor):
            raise TypeError(
                "set_energy_forcing expects the forcing field built from config['data'], "
                f"got {type(forcing).__name__}"
            )
        self.output_projector.set_forcing(forcing)

    def apply_output_constraint(self, pred, x_old=None):
        """Apply hard physical constraints to the predicted output.

        Args:
            pred: Predicted fields shaped ``(B, H, W, C)`` in normalized
                space.
            x_old: Previous-step fields shaped ``(B, H, W, C)`` in
                normalized space.  Required for energy balance enforcement;
                ignored when only divergence-free / zero-mean constraints
                are active.

        Returns:
            Constrained prediction of the same shape.
        """
        if not self.output_constraint_enabled:
            return pred
        if self._constraint_mean is None or self._constraint_std is None:
            raise RuntimeError("Output constraint normalizer has not been attached to the model")
        if pred.ndim != 4:
            raise ValueError("FNO2d output constraint expects predictions shaped (B, H, W, C)")

        mean = self._constraint_mean.to(device=pred.device, dtype=pred.dtype)
        std = self._constraint_std.to(device=pred.device, dtype=pred.dtype)
        eps = 1e-8

        velocity = torch.stack(
            [pred[..., channel_idx] for channel_idx in self.constraint_velocity_channels],
            dim=-1,
        )
        vel_mean = torch.stack(
            [mean[..., channel_idx] for channel_idx in self.constraint_velocity_channels],
            dim=-1,
        )
        vel_std = torch.stack(
            [std[..., channel_idx] for channel_idx in self.constraint_velocity_channels],
            dim=-1,
        )
        velocity_phys = velocity * (vel_std + eps) + vel_mean

        # Denormalize u_old if provided and energy balance is active
        vel_old_phys = None
        if x_old is not None and getattr(self.output_projector, "energy_balance", False):
            vel_old = torch.stack(
                [x_old[..., ch] for ch in self.constraint_velocity_channels],
                dim=-1,
            )
            vel_old_phys = (vel_old * (vel_std + eps) + vel_mean).permute(0, 3, 1, 2)

        velocity_proj = self.output_projector(
            velocity_phys.permute(0, 3, 1, 2),
            u_old=vel_old_phys,
        ).permute(0, 2, 3, 1)
        velocity_norm = (velocity_proj - vel_mean) / (vel_std + eps)

        constrained = pred.clone()
        for local_idx, channel_idx in enumerate(self.constraint_velocity_channels):
            constrained[..., channel_idx] = velocity_norm[..., local_idx]
        return constrained

    def forward(self, x):
        """Args:
            - x : (batch size, x_grid, y_grid, 2)

        Returns:
            - x: (batch size, x_grid, y_grid, 1)
        """
        size_1, size_2 = x.shape[1], x.shape[2]
        if max(self.pad_ratio) > 0:
            num_pad1 = [round(i * size_1) for i in self.pad_ratio]
            num_pad2 = [round(i * size_2) for i in self.pad_ratio]
        else:
            num_pad1 = num_pad2 = [0.0]

        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # B, C, X, Y
        x = add_padding2(x, num_pad1, num_pad2)
        size_x, size_y = x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws, strict=False)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(
                batchsize, self.layers[i + 1], size_x, size_y
            )
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = remove_padding2(x, num_pad1, num_pad2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # x = torch.rand(1, 128, 128, 3)
    # model = FNO2d(12, 12, 32, 128)

    x = torch.rand(2, 96, 192, 3)
    # model = FNO2d(12, 12, 32, img_size=(96, 192), normalize=True)

    model = FNO2d(
        modes1=[16, 16, 16, 16],
        modes2=[16, 16, 16, 16],
        fc_dim=128,
        layers=[64, 64, 64, 64, 64, 64],
        act="gelu",
        in_dim=3,
        out_dim=1,
    )
    print("total parameters:", sum(p.numel() for p in model.parameters()))
    y = model(x)
    print(y.shape)
