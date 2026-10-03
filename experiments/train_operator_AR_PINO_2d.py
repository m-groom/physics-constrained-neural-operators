import csv
import math
import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import yaml
from data_utils.datasets_dedalus import NSLoader2D
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from run_protocol import (
    best_logged_eval_loss,
    check_run_directory,
    loader_kwargs,
    seed_everything,
    start_training_log,
    write_run_manifest,
)
from tqdm import tqdm

from models.fno import FNO2d
from utils.criterion import (
    LpLoss,
    PINO_loss3d_physics,
    build_forcing_for_data,
    check_model_channels,
    curl_rel_l2_2d,
    max_abs_divergence_2d_velocity,
    physics_from_config,
    residual_has_scale,
    residual_scales_from_truth,
)
from utils.utilities import (
    count_parameters,
    save_checkpoint,
    torch2dgrid_2d,
)

TRAIN_LOG_COLUMNS = [
    "epoch",
    "train_l2",
    "train_cont_rel",
    "train_momx_rel",
    "train_momy_rel",
    "train_cont_abs",
    "train_momx_abs",
    "train_momy_abs",
    "train_div_max",
    "train_mean_vel",
    "train_curl_rel",
    "eval_l2",
    "eval_cont_rel",
    "eval_momx_rel",
    "eval_momy_rel",
    "eval_cont_abs",
    "eval_momx_abs",
    "eval_momy_abs",
    "eval_div_max",
    "eval_mean_vel",
]


def get_training_log_path(config):
    """Path of this run's training log, beside its checkpoints and manifest."""
    save_name_stem = os.path.splitext(config["train"]["save_name"])[0]
    return os.path.join(config["train"]["save_dir"], f"{save_name_stem}_training_log.csv")


def append_training_log(path, row):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_LOG_COLUMNS)
        writer.writerow(row)


def model_uses_output_constraint(model):
    return bool(getattr(model, "output_constraint_enabled", False))


def apply_model_output_constraint(model, pred, x_old=None):
    if model_uses_output_constraint(model):
        return model.apply_output_constraint(pred, x_old=x_old)
    return pred


def get_model_constraint_domain_lengths(model):
    """Domain of the model's output projector, as configured from config['data']."""
    domain_lengths = getattr(getattr(model, "output_projector", None), "domain_lengths", None)
    if domain_lengths is None:
        raise ValueError("the model's output projector carries no domain lengths")
    return tuple(float(length) for length in domain_lengths)


def compute_physics_losses(x_phys, pred_phys, physics, forcing, scales):
    """PDE losses for one (u_t, u_{t+1}) pair, in the configured formulation."""
    u = torch.stack([x_phys, pred_phys], dim=-1).permute(0, 3, 1, 2, 4)
    return PINO_loss3d_physics(u, forcing, physics, t_interval=physics.dt, scales=scales)


def diagnostic_domain_lengths(model, physics):
    """Domain the diagnostics are computed on: the projector's, or the data's."""
    if model_uses_output_constraint(model):
        return get_model_constraint_domain_lengths(model)
    return (float(physics.domain_length), float(physics.domain_length))


def velocity_in_physical_units(model, field, denorm_mean, denorm_std):
    """The two velocity channels of a normalised state, denormalised.

    Args:
        model: The operator, which names its own velocity channels when it is
            constrained; an unconstrained model carries (ux, uy) first.
        field: A state of shape ``(B, H, W, C)`` in normalised units.
        denorm_mean: Per-channel mean, shape ``(1, 1, C)``.
        denorm_std: Per-channel standard deviation, shape ``(1, 1, C)``.

    Returns:
        Tensor of shape ``(B, H, W, 2)`` in physical units.
    """
    velocity_channels = tuple(getattr(model, "constraint_velocity_channels", (0, 1)))
    field_phys = field * (denorm_std + 1e-8) + denorm_mean
    return torch.stack([field_phys[..., index] for index in velocity_channels], dim=-1)


def compute_model_div_max(model, pred, denorm_mean, denorm_std, physics):
    """Batch-mean max |div u| of the prediction, for every model.

    The unconstrained baseline needs this diagnostic as much as the constrained
    models do: without it the projector's divergence has nothing to beat.
    """
    if physics.formulation != "velocity" or denorm_mean is None or denorm_std is None:
        # Without normalisation statistics (a synthetic run) there is no physical
        # space to measure the divergence in; train_2d already refuses that case
        # for constrained models.
        return math.nan
    div_max = max_abs_divergence_2d_velocity(
        velocity_in_physical_units(model, pred, denorm_mean, denorm_std),
        domain_lengths=diagnostic_domain_lengths(model, physics),
    )
    return div_max.mean().item()


def compute_model_mean_vel(model, pred, denorm_mean, denorm_std, physics):
    """Mean absolute domain-mean velocity (averaged over batch), for every model."""
    if physics.formulation != "velocity" or denorm_mean is None or denorm_std is None:
        return math.nan
    velocity_phys = velocity_in_physical_units(model, pred, denorm_mean, denorm_std)
    mean_ux = velocity_phys[..., 0].mean(dim=(1, 2)).abs().mean()
    mean_uy = velocity_phys[..., 1].mean(dim=(1, 2)).abs().mean()
    return (mean_ux + mean_uy).item()


def compute_curl_loss(model, pred, y, denorm_mean, denorm_std, physics):
    """Relative L2 between the vorticity of the prediction and of the target.

    The gradient is kept, unlike the diagnostics above: this is a loss term, not a
    measurement. It is taken in physical units because the two velocity components
    carry separate normalising scales, and a curl of the normalised state would mix
    them.

    Raises:
        ValueError: The run cannot carry the term -- a formulation with no velocity,
            or no normalisation statistics to reach physical units with. Silently
            training without a term the configuration asked for is the worse failure.
    """
    if physics.formulation != "velocity":
        raise ValueError(
            f"train.curl_loss needs the velocity formulation, got {physics.formulation!r}"
        )
    if denorm_mean is None or denorm_std is None:
        raise ValueError("train.curl_loss needs the dataset's normalisation statistics")
    return curl_rel_l2_2d(
        velocity_in_physical_units(model, pred, denorm_mean, denorm_std),
        velocity_in_physical_units(model, y, denorm_mean, denorm_std),
        diagnostic_domain_lengths(model, physics),
    )


def evaluate_3d(model, test_loader, device):
    """Run a quick L2 evaluation on a held-out set."""
    lploss = LpLoss(size_average=True)
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch_size, S, _, T, _ = x.shape
            x_in = F.pad(x, (0, 0, 0, 5), "constant", 0)
            out = model(x_in).reshape(batch_size, S, S, T + 5)
            out = out[..., :-5]
            total += lploss(out.view(batch_size, S, S, T), y.view(batch_size, S, S, T)).item()
            batches += 1
    if batches == 0:
        return None
    return total / batches


def evaluate_step_ahead(
    model,
    test_loader,
    device,
    grid,
    forcing,
    physics,
    scales,
    denorm_mean=None,
    denorm_std=None,
    use_residual=False,
):
    """Evaluate one-step prediction u_t -> u_{t+1}."""
    lploss = LpLoss(size_average=True)

    model.eval()
    total = 0.0
    batches = 0
    loss_ic_total = 0.0
    loss_cont_total = 0.0
    loss_momx_total = 0.0
    loss_momy_total = 0.0
    loss_cont_rel_total = 0.0
    loss_momx_rel_total = 0.0
    loss_momy_rel_total = 0.0
    div_max_total = 0.0
    mean_vel_total = 0.0
    pred_plot = None
    target_plot = None
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            batch = x.shape[0]
            grid = grid.to(x.device)
            # print("x shape:", x.shape, "y shape:", y.shape, "grid shape:", grid.shape)
            x_in = torch.cat((x, grid.expand(batch, -1, -1, -1)), dim=-1)
            pred = model(x_in)
            if use_residual:
                pred = pred + x
            pred = apply_model_output_constraint(model, pred, x_old=x)
            total += lploss(pred, y).item()
            if pred_plot is None:
                pred_plot = pred.clone()
                target_plot = y.clone()
            div_max_total += compute_model_div_max(
                model,
                pred,
                denorm_mean,
                denorm_std,
                physics,
            )
            mean_vel_total += compute_model_mean_vel(
                model,
                pred,
                denorm_mean,
                denorm_std,
                physics,
            )

            if denorm_mean is not None:
                x_phys = x * (denorm_std + 1e-8) + denorm_mean
                pred_phys = pred * (denorm_std + 1e-8) + denorm_mean
            else:
                x_phys = x
                pred_phys = pred
            if residual_has_scale(physics, forcing):
                (
                    loss_ic,
                    loss_cont,
                    loss_momx,
                    loss_momy,
                    loss_cont_rel,
                    loss_momx_rel,
                    loss_momy_rel,
                ) = compute_physics_losses(x_phys, pred_phys, physics, forcing, scales)
            else:
                # An unforced vorticity run has no residual scale, so the metric is
                # missing rather than wrong. It is only ever a diagnostic here: a run
                # that weights the residual still fails loudly in PINO_loss3d.
                # loss_ic goes missing with the rest because PINO_loss3d computes it in
                # the same call that refuses; it is zero by construction anyway, since
                # the initial condition is the input.
                nan = torch.tensor(math.nan, device=device)
                loss_ic = loss_cont = loss_momx = loss_momy = nan
                loss_cont_rel = loss_momx_rel = loss_momy_rel = nan
            loss_ic_total += loss_ic.item()
            loss_cont_total += loss_cont.item()
            loss_momx_total += loss_momx.item()
            loss_momy_total += loss_momy.item()
            loss_cont_rel_total += loss_cont_rel.item()
            loss_momx_rel_total += loss_momx_rel.item()
            loss_momy_rel_total += loss_momy_rel.item()
            batches += 1
    if batches == 0:
        return None
    return (
        total / batches,
        loss_ic_total / batches,
        loss_cont_total / batches,
        loss_momx_total / batches,
        loss_momy_total / batches,
        loss_cont_rel_total / batches,
        loss_momx_rel_total / batches,
        loss_momy_rel_total / batches,
        div_max_total / batches,
        mean_vel_total / batches,
        pred_plot,
        target_plot,
    )


def _get_base_dataset(ds):
    """Return the underlying dataset (unwrap Subset/DataLoader)."""
    if isinstance(ds, DataLoader):
        ds = ds.dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return ds


def get_fixed_test_pair(
    model, test_source, grid, device, sample_idx=0, t_idx=0, use_residual=False
):
    """Grab a deterministic (x_t, x_{t+1}) pair from the test data without relying on
    the test loader's random timestep selection.
    """
    base_ds = _get_base_dataset(test_source)
    if not hasattr(base_ds, "data"):
        return None, None
    data = base_ds.data
    if sample_idx >= data.shape[0]:
        sample_idx = data.shape[0] - 1
    max_t = data.shape[-1] - 1
    if max_t <= 0:
        return None, None
    t_idx = min(t_idx, max_t - 1)

    sample = data[sample_idx]
    x = sample[..., t_idx].to(device)
    y = sample[..., t_idx + 1].to(device)
    grid_b = grid.unsqueeze(0).to(device)
    x_in = torch.cat((x.unsqueeze(0), grid_b), dim=-1)
    with torch.no_grad():
        pred = model(x_in)
        if pred.dim() == 5:
            pred = pred.squeeze(-2)
        if pred.dim() == 4:
            pred = pred.squeeze(-1)
        if use_residual:
            pred = pred + x.unsqueeze(0)
        pred = apply_model_output_constraint(model, pred, x_old=x.unsqueeze(0))
    return pred, y.unsqueeze(0)


def noise_amplitude(epoch, epochs, noise_std, warmup_frac):
    """Amplitude of the input noise at ``epoch``.

    Zero for the first ``warmup_frac`` of the schedule, then a linear ramp that reaches
    ``noise_std`` at the final epoch. Zero throughout when ``noise_std`` is not positive.

    Args:
        epoch: 0-based epoch index.
        epochs: Total epochs in the schedule.
        noise_std: Standard deviation the ramp reaches at the final epoch.
        warmup_frac: Fraction of the schedule trained without noise.

    Returns:
        The standard deviation of the noise added to this epoch's inputs.
    """
    if noise_std <= 0:
        return 0.0
    t_frac = epoch / max(1, epochs - 1)
    if t_frac < warmup_frac:
        return 0.0
    return noise_std * (t_frac - warmup_frac) / (1.0 - warmup_frac)


class RolloutPairs(Dataset):
    """One-step pairs re-targeted ``steps`` frames ahead, for pushforward training.

    The underlying dataset stores a trajectory as consecutive pairs, so pair ``i`` chains
    into pair ``i + 1`` exactly when the target of the first is the input of the second.
    At a realisation boundary it does not, and a start whose chain would cross one is
    dropped rather than stitching two stretches of flow into one rollout.

    Args:
        pairs: Dataset exposing ``X_data`` and ``y_data`` of shape ``(N, H, W, C)``.
        steps: Length of the rollout. ``1`` reproduces the underlying dataset exactly.
    """

    def __init__(self, pairs, steps):
        self.pairs = pairs
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError(f"steps must be at least 1, got {steps}")
        starts = torch.arange(len(pairs.X_data) - self.steps + 1)
        if self.steps > 1:
            chained = (pairs.y_data[:-1] == pairs.X_data[1:]).flatten(1).all(dim=1)
            for offset in range(self.steps - 1):
                starts = starts[chained[starts + offset]]
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = int(self.starts[idx])
        return self.pairs.X_data[start], self.pairs.y_data[start + self.steps - 1]


def resolve_pushforward_steps(train_config):
    """Detached steps to take before the step the gradient flows through.

    ``rollout_steps`` alone would change the target without changing the input, and
    ``pushforward`` alone would change neither, so the two keys have to agree. Both the
    dataset that supplies the target and the loop that takes the steps read the answer
    here, so a configuration that cannot be honoured is refused before either acts.

    Args:
        train_config: The ``train`` block of the configuration.

    Returns:
        int: ``rollout_steps - 1`` under pushforward training, and 0 otherwise.
    """
    rollout_steps = int(train_config.get("rollout_steps", 1))
    pushforward = bool(train_config.get("pushforward", False))
    if pushforward and rollout_steps < 2:
        raise ValueError(f"pushforward: true needs rollout_steps > 1, got {rollout_steps}")
    if rollout_steps > 1 and not pushforward:
        raise ValueError(f"rollout_steps {rollout_steps} needs pushforward: true")
    return rollout_steps - 1 if pushforward else 0


def model_step(model, x, grid, use_residual):
    """One autoregressive step: grid features, the model, the residual, the projector.

    Args:
        model: The operator being trained.
        x: Batch of states, shape ``(B, H, W, C)``, in normalised units.
        grid: Grid features of shape ``(H, W, G)`` shared by the batch.
        use_residual: Whether the model predicts the increment rather than the state.

    Returns:
        The predicted next state, shape ``(B, H, W, C)``.
    """
    x_in = torch.cat((x, grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1)), dim=-1)
    pred = model(x_in)
    if isinstance(pred, tuple):
        pred = pred[0]
    if use_residual:
        pred = pred + x
    return apply_model_output_constraint(model, pred, x_old=x)


def pushforward_input(model, x, grid, steps, use_residual):
    """Advance ``x`` by ``steps`` model steps with no gradient (the pushforward trick).

    Training the last step from the model's own prediction, rather than from the truth,
    puts the input distribution where an autoregressive rollout actually goes. Detaching
    the earlier steps keeps the backward pass one step deep, so the whole cost is one
    extra forward pass per step.

    Args:
        model: The operator being trained.
        x: Batch of ground-truth states, shape ``(B, H, W, C)``.
        grid: Grid features shared by the batch.
        steps: Number of detached steps to take. ``0`` returns ``x`` itself.
        use_residual: Whether the model predicts the increment rather than the state.

    Returns:
        The state the gradient-carrying step starts from, carrying no autograd graph.
    """
    if steps <= 0:
        return x
    with torch.no_grad():
        for _ in range(steps):
            x = model_step(model, x, grid, use_residual)
    return x


def train_step_ahead(
    model,
    train_loader,
    optimizer,
    scheduler,
    config,
    device,
    grid,
    test_loader=None,
    eval_step=10,
    save_step=1000,
    use_tqdm=True,
    writer=None,
    model_name="fno2d",
    start_ep=0,
    weight_dict=None,
    forcing=None,
    physics=None,
    scales=None,
    denorm_mean=None,
    denorm_std=None,
    use_residual=False,
    resume=False,
):
    """Train on one-step pairs (u_t, u_{t+1})."""
    lploss = LpLoss(size_average=True)
    epochs = config["train"]["epochs"]
    grad_clip = config["train"].get("grad_clip", 0.0)
    # float(): YAML 1.1 reads an exponent without a decimal point ("2e-2") as a string,
    # and a string noise level would abort the run at the first epoch.
    noise_std = float(config["train"].get("noise_std", 0.0))
    warmup_frac = float(config["train"].get("noise_warmup_frac", 0.2))
    pushforward_steps = resolve_pushforward_steps(config["train"])
    patience = config["train"].get("patience", 0)

    if start_ep >= epochs:
        print(f"start_ep ({start_ep}) >= epochs ({epochs}); skipping training loop.")
        return
    if use_tqdm:
        pbar = tqdm(range(start_ep, epochs), dynamic_ncols=True, smoothing=0.1)
    else:
        pbar = range(start_ep, epochs)
    data_weight = weight_dict["data_weight"]
    cont_weight = weight_dict["cont_weight"]
    ic_weight = weight_dict["ic_weight"]
    momx_weight = weight_dict["momx_weight"]
    momy_weight = weight_dict["momy_weight"]
    curl_weight = weight_dict.get("curl_weight", 0.0)
    training_log_path = get_training_log_path(config)
    start_training_log(training_log_path, TRAIN_LOG_COLUMNS, resume=resume)

    # On a resume the best validation loss so far lives in the log; without it the
    # first evaluation of the new job would overwrite _best.pt unconditionally.
    best_loss = best_logged_eval_loss(training_log_path) if resume else torch.inf
    epochs_no_improve = 0
    for ep in pbar:
        model.train()
        rel_l2_loss_total = 0.0
        total_loss_total = 0.0
        loss_ic_total = 0.0
        loss_cont_total = 0.0
        loss_momx_total = 0.0
        loss_momy_total = 0.0
        loss_cont_rel_total = 0.0
        loss_momx_rel_total = 0.0
        loss_momy_rel_total = 0.0
        div_max_total = 0.0
        mean_vel_total = 0.0
        loss_curl_total = 0.0

        batches = 0

        noise_amp = noise_amplitude(ep, epochs, noise_std, warmup_frac)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if noise_amp > 0:
                x = x + noise_amp * torch.randn_like(x)
            # Pushforward: the gradient-carrying step starts from the model's own
            # prediction, and `y` is already the target that many frames ahead, so
            # the data loss and the PDE residual below both see the stepped pair.
            x = pushforward_input(model, x, grid, pushforward_steps, use_residual)
            pred = model_step(model, x, grid, use_residual)
            data_loss = lploss(pred, y)
            div_max_total += compute_model_div_max(
                model,
                pred,
                denorm_mean,
                denorm_std,
                physics,
            )
            mean_vel_total += compute_model_mean_vel(
                model,
                pred,
                denorm_mean,
                denorm_std,
                physics,
            )

            # Curl-matching term: the velocity loss weighted by wavenumber, so the band
            # above the model's spectral cutoff is visible to it (#42, #28).
            if curl_weight != 0.0:
                curl_loss = compute_curl_loss(model, pred, y, denorm_mean, denorm_std, physics)
            else:
                curl_loss = torch.tensor(0.0, device=device)

            # PINO loss: rollout 2D model from u0 to get trajectory, then PDE/IC loss.
            use_pino = (
                (cont_weight != 0.0)
                or (ic_weight != 0.0)
                or (momx_weight != 0.0)
                or (momy_weight != 0.0)
            )
            if use_pino:
                if denorm_mean is not None:
                    x_phys = x * (denorm_std + 1e-8) + denorm_mean
                    pred_phys = pred * (denorm_std + 1e-8) + denorm_mean
                else:
                    x_phys = x
                    pred_phys = pred
                (
                    loss_ic,
                    loss_cont,
                    loss_momx,
                    loss_momy,
                    loss_cont_rel,
                    loss_momx_rel,
                    loss_momy_rel,
                ) = compute_physics_losses(x_phys, pred_phys, physics, forcing, scales)
            else:
                loss_ic = torch.tensor(0.0, device=device)
                loss_cont = torch.tensor(0.0, device=device)
                loss_momx = torch.tensor(0.0, device=device)
                loss_momy = torch.tensor(0.0, device=device)
                loss_cont_rel = torch.tensor(0.0, device=device)
                loss_momx_rel = torch.tensor(0.0, device=device)
                loss_momy_rel = torch.tensor(0.0, device=device)

            # Optimise with relative PDE residuals; keep absolute metrics for logging.
            loss = (
                data_loss * data_weight
                + loss_cont_rel * cont_weight
                + loss_ic * ic_weight
                + loss_momx_rel * momx_weight
                + loss_momy_rel * momy_weight
                + curl_loss * curl_weight
            )

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            rel_l2_loss_total += data_loss.item()
            total_loss_total += loss.item()
            loss_ic_total += loss_ic.item() if isinstance(loss_ic, torch.Tensor) else loss_ic
            loss_cont_total += (
                loss_cont.item() if isinstance(loss_cont, torch.Tensor) else loss_cont
            )
            loss_momx_total += (
                loss_momx.item() if isinstance(loss_momx, torch.Tensor) else loss_momx
            )
            loss_momy_total += (
                loss_momy.item() if isinstance(loss_momy, torch.Tensor) else loss_momy
            )
            loss_cont_rel_total += (
                loss_cont_rel.item() if isinstance(loss_cont_rel, torch.Tensor) else loss_cont_rel
            )
            loss_momx_rel_total += (
                loss_momx_rel.item() if isinstance(loss_momx_rel, torch.Tensor) else loss_momx_rel
            )
            loss_momy_rel_total += (
                loss_momy_rel.item() if isinstance(loss_momy_rel, torch.Tensor) else loss_momy_rel
            )
            loss_curl_total += curl_loss.item()
            batches += 1
        scheduler.step()
        rel_l2_loss_avg = rel_l2_loss_total / max(1, batches)
        total_loss_avg = total_loss_total / max(1, batches)
        loss_ic_avg = loss_ic_total / max(1, batches)
        loss_cont_avg = loss_cont_total / max(1, batches)
        loss_momx_avg = loss_momx_total / max(1, batches)
        loss_momy_avg = loss_momy_total / max(1, batches)
        loss_cont_rel_avg = loss_cont_rel_total / max(1, batches)
        loss_momx_rel_avg = loss_momx_rel_total / max(1, batches)
        loss_momy_rel_avg = loss_momy_rel_total / max(1, batches)
        div_max_avg = div_max_total / max(1, batches)
        mean_vel_avg = mean_vel_total / max(1, batches)
        loss_curl_avg = loss_curl_total / max(1, batches)
        print(
            f"Epoch {ep + 1}/{epochs}, train total: {total_loss_avg:.6f}, train L2: {rel_l2_loss_avg:.6f}, train IC: {loss_ic_avg:.6f}, "
            f"train PDE abs: {loss_cont_avg:.6f}, train momx abs: {loss_momx_avg:.6f}, train momy abs: {loss_momy_avg:.6f}, "
            f"train PDE rel: {loss_cont_rel_avg:.6f}, train momx rel: {loss_momx_rel_avg:.6f}, train momy rel: {loss_momy_rel_avg:.6f}, "
            f"train div max: {div_max_avg:.6e}, train mean vel: {mean_vel_avg:.6e}, "
            f"train curl rel: {loss_curl_avg:.6f}"
        )
        if writer is not None:
            writer.add_scalar("train/rel_l2", rel_l2_loss_avg, ep + 1)
            writer.add_scalar("train/ic", loss_ic_avg, ep + 1)
            writer.add_scalar("train/pde_abs", loss_cont_avg, ep + 1)
            writer.add_scalar("train/momx_abs", loss_momx_avg, ep + 1)
            writer.add_scalar("train/momy_abs", loss_momy_avg, ep + 1)
            writer.add_scalar("train/pde_rel", loss_cont_rel_avg, ep + 1)
            writer.add_scalar("train/momx_rel", loss_momx_rel_avg, ep + 1)
            writer.add_scalar("train/momy_rel", loss_momy_rel_avg, ep + 1)
            writer.add_scalar("train/div_max", div_max_avg, ep + 1)
            writer.add_scalar("train/mean_vel", mean_vel_avg, ep + 1)
        if use_tqdm:
            pbar.set_description(
                f"Train total/L2: {total_loss_avg:.3e}/{rel_l2_loss_avg:.3e}, IC: {loss_ic_avg:.6f}, "
                f"PDE abs/rel: {loss_cont_avg:.3e}/{loss_cont_rel_avg:.3e}, "
                f"momx abs/rel: {loss_momx_avg:.3e}/{loss_momx_rel_avg:.3e}, "
                f"momy abs/rel: {loss_momy_avg:.3e}/{loss_momy_rel_avg:.3e}, "
                f"div max: {div_max_avg:.3e}, mean vel: {mean_vel_avg:.3e}"
            )

        train_metrics = {
            "epoch": ep + 1,
            "train_l2": rel_l2_loss_avg,
            "train_cont_rel": loss_cont_rel_avg,
            "train_momx_rel": loss_momx_rel_avg,
            "train_momy_rel": loss_momy_rel_avg,
            "train_cont_abs": loss_cont_avg,
            "train_momx_abs": loss_momx_avg,
            "train_momy_abs": loss_momy_avg,
            "train_div_max": div_max_avg,
            "train_mean_vel": mean_vel_avg,
            "train_curl_rel": loss_curl_avg,
        }
        eval_metrics = {
            "eval_l2": math.nan,
            "eval_cont_rel": math.nan,
            "eval_momx_rel": math.nan,
            "eval_momy_rel": math.nan,
            "eval_cont_abs": math.nan,
            "eval_momx_abs": math.nan,
            "eval_momy_abs": math.nan,
            "eval_div_max": math.nan,
            "eval_mean_vel": math.nan,
        }

        # The final epoch is always evaluated, so the last trained weights are
        # both checkpointed and eligible for _best.pt.
        if (ep % eval_step == 0 or ep == epochs - 1) and test_loader is not None:
            (
                test_l2,
                loss_ic_avg,
                loss_cont_avg,
                loss_momx_avg,
                loss_momy_avg,
                loss_cont_rel_avg,
                loss_momx_rel_avg,
                loss_momy_rel_avg,
                eval_div_max,
                eval_mean_vel,
                pred_plot,
                target_plot,
            ) = evaluate_step_ahead(
                model,
                test_loader,
                device,
                grid,
                forcing,
                physics,
                scales,
                denorm_mean=denorm_mean,
                denorm_std=denorm_std,
                use_residual=use_residual,
            )
            print(f"Random test split relative L2: {test_l2:.6f}")
            if writer is not None:
                writer.add_scalar("eval/test_l2", test_l2, ep + 1)
                writer.add_scalar("eval/test_ic", loss_ic_avg, ep + 1)
                writer.add_scalar("eval/test_pde_abs", loss_cont_avg, ep + 1)
                writer.add_scalar("eval/test_momx_abs", loss_momx_avg, ep + 1)
                writer.add_scalar("eval/test_momy_abs", loss_momy_avg, ep + 1)
                writer.add_scalar("eval/test_pde_rel", loss_cont_rel_avg, ep + 1)
                writer.add_scalar("eval/test_momx_rel", loss_momx_rel_avg, ep + 1)
                writer.add_scalar("eval/test_momy_rel", loss_momy_rel_avg, ep + 1)
                writer.add_scalar("eval/div_max", eval_div_max, ep + 1)
                writer.add_scalar("eval/mean_vel", eval_mean_vel, ep + 1)

            eval_metrics.update(
                {
                    "eval_l2": test_l2,
                    "eval_cont_rel": loss_cont_rel_avg,
                    "eval_momx_rel": loss_momx_rel_avg,
                    "eval_momy_rel": loss_momy_rel_avg,
                    "eval_cont_abs": loss_cont_avg,
                    "eval_momx_abs": loss_momx_avg,
                    "eval_momy_abs": loss_momy_avg,
                    "eval_div_max": eval_div_max,
                    "eval_mean_vel": eval_mean_vel,
                }
            )

            save_checkpoint(
                config["train"]["save_dir"],
                config["train"]["save_name"],
                model,
                ep,
                optimizer,
                scheduler,
            )

            # Guarded selection: with noise injection on, the clean validation loss
            # rises while the amplitude ramps, so its minimum sits at an epoch trained
            # with little or no noise and _best.pt would silently discard the
            # stabilisation the run was for (#42). Only an epoch at the full amplitude
            # is eligible; under the ramp above that is the final epoch, which is
            # always evaluated.
            eligible_for_best = noise_amp >= noise_std
            if eligible_for_best and test_l2 < best_loss:
                best_loss = test_l2
                epochs_no_improve = 0
                best_name = config["train"]["save_name"].replace(".pt", "_best.pt")
                save_checkpoint(
                    config["train"]["save_dir"], best_name, model, ep, optimizer, scheduler
                )
            elif eligible_for_best:
                # Only an epoch that could have been selected counts against the patience,
                # or a noise run would early-stop on the schedule rather than on the loss.
                epochs_no_improve += 1

            if patience > 0 and epochs_no_improve >= patience:
                append_training_log(training_log_path, {**train_metrics, **eval_metrics})
                print(
                    f"Early stopping: no improvement for {patience} eval cycles "
                    f"(best L2: {best_loss:.6f})"
                )
                break

        append_training_log(training_log_path, {**train_metrics, **eval_metrics})


def build_synthetic_dataset(data_config, n_samples, step_ahead=False):
    """Create a random dataset that mimics NSLoader/NSLoader2D output."""
    sub = data_config.get("sub", 1)
    sub_t = data_config.get("sub_t", 1)
    nx = data_config.get("nx", 128)
    nt = data_config.get("nt", 64)
    time_scale = data_config.get("time_interval", 1.0)
    S = nx // sub
    T = 64  # channel size
    C = 3  # time steps

    if step_ahead:
        data = torch.rand(n_samples, S, S, C, T)

        class SyntheticStepDataset(Dataset):
            def __init__(self, arr):
                self.data = arr
                self.max_t = arr.shape[-1] - 1
                self.T = int(nt * time_scale) // sub_t + 1

            def __len__(self):
                return self.data.shape[0]

            def __getitem__(self, idx):
                sample = self.data[idx]
                t = torch.randint(0, self.max_t, ()).item()
                return sample[..., t], sample[..., t + 1]

        return SyntheticStepDataset(data), (S, S), 1


def train_2d(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_config = config["data"]
    physics = physics_from_config(data_config)

    # prepare dataloader for training with data (real or synthetic)
    if args.synthetic_samples > 0:
        full_dataset, S_data, _ = build_synthetic_dataset(
            data_config, args.synthetic_samples, step_ahead=True
        )
    else:
        full_dataset = NSLoader2D(
            datapath=data_config["datapath"],
            state="train",
            train=True,
            # Reuse the statistics if the dataset ships them, and write them only
            # when it does not: recomputing per run gives every seed the same
            # numbers, and concurrent runs would race to rewrite one shared file.
            normalizer_path=data_config.get("normalizer_path"),
            save_normalizer_path=data_config.get("normalizer_path"),
            velocity_channels=data_config.get("velocity_channels", (0, 1)),
            filename=data_config.get("filename", "kolmogorov_dataset.npz"),
        )
        S_data = full_dataset.S

    # Normalisation statistics for denormalising before PDE loss.
    if hasattr(full_dataset, "mean") and hasattr(full_dataset, "std"):
        denorm_mean = full_dataset.mean.permute(1, 2, 0).to(device)  # (1, 1, C)
        denorm_std = full_dataset.std.permute(1, 2, 0).to(device)  # (1, 1, C)
    else:
        denorm_mean = None
        denorm_std = None

    # split dataset into training and validation sets by test_ratio
    if args.test_ratio > 0:
        test_size = max(1, int(len(full_dataset) * args.test_ratio))
        if len(full_dataset) - test_size <= 0:
            raise ValueError("test_ratio is too large; no samples left for training.")
        train_size = len(full_dataset) - test_size
        train_set, test_set = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        test_set.train = False  # set test set to not train
        test_loader = DataLoader(test_set, batch_size=config["train"]["batchsize"], shuffle=False)
    else:
        train_set = full_dataset
        test_set = NSLoader2D(
            datapath=data_config["datapath"],
            state="val",
            train=False,
            normalizer_path=data_config.get("normalizer_path", None),
            velocity_channels=data_config.get("velocity_channels", (0, 1)),
            filename=data_config.get("filename", "kolmogorov_dataset.npz"),
        )
        test_loader = DataLoader(test_set, batch_size=config["train"]["batchsize"], shuffle=False)

    # Pushforward training re-targets each pair `rollout_steps` frames ahead. Only the
    # training loader is re-targeted: validation stays one-step, so `_best.pt` is
    # selected on the same quantity as every other run in the campaign.
    rollout_steps = resolve_pushforward_steps(config["train"]) + 1
    if rollout_steps > 1:
        if not hasattr(train_set, "X_data"):
            raise ValueError(
                "pushforward training needs a dataset of consecutive pairs; "
                "a randomly split training set cannot supply one"
            )
        train_set = RolloutPairs(train_set, rollout_steps)
        print(f"Pushforward: {len(train_set)} {rollout_steps}-step starts")

    train_loader = DataLoader(
        train_set,
        batch_size=config["train"]["batchsize"],
        shuffle=data_config["shuffle"],
        **loader_kwargs(args.seed),
    )

    # loader for initial condition and PDE loss computation
    # PINO loss: parameters and IC dataloader (take the first time step of each realization) (see neuraloperator/physics_informed train_pino.py)
    data_weight = config["train"].get("xy_loss", 1.0)
    cont_weight = config["train"].get("f_loss", 0.0)
    ic_weight = config["train"].get("ic_loss", 0.0)
    momx_weight = config["train"].get("momx_loss", 0.0)
    momy_weight = config["train"].get("momy_loss", 0.0)
    curl_weight = config["train"].get("curl_loss", 0.0)
    weight_dict = {
        "data_weight": data_weight,
        "cont_weight": cont_weight,
        "ic_weight": ic_weight,
        "momx_weight": momx_weight,
        "momy_weight": momy_weight,
        "curl_weight": curl_weight,
    }
    # Forcing at the resolution the residual is evaluated on: the data resolution for
    # the velocity formulation, the refined one for the vorticity formulation (PINO).
    forcing = build_forcing_for_data(physics, S_data, device=device)
    use_residual = config["model"].get("residual", False)

    # Denominators of the relative residuals, measured once on ground-truth pairs
    # from the first (unshuffled) validation batch, so that they are a property of
    # the data and identical across model variants. The evaluation script measures
    # its own from the test trajectories, so the residuals in this log and those in
    # the evaluation CSV carry different denominators; both are recorded.
    scales = None
    if physics.formulation == "velocity":
        val_x, val_y = next(iter(test_loader))
        val_x, val_y = val_x.to(device), val_y.to(device)
        if denorm_mean is not None:
            val_x = val_x * (denorm_std + 1e-8) + denorm_mean
            val_y = val_y * (denorm_std + 1e-8) + denorm_mean
        scales = residual_scales_from_truth(
            torch.stack([val_x, val_y], dim=-1).permute(0, 3, 1, 2, 4), physics, physics.dt
        )
        print(f"Residual reference scales (ground truth): {scales}")

    # create model
    print("device: ", device)
    model_cfg = config["model"]
    check_model_channels(physics, model_cfg.get("out_dim", 1))
    model_name = model_cfg.get("name", "fno2d").lower()
    output_constraint_cfg = model_cfg.get("output_constraint", {})
    output_constraint_enabled = bool(output_constraint_cfg.get("enabled", False))

    if output_constraint_enabled and model_name != "fno2d":
        raise ValueError('output_constraint is only supported for model.name == "fno2d"')
    if output_constraint_enabled and (
        not hasattr(full_dataset, "mean") or not hasattr(full_dataset, "std")
    ):
        raise ValueError("output_constraint requires dataset normalization statistics")

    if model_name == "fno2d":
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
            #   pad_ratio=model_cfg.get('pad_ratio', [0., 0.])
        ).to(device)
    else:
        raise ValueError(f"Model {model_name} not supported")
    if output_constraint_enabled:
        model.set_output_normalizer(full_dataset.mean, full_dataset.std)
        model.set_energy_forcing(forcing.squeeze(-1))  # no-op if energy_balance not enabled
    print("model structure: ", model)
    n_parameters = count_parameters(model)

    # create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["train"]["milestones"],
        gamma=config["train"]["scheduler_gamma"],
    )

    start_ep = 0
    eval_step = config["train"].get("eval_step", 10)

    ckpt_path = os.path.join(config["train"]["save_dir"], config["train"]["save_name"])
    best_ckpt_path = ckpt_path.replace(".pt", "_best.pt")
    training_log_path = get_training_log_path(config)
    manifest_path = os.path.join(config["train"]["save_dir"], "run_manifest.json")

    resumed = False
    if args.resume_training:
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            parsed_ep = ckpt["epoch"]
            model.load_state_dict(ckpt["model"])
            if ckpt.get("optim") is not None:
                optimizer.load_state_dict(ckpt["optim"])
            if ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
                sched_epoch = scheduler.state_dict().get("last_epoch", -1) + 1
                parsed_ep = max(parsed_ep, sched_epoch)
            start_ep = max(parsed_ep, 0)
            resumed = True
            print(f"Weights loaded from {ckpt_path}, resuming at epoch {start_ep + 1}")
        else:
            print("resume_training requested but no checkpoint found; starting from scratch.")

    # A run that did not resume is a fresh one, and a fresh run may not silently
    # replace the checkpoints or the log of an earlier one.
    check_run_directory(
        [ckpt_path, best_ckpt_path, training_log_path, manifest_path],
        overwrite=args.overwrite,
        resume=resumed,
    )

    save_dir = config["train"]["save_dir"] if torch.cuda.is_available() else "saved_models"
    tensorboard_dir = config["train"].get("tensorboard_dir")
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    grid = torch2dgrid_2d(
        S_data[0], S_data[1], form=config["data"]["grid_form"], device=device, dtype=torch.float32
    )

    write_run_manifest(
        config["train"]["save_dir"],
        config,
        args.seed,
        [ckpt_path, best_ckpt_path],
        extra={
            "training_log": training_log_path,
            "resumed": resumed,
            "residual_scales": None if scales is None else vars(scales),
            "n_parameters": n_parameters,
        },
    )
    print(f"Run manifest written to {manifest_path}")

    train_step_ahead(
        model,
        train_loader,
        optimizer,
        scheduler,
        config,
        device,
        grid,
        test_loader=test_loader,
        writer=writer,
        model_name=model_name,
        start_ep=start_ep,
        weight_dict=weight_dict,
        forcing=forcing,
        physics=physics,
        scales=scales,
        eval_step=eval_step,
        denorm_mean=denorm_mean,
        denorm_std=denorm_std,
        use_residual=use_residual,
        resume=resumed,
    )

    if test_loader is not None:
        (
            test_l2,
            test_ic_abs,
            test_pde_abs,
            test_momx_abs,
            test_momy_abs,
            test_pde_rel,
            test_momx_rel,
            test_momy_rel,
            test_div_max,
            test_mean_vel,
            _,
            _,
        ) = evaluate_step_ahead(
            model,
            test_loader,
            device,
            grid,
            forcing,
            physics,
            scales,
            denorm_mean=denorm_mean,
            denorm_std=denorm_std,
            use_residual=use_residual,
        )
        print(f"Random test split relative L2: {test_l2:.6f}")
        if writer is not None:
            writer.add_scalar("eval/test_l2", test_l2, config["train"]["epochs"])
            writer.add_scalar("eval/test_ic", test_ic_abs, config["train"]["epochs"])
            writer.add_scalar("eval/test_pde_abs", test_pde_abs, config["train"]["epochs"])
            writer.add_scalar("eval/test_momx_abs", test_momx_abs, config["train"]["epochs"])
            writer.add_scalar("eval/test_momy_abs", test_momy_abs, config["train"]["epochs"])
            writer.add_scalar("eval/test_pde_rel", test_pde_rel, config["train"]["epochs"])
            writer.add_scalar("eval/test_momx_rel", test_momx_rel, config["train"]["epochs"])
            writer.add_scalar("eval/test_momy_rel", test_momy_rel, config["train"]["epochs"])
            writer.add_scalar("eval/div_max", test_div_max, config["train"]["epochs"])
            writer.add_scalar("eval/mean_vel", test_mean_vel, config["train"]["epochs"])
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parse options
    parser = ArgumentParser(description="Basic paser")
    parser.add_argument("--config_path", type=str, help="Path to the configuration file")
    parser.add_argument("--log", action="store_true", help="Turn on the wandb")
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.0,
        help="Hold out this fraction of samples for a random test split",
    )
    parser.add_argument(
        "--seed",
        "--test_seed",
        dest="seed",
        type=int,
        default=42,
        help="Seed for weight initialisation, shuffling and the checkpoint name",
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=0,
        help="Use random synthetic data with this many samples to sanity-check the 3D pipeline",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="Resume training from the last checkpoint"
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Specific checkpoint filename to resume from (in save_dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to overwrite the checkpoints and log of an earlier one",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="Cap the configured number of epochs (for smoke runs)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override train.save_dir, so one config can serve a whole seed campaign",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    config_file = args.config_path
    with open(config_file) as stream:
        config = yaml.load(stream, yaml.FullLoader)
        config["train"]["save_name"] = config["train"]["save_name"].replace(
            ".pt", f"_seed{args.seed}.pt"
        )
        if args.max_epochs is not None:
            config["train"]["epochs"] = min(config["train"]["epochs"], args.max_epochs)
        if args.save_dir is not None:
            config["train"]["save_dir"] = args.save_dir

    train_2d(args, config)
