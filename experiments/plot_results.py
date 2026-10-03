import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

FNO_COLOUR = "#0072B2"
FNOC_COLOUR = "#56B4E9"
PINO_COLOUR = "#D55E00"
PINO_CONT_COLOUR = "#882255"
PINO_TTO_COLOUR = "#009E73"
TRUTH_COLOUR = "#000000"
COMPONENT_COLOURS = {
    "cont": "#009E73",
    "momx": "#0072B2",
    "momy": "#CC79A7",
}
MODEL_LINESTYLES = {
    "FNO": "-",
    "FNOC": ":",
    "PINO": "--",
    "PINO-C": (0, (3, 1, 1, 1)),
    "PINO+TTO": "-.",
}
LOG_FLOOR = 1e-12


def configure_matplotlib():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "lines.linewidth": 2.0,
        }
    )


def find_single_file(root_dir, pattern):
    matches = sorted(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} found under {root_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous match for {pattern!r} under {root_dir}: {matches}")
    return matches[0]


def build_artifact_pattern(stem, seed, artifact_name, file_suffix=""):
    seed_tag = f"_seed{seed}_{artifact_name}{file_suffix}"
    if stem:
        return f"**/{stem}{seed_tag}"
    return f"**/*{seed_tag}"


def mark_axis_unavailable(ax, title, ylabel=None):
    ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_box_aspect(1)
    ax.text(0.5, 0.5, "Results unavailable", ha="center", va="center", transform=ax.transAxes)


def spectra_files_for_suffix(paths, file_suffix=""):
    """Keep the spectra files whose step number is followed by exactly this suffix.

    The glob that finds them is ``*_spectra_t*{suffix}.npz``, and with an empty
    suffix the wildcard after the step number also matches a variant suffix. A
    long-horizon or other-split evaluation writing into the same directory would
    then supply the default figure's final spectra, because its step number is
    larger.

    Args:
        paths: Candidate paths or file names.
        file_suffix: The variant suffix these spectra must carry, "" for none.

    Returns:
        list: The subset of ``paths`` that match exactly, in the order given.
    """
    exact = re.compile(rf"_spectra_t\d+{re.escape(file_suffix or '')}\.npz$")
    return [path for path in paths if exact.search(os.path.basename(path))]


def load_run_artifacts(results_dir, seed, file_suffix="", require_training=True, stem=None):
    training_csv = None
    training_df = None
    if require_training:
        try:
            training_csv = find_single_file(
                results_dir,
                build_artifact_pattern(stem, seed, "training_log.csv"),
            )
        except (FileNotFoundError, RuntimeError):
            # Eval-only runs may reuse an existing checkpoint directory without
            # producing a stem-matched training log. Fall back to the unique
            # training log in the directory if available.
            training_csv = find_single_file(
                results_dir,
                build_artifact_pattern(None, seed, "training_log.csv"),
            )
    per_sample_npz = find_single_file(
        results_dir,
        os.path.join(
            "**/evaluation_metrics",
            os.path.basename(
                build_artifact_pattern(stem, seed, "per_sample_metrics", file_suffix=file_suffix)
                + ".npz"
            ),
        ),
    )
    vorticity_npz = find_single_file(
        results_dir,
        os.path.join(
            "**/saved_plots",
            os.path.basename(
                build_artifact_pattern(stem, seed, "vorticity_sample0", file_suffix=file_suffix)
                + ".npz"
            ),
        ),
    )

    all_spectra = sorted(
        glob.glob(
            os.path.join(
                results_dir,
                os.path.join(
                    "**/saved_plots",
                    f"{stem}_seed{seed}_spectra_t*{file_suffix}.npz"
                    if stem
                    else f"*_seed{seed}_spectra_t*{file_suffix}.npz",
                ),
            ),
            recursive=True,
        )
    )
    all_spectra = spectra_files_for_suffix(all_spectra, file_suffix)
    if not all_spectra:
        raise FileNotFoundError(f"No saved spectra found under {results_dir}")

    def extract_step(path):
        stem = Path(path).stem
        match = re.search(r"_spectra_t(\d+)", stem)
        if match is None:
            raise ValueError(f"Could not extract spectra step from {path}")
        return int(match.group(1))

    spectra_t1 = [p for p in all_spectra if extract_step(p) == 1]
    if not spectra_t1:
        raise FileNotFoundError(f"No one-step spectra file found under {results_dir}")
    one_step_spectra_npz = spectra_t1[0]

    final_spectra_npz = max(all_spectra, key=extract_step)

    if require_training:
        training_df = pd.read_csv(training_csv)
        training_df = training_df.drop_duplicates(subset="epoch", keep="last").sort_values("epoch")

    return {
        "training": training_df,
        "per_sample": np.load(per_sample_npz),
        "vorticity": np.load(vorticity_npz),
        "spectra_t1": np.load(one_step_spectra_npz),
        "spectra_final": np.load(final_spectra_npz),
        "training_csv_path": training_csv,
        "per_sample_path": per_sample_npz,
    }


def maybe_load_run_artifacts(results_dir, seed, label, stem=None):
    if not results_dir:
        print(f"{label} results not provided; corresponding panels will be blank.")
        return None
    try:
        return load_run_artifacts(results_dir, seed, stem=stem)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{label} results unavailable: {exc}")
        return None


def maybe_load_eval_artifacts(results_dir, seed, label, file_suffix="", stem=None):
    if not results_dir:
        print(f"{label} results not provided; corresponding panels will be blank.")
        return None
    try:
        return load_run_artifacts(
            results_dir,
            seed,
            file_suffix=file_suffix,
            require_training=False,
            stem=stem,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{label} results unavailable: {exc}")
        return None


def save_figure(fig, output_dir, name, use_tight_layout=True):
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    if use_tight_layout:
        fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figures to {pdf_path} and {png_path}")


def floor_for_log(values, floor=LOG_FLOOR):
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isnan(arr), np.nan, np.maximum(arr, floor))
    return arr


def mean_and_std(samples):
    arr = np.asarray(samples, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready training and rollout figures."
    )
    parser.add_argument(
        "--fno_results", type=str, default=None, help="Path to the FNO save directory"
    )
    parser.add_argument(
        "--fnoc_results",
        type=str,
        default=None,
        help="Path to the FNO + divergence-constraint save directory",
    )
    parser.add_argument(
        "--pino_results", type=str, default=None, help="Path to the PINO save directory"
    )
    parser.add_argument(
        "--pino_cont_results",
        type=str,
        default=None,
        help="Path to the continuity-only PINO save directory; included only when provided",
    )
    parser.add_argument(
        "--pino_tto_results",
        type=str,
        default=None,
        help="Path to the PINO+TTO save directory; defaults to --pino_results if omitted",
    )
    parser.add_argument(
        "--include_pino_tto",
        action="store_true",
        help="Include PINO+TTO artefacts in rollout, spectra, and vorticity plots",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used in filenames")
    parser.add_argument(
        "--fno_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to disambiguate FNO artifacts",
    )
    parser.add_argument(
        "--fnoc_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to disambiguate FNOC artifacts",
    )
    parser.add_argument(
        "--pino_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to disambiguate PINO artifacts",
    )
    parser.add_argument(
        "--pino_cont_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to disambiguate continuity-only PINO artifacts",
    )
    parser.add_argument(
        "--pino_tto_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to disambiguate PINO+TTO artifacts",
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures", help="Directory for output PDFs"
    )
    return parser


def get_vorticity_column_titles(include_pino_tto=False, include_pino_cont=False):
    column_titles = ["FNO", "FNOC", "PINO"]
    if include_pino_cont:
        column_titles.append("PINO-C")
    if include_pino_tto:
        column_titles.append("PINO+TTO")
    column_titles.append("Ground truth")
    return column_titles


def mean_and_quantiles(samples, lower_q=0.05, upper_q=0.95):
    arr = np.asarray(samples, dtype=float)
    mean = arr.mean(axis=0)
    lower = np.quantile(arr, lower_q, axis=0)
    upper = np.quantile(arr, upper_q, axis=0)
    return mean, lower, upper


def median_and_quantiles(samples, lower_q=0.05, upper_q=0.95):
    arr = np.asarray(samples, dtype=float)
    median = np.median(arr, axis=0)
    lower = np.quantile(arr, lower_q, axis=0)
    upper = np.quantile(arr, upper_q, axis=0)
    return median, lower, upper


def plot_with_band(ax, x, mean, std, colour, label, linestyle="-"):
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    lower = mean - std
    upper = mean + std
    ax.plot(x, mean, color=colour, label=label, linestyle=linestyle)
    ax.fill_between(x, lower, upper, color=colour, alpha=0.2)


def plot_interval(ax, x, mean, lower, upper, colour, label, linestyle="-"):
    ax.plot(x, mean, color=colour, label=label, linestyle=linestyle)
    ax.fill_between(x, lower, upper, color=colour, alpha=0.2)


def plot_line(ax, x, values, colour, label, linestyle="-"):
    values = np.asarray(values, dtype=float)
    ax.plot(x, values, color=colour, label=label, linestyle=linestyle)


def plot_log_with_band(ax, x, mean, std, colour, label, linestyle="-"):
    mean = floor_for_log(mean)
    std = np.asarray(std, dtype=float)
    lower = floor_for_log(mean - std)
    upper = floor_for_log(mean + std)
    ax.plot(x, mean, color=colour, label=label, linestyle=linestyle)
    ax.fill_between(x, lower, upper, color=colour, alpha=0.2)


def plot_log_interval(ax, x, mean, lower, upper, colour, label, linestyle="-"):
    mean = floor_for_log(mean)
    lower = floor_for_log(np.minimum(lower, mean))
    upper = floor_for_log(np.maximum(upper, mean))
    ax.plot(x, mean, color=colour, label=label, linestyle=linestyle)
    ax.fill_between(x, lower, upper, color=colour, alpha=0.2)


def plot_log_line(ax, x, values, colour, label, linestyle="-"):
    values = floor_for_log(values)
    ax.plot(x, values, color=colour, label=label, linestyle=linestyle)


def apply_linear_ticks(ax, nbins=5):
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
    ax.tick_params(axis="y", which="major", labelleft=True, pad=4)


def apply_log_ticks(ax):
    ymin, ymax = ax.get_ylim()
    if ymin <= 0 or not np.isfinite(ymin) or not np.isfinite(ymax):
        return
    exp_min = int(np.floor(np.log10(ymin)))
    exp_max = int(np.ceil(np.log10(ymax)))
    ticks = [10.0**e for e in range(exp_min, exp_max + 1)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.tick_params(axis="y", which="major", labelleft=True, pad=4)


def make_plot_a1(fno, fnoc, pino, pino_cont, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    if fno is not None:
        fno_df = fno["training"]
        ax.plot(
            fno_df["epoch"], floor_for_log(fno_df["train_l2"]), color=FNO_COLOUR, label="FNO train"
        )
        fno_eval = floor_for_log(fno_df["eval_l2"].to_numpy())
        fno_eval_epochs = fno_df["epoch"].to_numpy()
        valid = np.isfinite(fno_eval)
        if np.any(valid):
            ax.plot(
                fno_eval_epochs[valid],
                fno_eval[valid],
                color=FNO_COLOUR,
                linestyle="--",
                label="FNO val",
            )
    if fnoc is not None:
        fnoc_df = fnoc["training"]
        ax.plot(
            fnoc_df["epoch"],
            floor_for_log(fnoc_df["train_l2"]),
            color=FNOC_COLOUR,
            label="FNOC train",
        )
        fnoc_eval = floor_for_log(fnoc_df["eval_l2"].to_numpy())
        fnoc_eval_epochs = fnoc_df["epoch"].to_numpy()
        valid = np.isfinite(fnoc_eval)
        if np.any(valid):
            ax.plot(
                fnoc_eval_epochs[valid],
                fnoc_eval[valid],
                color=FNOC_COLOUR,
                linestyle="--",
                label="FNOC val",
            )
    if pino is not None:
        pino_df = pino["training"]
        ax.plot(
            pino_df["epoch"],
            floor_for_log(pino_df["train_l2"]),
            color=PINO_COLOUR,
            label="PINO train",
        )
        pino_eval = floor_for_log(pino_df["eval_l2"].to_numpy())
        pino_eval_epochs = pino_df["epoch"].to_numpy()
        valid = np.isfinite(pino_eval)
        if np.any(valid):
            ax.plot(
                pino_eval_epochs[valid],
                pino_eval[valid],
                color=PINO_COLOUR,
                linestyle="--",
                label="PINO val",
            )
    if pino_cont is not None:
        pino_cont_df = pino_cont["training"]
        ax.plot(
            pino_cont_df["epoch"],
            floor_for_log(pino_cont_df["train_l2"]),
            color=PINO_CONT_COLOUR,
            label="PINO-C train",
        )
        pino_cont_eval = floor_for_log(pino_cont_df["eval_l2"].to_numpy())
        pino_cont_eval_epochs = pino_cont_df["epoch"].to_numpy()
        valid = np.isfinite(pino_cont_eval)
        if np.any(valid):
            ax.plot(
                pino_cont_eval_epochs[valid],
                pino_cont_eval[valid],
                color=PINO_CONT_COLOUR,
                linestyle="--",
                label="PINO-C val",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative L2")
    ax.set_title("A1. Training and validation L2")
    if fno is None or fnoc is None or pino is None:
        missing = []
        if fno is None:
            missing.append("FNO")
        if fnoc is None:
            missing.append("FNOC")
        if pino is None:
            missing.append("PINO")
        ax.text(
            0.02,
            0.02,
            f"Missing: {', '.join(missing)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )
    if ax.lines:
        ax.legend(ncol=2, frameon=False)
    save_figure(fig, output_dir, "A1_l2_vs_epoch")


def make_pde_epoch_plot(
    fno,
    fnoc,
    pino,
    pino_cont,
    prefix,
    metric_kind,
    title,
    output_dir,
    filename,
):
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    components = [
        ("cont", "Continuity residual"),
        ("momx", "x-momentum residual"),
        ("momy", "y-momentum residual"),
    ]
    metric_suffix = "abs" if metric_kind == "abs" else "rel"

    for ax, (component, ylabel) in zip(axes, components, strict=False):
        if fno is not None:
            fno_df = fno["training"]
            fno_values = floor_for_log(fno_df[f"{prefix}_{component}_{metric_suffix}"].to_numpy())
            fno_epochs = fno_df["epoch"].to_numpy()
            valid = np.isfinite(fno_values)
            if np.any(valid):
                ax.plot(
                    fno_epochs[valid],
                    fno_values[valid],
                    color=FNO_COLOUR,
                    label="FNO",
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                )
        if fnoc is not None:
            fnoc_df = fnoc["training"]
            fnoc_values = floor_for_log(fnoc_df[f"{prefix}_{component}_{metric_suffix}"].to_numpy())
            fnoc_epochs = fnoc_df["epoch"].to_numpy()
            valid = np.isfinite(fnoc_values)
            if np.any(valid):
                ax.plot(
                    fnoc_epochs[valid],
                    fnoc_values[valid],
                    color=FNOC_COLOUR,
                    label="FNOC",
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                )
        if pino is not None:
            pino_df = pino["training"]
            pino_values = floor_for_log(pino_df[f"{prefix}_{component}_{metric_suffix}"].to_numpy())
            pino_epochs = pino_df["epoch"].to_numpy()
            valid = np.isfinite(pino_values)
            if np.any(valid):
                ax.plot(
                    pino_epochs[valid],
                    pino_values[valid],
                    color=PINO_COLOUR,
                    label="PINO",
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                )
        if pino_cont is not None:
            pino_cont_df = pino_cont["training"]
            pino_cont_values = floor_for_log(
                pino_cont_df[f"{prefix}_{component}_{metric_suffix}"].to_numpy()
            )
            pino_cont_epochs = pino_cont_df["epoch"].to_numpy()
            valid = np.isfinite(pino_cont_values)
            if np.any(valid):
                ax.plot(
                    pino_cont_epochs[valid],
                    pino_cont_values[valid],
                    color=PINO_CONT_COLOUR,
                    label="PINO-C",
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                )
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
        if ax.lines:
            ax.legend(frameon=False)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(title, y=1.02)
    save_figure(fig, output_dir, filename)


def make_plot_b1(fno, fnoc, pino, pino_cont, pino_tto, output_dir):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    runs = [
        ("FNO", fno, FNO_COLOUR),
        ("FNOC", fnoc, FNOC_COLOUR),
        ("PINO", pino, PINO_COLOUR),
        ("PINO-C", pino_cont, PINO_CONT_COLOUR),
        ("PINO+TTO", pino_tto, PINO_TTO_COLOUR),
    ]

    available = [(label, run, colour) for label, run, colour in runs if run is not None]
    if not available:
        mark_axis_unavailable(ax, "B1. Rollout relative L2", ylabel="Relative L2")
        ax.set_xlabel("Rollout step")
        save_figure(fig, output_dir, "B1_rollout_l2")
        return

    for label, run, colour in available:
        steps = run["per_sample"]["step"]
        l2 = run["per_sample"]["l2"]
        mean, _ = mean_and_std(l2)
        plot_line(ax, steps, mean, colour, label)

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2")
    ax.set_title("B1. Rollout relative L2")
    apply_linear_ticks(ax)
    ax.legend(frameon=False, ncol=min(4, len(available)))
    save_figure(fig, output_dir, "B1_rollout_l2")


def make_plot_b2(fno, fnoc, pino, pino_cont, pino_tto, output_dir):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    runs = [
        ("FNO", fno),
        ("FNOC", fnoc),
        ("PINO", pino),
        ("PINO-C", pino_cont),
        ("PINO+TTO", pino_tto),
    ]

    available = [(label, run) for label, run in runs if run is not None]
    if not available:
        mark_axis_unavailable(ax, "B2. Rollout PDE residuals", ylabel="Absolute PDE residual")
        ax.set_xlabel("Rollout step")
        save_figure(fig, output_dir, "B2_rollout_pde")
        return

    for label, run in available:
        steps = run["per_sample"]["step"]
        linestyle = MODEL_LINESTYLES[label]
        for component, colour in COMPONENT_COLOURS.items():
            pred = run["per_sample"][f"pred_{component}_abs"]
            pred_mean, _ = mean_and_std(pred)
            plot_log_line(
                ax,
                steps,
                pred_mean,
                colour,
                f"{component} {label}",
                linestyle=linestyle,
            )

            # truth = run['per_sample'][f'truth_{component}_rel']
            # truth_mean, _ = mean_and_std(truth)
            # ax.plot(
            #     steps, floor_for_log(truth_mean),
            #     color=colour, linestyle=':',
            #     label=f'{component} truth'
            # )

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Absolute PDE residual")
    ax.set_title("B2. Rollout PDE residuals")
    ax.set_yscale("log")
    apply_log_ticks(ax)
    ax.legend(frameon=False, ncol=4)
    save_figure(fig, output_dir, "B2_rollout_pde")


def make_plot_b3_constraint_diagnostics(fno, fnoc, pino, pino_cont, pino_tto, output_dir):
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.0), sharex=True)
    runs = [
        ("FNO", fno, FNO_COLOUR),
        ("FNOC", fnoc, FNOC_COLOUR),
        ("PINO", pino, PINO_COLOUR),
        ("PINO-C", pino_cont, PINO_CONT_COLOUR),
        ("PINO+TTO", pino_tto, PINO_TTO_COLOUR),
    ]

    available = []
    for label, run, colour in runs:
        if run is None:
            continue
        per_sample = run["per_sample"]
        if "pred_energy_balance_abs_residual" not in per_sample:
            continue
        available.append((label, run, colour))

    if not available:
        mark_axis_unavailable(axes[0], "Energy-balance residual", ylabel="Absolute residual")
        mark_axis_unavailable(axes[1], r"Max divergence", ylabel=r"$\max |\nabla \cdot u|$")
        mark_axis_unavailable(axes[2], r"Mean flow magnitude", ylabel=r"$|\langle u \rangle|$")
        axes[-1].set_xlabel("Rollout step")
        fig.suptitle("B3. Constraint diagnostics", y=1.01)
        save_figure(fig, output_dir, "B3_constraint_diagnostics")
        return

    for label, run, colour in available:
        steps = run["per_sample"]["step"]
        linestyle = MODEL_LINESTYLES.get(label, "-")

        pred_abs = run["per_sample"]["pred_energy_balance_abs_residual"]
        pred_mean, _ = mean_and_std(pred_abs)
        plot_log_line(axes[0], steps, pred_mean, colour, label, linestyle=linestyle)

        per_sample = run["per_sample"]
        if "pred_div_max" in per_sample:
            div_mean, _ = mean_and_std(per_sample["pred_div_max"])
            plot_log_line(axes[1], steps, div_mean, colour, label, linestyle=linestyle)

        mean_speed = per_sample["pred_domain_mean_speed"]
        mean_speed_mean, _ = mean_and_std(mean_speed)
        plot_log_line(axes[2], steps, mean_speed_mean, colour, label, linestyle=linestyle)

    axes[0].set_ylabel("Absolute residual")
    axes[0].set_title("Energy-balance residual")
    axes[0].set_yscale("log")
    apply_log_ticks(axes[0])
    axes[0].legend(frameon=False, ncol=min(4, len(available)))

    axes[1].set_ylabel(r"$\max |\nabla \cdot u|$")
    axes[1].set_title(r"Max divergence")
    axes[1].set_yscale("log")
    apply_log_ticks(axes[1])

    axes[2].set_xlabel("Rollout step")
    axes[2].set_ylabel(r"$|\langle u \rangle|$")
    axes[2].set_title(r"Mean flow magnitude")
    axes[2].set_yscale("log")
    apply_log_ticks(axes[2])

    fig.suptitle("B3. Constraint diagnostics", y=1.01)
    save_figure(fig, output_dir, "B3_constraint_diagnostics")


def make_spectra_plot(
    fno_npz, fnoc_npz, pino_npz, pino_cont_npz, pino_tto_npz, output_dir, filename, title
):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=False)
    runs = [
        ("FNO", fno_npz, FNO_COLOUR),
        ("FNOC", fnoc_npz, FNOC_COLOUR),
        ("PINO", pino_npz, PINO_COLOUR),
        ("PINO-C", pino_cont_npz, PINO_CONT_COLOUR),
        ("PINO+TTO", pino_tto_npz, PINO_TTO_COLOUR),
    ]
    reference = next((run for _, run, _ in runs if run is not None), None)

    if reference is None:
        mark_axis_unavailable(axes[0], "Energy spectrum", ylabel=r"Energy spectrum $E(k)$")
        mark_axis_unavailable(axes[1], "Enstrophy spectrum", ylabel=r"Enstrophy spectrum $Z(k)$")
        save_figure(fig, output_dir, filename)
        return

    k = reference["k_bins"]
    truth_energy_mean, _ = mean_and_std(reference["Ek_true"])
    truth_enstrophy_mean, _ = mean_and_std(reference["Zk_true"])

    plot_log_line(
        axes[0],
        k,
        truth_energy_mean,
        TRUTH_COLOUR,
        "Truth",
    )
    plot_log_line(
        axes[1],
        k,
        truth_enstrophy_mean,
        TRUTH_COLOUR,
        "Truth",
    )

    for label, run, colour in runs:
        if run is None:
            continue
        energy_mean, _ = mean_and_std(run["Ek_pred"])
        enstrophy_mean, _ = mean_and_std(run["Zk_pred"])
        plot_log_line(axes[0], k, energy_mean, colour, label)
        plot_log_line(axes[1], k, enstrophy_mean, colour, label)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Wavenumber $k$")
    axes[0].set_ylabel(r"Energy spectrum $E(k)$")
    axes[0].set_title("Energy spectrum")
    apply_log_ticks(axes[0])
    axes[0].legend(frameon=False)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Wavenumber $k$")
    axes[1].set_ylabel(r"Enstrophy spectrum $Z(k)$")
    axes[1].set_title("Enstrophy spectrum")
    apply_log_ticks(axes[1])
    axes[1].legend(frameon=False)

    fig.suptitle(title, y=1.03)
    save_figure(fig, output_dir, filename)


def make_plot_d1(fno, fnoc, pino, pino_cont, pino_tto, output_dir, include_pino_tto=False):
    reference = fno if fno is not None else fnoc
    if reference is None:
        reference = pino
    if reference is None:
        reference = pino_cont
    if reference is None:
        reference = pino_tto
    include_pino_cont = pino_cont is not None
    column_titles = get_vorticity_column_titles(
        include_pino_tto=include_pino_tto,
        include_pino_cont=include_pino_cont,
    )
    if reference is None:
        fig, axes = plt.subplots(
            4, len(column_titles), figsize=(3.5 * len(column_titles), 12), sharex=True, sharey=True
        )
        for row in range(4):
            for col, title in enumerate(column_titles):
                ylabel = f"Snapshot {row + 1}" if col == 0 else None
                mark_axis_unavailable(axes[row, col], title if row == 0 else "", ylabel=ylabel)
        save_figure(fig, output_dir, "D1_vorticity_snapshots")
        return

    fno_vort = None if fno is None else fno["vorticity"]
    fnoc_vort = None if fnoc is None else fnoc["vorticity"]
    pino_vort = None if pino is None else pino["vorticity"]
    pino_cont_vort = None if pino_cont is None else pino_cont["vorticity"]
    pino_tto_vort = None if pino_tto is None else pino_tto["vorticity"]

    truth = reference["vorticity"]["vorticity_truth"]
    fno_pred = None if fno_vort is None else fno_vort["vorticity_pred"]
    fnoc_pred = None if fnoc_vort is None else fnoc_vort["vorticity_pred"]
    pino_pred = None if pino_vort is None else pino_vort["vorticity_pred"]
    pino_cont_pred = None if pino_cont_vort is None else pino_cont_vort["vorticity_pred"]
    pino_tto_pred = None if pino_tto_vort is None else pino_tto_vort["vorticity_pred"]
    step_indices = reference["vorticity"]["step_indices"]

    if step_indices.size < 4:
        snapshot_steps = np.linspace(0, step_indices[-1], 4).round().astype(int)
    else:
        snapshot_positions = np.linspace(1, step_indices.size - 1, 4).round().astype(int)
        snapshot_steps = step_indices[snapshot_positions]

    selected_arrays = []
    for step in snapshot_steps[:4]:
        selected_arrays.append(truth[..., step])
        if fno_pred is not None:
            selected_arrays.append(fno_pred[..., step])
        if fnoc_pred is not None:
            selected_arrays.append(fnoc_pred[..., step])
        if pino_pred is not None:
            selected_arrays.append(pino_pred[..., step])
        if pino_cont_pred is not None:
            selected_arrays.append(pino_cont_pred[..., step])
        if pino_tto_pred is not None:
            selected_arrays.append(pino_tto_pred[..., step])
    vmax = max(np.abs(arr).max() for arr in selected_arrays)

    fig, axes = plt.subplots(
        4, len(column_titles), figsize=(3.5 * len(column_titles), 12), sharex=True, sharey=True
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90, wspace=0.08, hspace=0.06)

    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    im = None
    for row, step in enumerate(snapshot_steps[:4]):
        images = [fno_pred, fnoc_pred, pino_pred]
        if include_pino_cont:
            images.append(pino_cont_pred)
        if include_pino_tto:
            images.append(pino_tto_pred)
        images.append(truth)
        for col, image_stack in enumerate(images):
            if image_stack is None:
                title = column_titles[col] if row == 0 else ""
                ylabel = f"Step {step}" if col == 0 else None
                mark_axis_unavailable(axes[row, col], title, ylabel=ylabel)
                continue
            im = axes[row, col].imshow(
                image_stack[..., step],
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axes[row, col].set_box_aspect(1)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(f"Step {step}")

    if im is not None:
        cax = fig.add_axes([0.22, 0.05, 0.56, 0.025])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label(r"Vorticity $\omega$")
    fig.suptitle("D1. Vorticity snapshots", y=1.01)
    save_figure(fig, output_dir, "D1_vorticity_snapshots", use_tight_layout=False)


def main():
    parser = build_parser()
    args = parser.parse_args()

    configure_matplotlib()
    os.makedirs(args.output_dir, exist_ok=True)

    fno = maybe_load_run_artifacts(args.fno_results, args.seed, "FNO", stem=args.fno_prefix)
    fnoc = maybe_load_run_artifacts(args.fnoc_results, args.seed, "FNOC", stem=args.fnoc_prefix)
    pino = maybe_load_run_artifacts(args.pino_results, args.seed, "PINO", stem=args.pino_prefix)
    pino_cont = maybe_load_run_artifacts(
        args.pino_cont_results, args.seed, "PINO-C", stem=args.pino_cont_prefix
    )
    pino_tto = None
    if args.include_pino_tto:
        pino_tto_root = (
            args.pino_tto_results if args.pino_tto_results is not None else args.pino_results
        )
        pino_tto = maybe_load_eval_artifacts(
            pino_tto_root,
            args.seed,
            "PINO+TTO",
            file_suffix="_tto",
            stem=args.pino_tto_prefix if args.pino_tto_prefix is not None else args.pino_prefix,
        )
    if fno is None and fnoc is None and pino is None:
        raise ValueError(
            "At least one of --fno_results, --fnoc_results, or --pino_results must contain valid results."
        )

    make_plot_a1(fno, fnoc, pino, pino_cont, args.output_dir)
    make_pde_epoch_plot(
        fno,
        fnoc,
        pino,
        pino_cont,
        prefix="eval",
        metric_kind="abs",
        title="A2. Validation PDE residuals",
        output_dir=args.output_dir,
        filename="A2_validation_pde_vs_epoch",
    )
    make_pde_epoch_plot(
        fno,
        fnoc,
        pino,
        pino_cont,
        prefix="train",
        metric_kind="rel",
        title="A3. Training PDE residuals",
        output_dir=args.output_dir,
        filename="A3_training_pde_vs_epoch",
    )
    make_plot_b1(fno, fnoc, pino, pino_cont, pino_tto, args.output_dir)
    make_plot_b2(fno, fnoc, pino, pino_cont, pino_tto, args.output_dir)
    make_plot_b3_constraint_diagnostics(fno, fnoc, pino, pino_cont, pino_tto, args.output_dir)
    make_spectra_plot(
        None if fno is None else fno["spectra_final"],
        None if fnoc is None else fnoc["spectra_final"],
        None if pino is None else pino["spectra_final"],
        None if pino_cont is None else pino_cont["spectra_final"],
        None if pino_tto is None else pino_tto["spectra_final"],
        output_dir=args.output_dir,
        filename="C1_final_energy_spectra",
        title="C1. Final-step spectra",
    )
    make_spectra_plot(
        None if fno is None else fno["spectra_t1"],
        None if fnoc is None else fnoc["spectra_t1"],
        None if pino is None else pino["spectra_t1"],
        None if pino_cont is None else pino_cont["spectra_t1"],
        None if pino_tto is None else pino_tto["spectra_t1"],
        output_dir=args.output_dir,
        filename="C2_one_step_energy_spectra",
        title="C2. One-step spectra",
    )
    make_plot_d1(
        fno,
        fnoc,
        pino,
        pino_cont,
        pino_tto,
        args.output_dir,
        include_pino_tto=args.include_pino_tto,
    )


if __name__ == "__main__":
    main()
