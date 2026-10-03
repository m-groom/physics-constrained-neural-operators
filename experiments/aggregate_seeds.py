"""Aggregate a multi-seed campaign into one summary.

Reads the ``<model>_seed<seed>_per_step_metrics<suffix>.csv`` files that
``test_operator_AR_2d.py`` writes, together with the run-level
``<model>_seed<seed>_metrics<suffix>.csv`` beside each one, and reports every
model variant as a mean, a median and a standard deviation over its seeds, plus
the paired per-seed difference against a baseline variant.

A variant is a model and a file suffix: the plain rollout of ``PINO`` and its
test-time-optimised rollout ``PINO_tto`` are two variants of one checkpoint.

Metrics a run does not carry are reported as missing rather than as an error, so
the vorticity formulation — which has no velocity, and therefore no divergence,
domain-mean velocity or kinetic energy — aggregates through the same path as the
velocity formulation.

Usage:
    python aggregate_seeds.py --runs RUNS --out SUMMARY [--baseline FNO]
    python aggregate_seeds.py RUNS/*/evaluation_metrics/*_per_step_metrics.csv \
        --output_dir SUMMARY [--steps 1 16 32 64] [--baseline FNO_vort]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

# Rollout steps reported individually beside the horizon mean.
DEFAULT_STEPS = (1, 16, 32, 64)

# A rollout whose relative L2 reaches this has lost all predictive skill: the
# error is the size of the field itself. Runs above it are counted, not dropped,
# because the mean over seeds is not a central tendency once some seeds diverge.
BOUNDED_L2 = 1.0

# The per-step frame of one run. The suffix is empty for the plain rollout and
# "_tto" for the adapted one; it names a variant, not a different run.
PER_STEP_NAME = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_per_step_metrics(?P<suffix>.*)\.csv$")

# Metrics carried into the summary, in the order they are reported:
# (column, label, format).
METRIC_LABELS = [
    ("l2_rollout_mean", "rel-L2 (horizon mean)", "{:.4f}"),
    ("div_max", "max|div u| (worst step)", "{:.3e}"),
    ("truth_div_max", "max|div u| (truth, worst step)", "{:.3e}"),
    ("mean_speed_final", "|mean velocity| (final step)", "{:.3e}"),
    ("truth_mean_speed_final", "|mean velocity| (truth, final)", "{:.3e}"),
    ("energy_error_final", "kinetic energy error vs truth (final)", "{:+.4f}"),
    ("energy_drift", "kinetic energy drift over the rollout", "{:+.4f}"),
    ("truth_energy_drift", "kinetic energy drift (truth)", "{:+.4f}"),
    ("residual_rel_pred", "continuity residual (pred)", "{:.3e}"),
    ("residual_rel_truth", "continuity residual (truth)", "{:.3e}"),
    ("momx_rel", "momentum-x residual (pred)", "{:.3e}"),
    ("truth_momx_rel", "momentum-x residual (truth)", "{:.3e}"),
    ("momy_rel", "momentum-y residual (pred)", "{:.3e}"),
    ("truth_momy_rel", "momentum-y residual (truth)", "{:.3e}"),
    ("blowup_step", "blow-up step (median trajectory)", "{:.1f}"),
    ("blowup_never_frac", "trajectories never blown up", "{:.2f}"),
    ("high_band_frac_final", "energy fraction above the mode cutoff (final)", "{:.3e}"),
    ("truth_high_band_frac_final", "energy fraction above the cutoff (truth, final)", "{:.3e}"),
    ("seconds_per_epoch", "training s/epoch", "{:.1f}"),
    ("inference_s_per_trajectory", "inference s/trajectory", "{:.4f}"),
    ("tto_s_per_trajectory", "TTO s/trajectory", "{:.3f}"),
]


def find_metric_files(runs_dir):
    """Every per-step metrics CSV under ``runs_dir``, one run variant each."""
    pattern = os.path.join(runs_dir, "*", "evaluation_metrics", "*.csv")
    return sorted(
        path for path in glob.glob(pattern) if PER_STEP_NAME.match(os.path.basename(path))
    )


def run_level_path(per_step_path):
    """The run-level metrics CSV beside a per-step frame.

    Derived from the file name alone: the directory is called
    ``evaluation_metrics`` and a substitution on the whole path would rewrite it.
    """
    directory, name = os.path.split(per_step_path)
    return os.path.join(directory, name.replace("_per_step_metrics", "_metrics", 1))


def training_seconds_per_epoch(run_dir):
    """Training cost of a run, or NaN when the run wrote no timing file.

    ``train_timing.json`` is written by the campaign's SLURM job script, which
    lives with the campaign on the cluster rather than in this repository;
    a run directory without one simply reports no training cost.
    """
    path = os.path.join(run_dir, "train_timing.json")
    if not os.path.exists(path):
        return float("nan")
    with open(path, encoding="utf-8") as handle:
        timing = json.load(handle)
    return float(timing.get("seconds_per_epoch", float("nan")))


def _mean(frame, column):
    """Mean of a per-step column over the horizon, or NaN when it is absent."""
    return float(frame[column].mean()) if column in frame else float("nan")


def _final(frame, column):
    """Value of a per-step column at the last step, or NaN when it is absent."""
    return float(frame[column].iloc[-1]) if column in frame else float("nan")


def _worst(frame, column):
    """Largest value a per-step column reaches, or NaN when it is absent."""
    return float(frame[column].max()) if column in frame else float("nan")


def read_run(per_step_path, steps=DEFAULT_STEPS):
    """One campaign run variant as a flat record: model, seed and its metrics."""
    match = PER_STEP_NAME.match(os.path.basename(per_step_path))
    if match is None:
        raise ValueError(
            f"{per_step_path} is not named <model>_seed<seed>_per_step_metrics<suffix>.csv, "
            "so the variant and the seed of the run cannot be read from it"
        )
    per_step = pd.read_csv(per_step_path)
    run_dir = os.path.dirname(os.path.dirname(per_step_path))

    level_path = run_level_path(per_step_path)
    level = pd.read_csv(level_path).iloc[0] if os.path.exists(level_path) else {}

    record = {
        "model": f"{match['model']}{match['suffix']}",
        "seed": int(match["seed"]),
        "run_dir": run_dir,
        "n_steps": len(per_step),
        "n_trajectories": level.get("n_trajectories", float("nan")),
        "non_finite_total": level.get("non_finite_total", float("nan")),
        "l2_rollout_mean": _mean(per_step, "l2_mean"),
        "div_max": _worst(per_step, "pred_div_max_mean"),
        "truth_div_max": _worst(per_step, "truth_div_max_mean"),
        "mean_speed_final": _final(per_step, "pred_domain_mean_speed_mean"),
        "truth_mean_speed_final": _final(per_step, "truth_domain_mean_speed_mean"),
        "residual_rel_pred": _mean(per_step, "pred_cont_rel_mean"),
        "residual_rel_truth": _mean(per_step, "truth_cont_rel_mean"),
        "momx_rel": _mean(per_step, "pred_momx_rel_mean"),
        "truth_momx_rel": _mean(per_step, "truth_momx_rel_mean"),
        "momy_rel": _mean(per_step, "pred_momy_rel_mean"),
        "truth_momy_rel": _mean(per_step, "truth_momy_rel_mean"),
        # The stability column (#42): where the median trajectory first loses all skill,
        # and how many never do. Infinite when more than half the rollouts survive.
        "blowup_step": float(level.get("blowup_step_median", float("nan"))),
        "blowup_never_frac": float(level.get("blowup_never_frac", float("nan"))),
        "high_band_frac_final": _final(per_step, "pred_frac_energy_above_cutoff_mean"),
        "truth_high_band_frac_final": _final(per_step, "truth_frac_energy_above_cutoff_mean"),
        "seconds_per_epoch": training_seconds_per_epoch(run_dir),
        "inference_s_per_trajectory": float(
            level.get("rollout_seconds_per_trajectory", float("nan"))
        ),
        "tto_s_per_trajectory": float(level.get("tto_seconds_per_trajectory", float("nan"))),
    }
    by_step = per_step.set_index("step")["l2_mean"]
    for step in steps:
        record[f"l2_step{step}"] = float(by_step.get(step, float("nan")))

    # Two different questions: how far the rollout ends from the truth's energy,
    # and how much energy the rollout itself gained or lost. The truth drifts
    # too, so the second needs its own reference.
    pred_first, pred_last = (
        _mean(per_step.head(1), "pred_energy_mean"),
        _final(per_step, "pred_energy_mean"),
    )
    truth_first, truth_last = (
        _mean(per_step.head(1), "truth_energy_mean"),
        _final(per_step, "truth_energy_mean"),
    )
    record["energy_error_final"] = (pred_last - truth_last) / truth_last
    record["energy_drift"] = (pred_last - pred_first) / pred_first
    record["truth_energy_drift"] = (truth_last - truth_first) / truth_first
    return record


def summarise(runs, metrics, baseline=None, bounded_column=None):
    """Mean, median and std over seeds per variant, with paired differences.

    The paired difference is taken seed by seed, so it removes the shared
    initialisation variance that a difference of means would leave in.

    ``bounded_column`` names the relative-L2 column that decides whether a seed's
    rollout stayed bounded; without it no seed is counted.
    """
    summary = []
    baseline_rows = runs[runs["model"] == baseline].set_index("seed") if baseline else None
    for model, group in runs.groupby("model", sort=True):
        record = {
            "model": model,
            "n_seeds": len(group),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"])),
        }
        # The median is reported beside the mean: a variant whose rollout diverges
        # on some seeds and not others has a bimodal spread that a mean hides.
        record["n_bounded"] = (
            int((group[bounded_column] < BOUNDED_L2).sum()) if bounded_column else 0
        )
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = np.mean(values)
            record[f"{metric}_median"] = np.median(values)
            record[f"{metric}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
            if baseline_rows is not None and model != baseline:
                paired = group.set_index("seed")[metric] - baseline_rows[metric]
                paired = paired.dropna().to_numpy(dtype=float)
                record[f"{metric}_paired_mean"] = np.mean(paired) if len(paired) else np.nan
                record[f"{metric}_paired_median"] = np.median(paired) if len(paired) else np.nan
                record[f"{metric}_paired_std"] = np.std(paired, ddof=1) if len(paired) > 1 else 0.0
        summary.append(record)
    return pd.DataFrame(summary)


def paired_table(runs, baseline, metric="l2_rollout_mean"):
    """Per-seed value of every variant and its difference from ``baseline``."""
    wide = runs.pivot_table(index="seed", columns="model", values=metric)
    for model in [name for name in wide.columns if name != baseline]:
        wide[f"{model} - {baseline}"] = wide[model] - wide[baseline]
    return wide.reset_index()


def _cell(summary_row, metric, fmt, suffix=""):
    """A ``mean ± std`` cell, or ``--`` when the metric was never recorded."""
    mean = summary_row.get(f"{metric}{suffix}_mean", np.nan)
    std = summary_row.get(f"{metric}{suffix}_std", np.nan)
    # A metric the run never recorded is missing; one that overflowed is not.
    if np.isnan(mean):
        return "--"
    if np.isinf(mean):
        return "inf" if mean > 0 else "-inf"
    if not np.isfinite(std):
        return fmt.format(mean)
    # A spread is never signed, even where the metric it belongs to is.
    return f"{fmt.format(mean)} ± {fmt.format(std).lstrip('+')}"


def _table(summary, metric_labels, cell):
    """One Markdown table: a row per metric, a column per variant."""
    header = ["metric", *list(summary["model"])]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for metric, label, fmt in metric_labels:
        cells = [cell(row, metric, fmt) for _, row in summary.iterrows()]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return lines


def markdown_table(frame):
    """Render a data frame as a Markdown table, one row per record."""
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.itertuples(index=False):
        cells = [
            value if isinstance(value, str) else ("" if pd.isna(value) else f"{value:g}")
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(path, summary, runs, steps, baseline=None, title="Campaign summary"):
    """Write the human-readable summary: headline, medians, paired, per-run."""
    step_metrics = [(f"l2_step{step}", f"rel-L2 @ {step}", "{:.4f}") for step in steps]
    metric_labels = [*METRIC_LABELS[:1], *step_metrics, *METRIC_LABELS[1:]]

    seeds = sorted(int(seed) for seed in runs["seed"].unique())
    lines = [
        f"# {title}",
        "",
        (
            f"{len(runs)} runs, {runs['model'].nunique()} model variants, seeds {seeds}, "
            f"{int(runs['n_steps'].max())} rollout steps. "
            "Mean ± standard deviation over seeds."
        ),
        "",
    ]
    bounded = [f"{int(row['n_bounded'])} / {int(row['n_seeds'])}" for _, row in summary.iterrows()]
    table = _table(summary, metric_labels, _cell)
    table.insert(
        2,
        "| "
        + " | ".join([f"seeds bounded (rel-L2 < {BOUNDED_L2:g} at the final step)", *bounded])
        + " |",
    )
    lines += table

    def median_cell(row, metric, fmt):
        value = row.get(f"{metric}_median", np.nan)
        if np.isnan(value):
            return "--"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return fmt.format(value)

    lines += ["", "## Median over seeds", ""]
    lines += _table(summary, metric_labels, median_cell)

    if baseline:
        lines += ["", f"## Paired per-seed difference against `{baseline}`", ""]
        others = summary[summary["model"] != baseline]
        lines += _table(others, metric_labels, lambda row, m, f: _cell(row, m, f, suffix="_paired"))
        lines += ["", f"### Rollout relative L2 per seed, against `{baseline}`", ""]
        lines += markdown_table(paired_table(runs, baseline))

    lines += ["", "## Per-run values", ""]
    per_run_columns = [
        "model",
        "seed",
        "l2_rollout_mean",
        f"l2_step{steps[-1]}",
        "div_max",
        "energy_error_final",
        "non_finite_total",
    ]
    lines += markdown_table(runs.sort_values(["model", "seed"])[per_run_columns])

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_steps(values):
    """Rollout steps from ``--steps 1 16 32 64`` or ``--steps 1,16,32,64``."""
    if not values:
        return tuple(DEFAULT_STEPS)
    return tuple(int(step) for value in values for step in str(value).split(",") if step)


def main(argv=None):
    """Read a campaign's per-run CSVs and write summary.csv, runs.csv, summary.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_csv", nargs="*", help="per-step metrics CSVs of the runs to aggregate"
    )
    parser.add_argument(
        "--runs", default=None, help="directory holding the per-run <config>_seed<k> directories"
    )
    parser.add_argument(
        "--out", "--output_dir", dest="out", default=None, help="directory for the summary"
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="variant the paired differences are taken against "
        "(default: the first variant by name)",
    )
    parser.add_argument(
        "--steps", nargs="*", default=None, help="rollout steps to report individually"
    )
    parser.add_argument("--title", default="Campaign summary", help="title of summary.md")
    args = parser.parse_args(argv)

    steps = parse_steps(args.steps)
    metric_files = args.metrics_csv or (find_metric_files(args.runs) if args.runs else [])
    if not metric_files:
        raise SystemExit("no per-step metrics CSVs given; pass paths or --runs")
    runs = pd.DataFrame([read_run(path, steps=steps) for path in metric_files])
    duplicates = runs[runs.duplicated(subset=["model", "seed"], keep=False)]
    if not duplicates.empty:
        pairs = sorted(set(zip(duplicates["model"], duplicates["seed"], strict=True)))
        raise SystemExit(
            f"duplicate (model, seed) run files for {pairs}: "
            "a stale smoke or debug directory is being counted as an extra seed"
        )
    runs = runs.sort_values(["model", "seed"]).reset_index(drop=True)

    metrics = [name for name, _, _ in METRIC_LABELS]
    metrics += [f"l2_step{step}" for step in steps]
    models = sorted(runs["model"].unique())
    baseline = args.baseline if args.baseline is not None else (models[0] if models else None)
    if baseline is not None and baseline not in models:
        raise SystemExit(f"baseline {baseline!r} is not among the variants {models}")
    summary = summarise(runs, metrics, baseline=baseline, bounded_column=f"l2_step{steps[-1]}")

    out_dir = args.out or args.runs
    if out_dir is None:
        raise SystemExit("pass --out to say where the summary should be written")
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    if baseline is not None and len(models) > 1:
        paired_table(runs, baseline).to_csv(os.path.join(out_dir, "paired.csv"), index=False)
    write_markdown(
        os.path.join(out_dir, "summary.md"),
        summary,
        runs,
        steps,
        baseline=baseline,
        title=args.title,
    )
    print(f"Wrote summary.csv, runs.csv and summary.md to {out_dir}")


if __name__ == "__main__":
    main()
