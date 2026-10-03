r"""Paper figures and LaTeX table bodies for the STODY submission.

Reads the per-step metrics that ``test_operator_AR_2d.py`` writes for every run,
the end-of-rollout spectra it saves beside them, the free-running energy traces
``rollout_stability.py`` writes, the per-epoch training logs, the spectral-flux
archive of the flux check, and the seed summary that ``aggregate_seeds.py``
builds from the evaluations. Writes the paper's figures as PDF and the table
bodies as ``\input``-able LaTeX. Every plotted or tabulated number is read from
those files; nothing here synthesises data, and a metric a campaign does not
carry is reported as missing rather than filled in.

Which campaign feeds which figure is set in ``figures.toml``, so a campaign's
numbers drop into the same slots by appearing at the path that file names, and an
arm that has not run yet is left out rather than drawn empty.

Run from the repository root:

    uv run --no-sync python figures/make_figures.py

Spread convention. The figures show the median over seeds inside an
interquartile band; the tables keep the mean with the seed standard deviation
that issue #24 asked for and the captions promise. The two differ because the
seeds do: on E1 three of the five plain-FNO seeds and all five
PINO-continuity-only seeds lose predictive skill, the last of them reaching a
relative L2 above 1e10, and an average over seeds is then not a central
tendency --- one diverged seed sets the mean and the band. ``aggregate_seeds.py``
makes the same point where it counts bounded runs. A median curve stays on the
panel and an interquartile band still widens when the seeds disagree, so the
disagreement is visible rather than off-scale.

The training curves are the one exception: their band is the full seed range,
because that figure's question is whether any seed's validation loss turns up,
and a quartile band hides the seed that does.
"""

import argparse
import glob
import re
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# The file-naming contract of a run variant belongs to the evaluation script and
# is already written down once, in the aggregator. Importing it keeps the figures
# and summary.csv reading the same files by the same rule.
from experiments.aggregate_seeds import PER_STEP_NAME

DEFAULT_CONFIG = Path(__file__).with_name("figures.toml")

# The spectra archive one run writes per rollout step, named like the per-step
# frame beside it: <model>_seed<seed>_spectra_t<step><suffix>.npz.
SPECTRA_NAME = re.compile(
    r"^(?P<model>.+)_seed(?P<seed>\d+)_spectra_t(?P<step>\d+)(?P<suffix>.*)\.npz$"
)

# The summary one run writes per evaluation, which carries the epoch the guarded
# validation selection kept: <model>_seed<seed>_metrics<suffix>.csv.
METRICS_NAME = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)_metrics(?P<suffix>.*)\.csv$")

# A run directory, and a free-rollout archive named for the arm it belongs to,
# are both read as <model>_seed<seed>.
RUN_NAME = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)$")

# A test-time-optimised campaign names its free-rollout archive for the run whose
# checkpoint it adapted, where the evaluation names the adapted result with a
# `_tto` suffix. The trace is of the adapted weights --- each trajectory free-runs
# under the weights adapted on it --- so it belongs to the adapted variant, and
# the unadapted checkpoint's own trace is the plain arm's, filed under that name.
FREE_ALIASES = {
    "PINO_TTO": "PINO_TTO_tto",
    "PINO_cont_TTO": "PINO_cont_TTO_tto",
    "PINO_vort_TTO": "PINO_vort_TTO_tto",
}

# The closed-energy campaign evaluated at a pinned band cutoff of 16, so every
# archive it wrote carries this evaluation-variant suffix; the campaign arms it
# is compared against wrote their default evaluation under no suffix at all.
CLOSED_SUFFIX = "_k16"

# The truth is a reference, not another series, so it wears neutral ink rather
# than a categorical hue and a dotted stroke rather than a marker.
TRUTH_STYLE = {"color": "#3d3d3a", "linestyle": (0, (1.2, 1.2)), "linewidth": 0.9}
TRUTH_LABEL = "truth"

# Ink for text that is not a tick label: row labels and panel annotations.
MUTED_INK = "#52514e"

# Seed bands are a wash, never a saturated block.
BAND_ALPHA = 0.18

# Marks ride the median curve often enough to name the arm and rarely enough to
# leave the curve readable.
MARKER_SIZE = 2.6
MARKERS_PER_CURVE = 8

# Both the rollout figure and the stability figure put the rollout step on the x
# axis. The rollout figure's caption calls it lead time, so the label names both.
STEP_LABEL = "lead time (rollout step $n$)"

# A free-running trajectory counts as bounded while its kinetic energy stays
# within this factor of its initial value, above or below. ``rollout_stability.py``
# rolls under the upper bound alone and writes its blow-up steps by it; the lower
# bound is applied here, from the energy trace, so that a rollout that drains
# away is counted as leaving the bounded set too. The stability figure marks both.
BLOWUP_GUARD = 10.0

# The spectrum panel's inset magnifies the high-wavenumber band, where the arms
# that are not adapted at test time sit within half a decade of one another
# while the panel's own axis spans seven decades on E2 and eleven on E1. Which
# band it magnifies is a figure setting; how the inset is drawn is here.
#
# `INSET_BOX` is its position and size inside the panel, in axes fractions. The
# lower left of a log-log energy spectrum is empty on both campaigns --- E2's
# spectrum is flat and E1's falls from the top left to the bottom right --- and
# the box stands clear of the panel's own tick labels on the left and of the
# lowest curve above its top right corner, which on E1 is the arm adapted on the
# continuity residual alone as it dives past k = 8.
INSET_BOX = (0.22, 0.10, 0.36, 0.30)

# Where a curve falls below this fraction of the truth's own energy it has shed
# the band rather than resolved it differently, and those wavenumbers set none
# of the inset's vertical limits: letting them would stretch the axis back over
# the orders of magnitude the inset exists to close up. The curve is still
# drawn, and falls outside the inset's range.
INSET_FLOOR = 0.1

# Room left above the highest curve and below the lowest, as a factor, so that
# neither rides the inset's frame.
INSET_PAD = 1.15

# The two series of the flux figure. They are quantities rather than arms, so
# they take the first two slots of the categorical theme in their own right. The
# dissipation is drawn first and drawn wider, because on E2 the two curves
# coincide below the forcing band and a reader has to be able to see that they do.
FLUX_SERIES = {
    "D_in": {
        "colour": "#eb6834",
        "marker": "s",
        "label": r"$D_{\mathrm{in}}(K)$",
        "linewidth": 1.8,
    },
    "Pi_mean": {"colour": "#2a78d6", "marker": "o", "label": r"$\Pi(K)$", "linewidth": 1.0},
}

# One panel of the flux figure per dataset. `prefix` names the arrays inside the
# archive; `k_max` is where the panel stops, chosen to hold the dataset's own
# forcing scale and both cutoffs; `forcing` is the forced wavenumber, or the band
# the stochastic forcing occupies, which is where Pi(K) parts from D_in(K).
FLUX_PANELS = {
    "E1": {"prefix": "E1", "k_max": 32, "forcing": (4,)},
    "E3": {"prefix": "E3", "k_max": 100, "forcing": (56, 72)},
}

# The cutoffs marked on every flux panel: 16 is the FNO's own spectral cutoff on
# E1 and the band diagnostic on E3, 31 the coarse grid's Nyquist.
FLUX_CUTOFFS = (16, 31)


@contextmanager
def paper_style():
    """Draw inside the paper's figure styling, restoring the caller's afterwards.

    Body text of the NeurIPS template: a 5.5in block, serif to match it, small
    labels. Marks are thin and the grid is a solid recessive hairline one step off
    the page, so the curves are the only loud thing on a panel.
    """
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#8f8e8a",
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "lines.linewidth": 1.0,
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "xtick.labelcolor": "#0b0b0b",
            "ytick.labelcolor": "#0b0b0b",
            "figure.constrained_layout.use": True,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        yield


def load_config(path):
    """The figure configuration as nested dictionaries."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def style_entries(models):
    """Drawing styles keyed by arm, from a ``[models]``-shaped configuration table."""
    return {
        key: {
            "colour": entry["colour"],
            "marker": entry["marker"],
            # `dash` is the on/off sequence in points; matplotlib wants it paired
            # with a phase, and every arm's pattern starts at the same phase.
            "dash": (0, tuple(entry["dash"])) if "dash" in entry else "-",
            "label": entry["label"],
        }
        for key, entry in models.items()
    }


def panel_grid(width, columns, rows, aspect=0.85, legend=0.30):
    """Figure size for a ``rows`` by ``columns`` grid of panels.

    ``width`` is the width the TeX includes the figure at, so a panel is that
    width divided by the number of columns and the labels arrive on the page at
    the size ``paper_style`` sets. ``legend`` is the strip left under the last
    row for the shared legend.
    """
    return (width, rows * (width / columns) * aspect + legend)


def discover(runs_dir, pattern, subdir, **require):
    """Run files under ``runs_dir`` grouped as ``{variant: {seed: path}}``.

    A run directory is named for the run and a file for the variant it belongs
    to, and the two differ: the test-time-optimised rollout of ``PINO_TTO`` is
    written into the ``PINO_TTO`` run's directory alongside the plain one. The
    variant is therefore read from the file name and never from the directory.

    Args:
        runs_dir: Directory of run directories.
        pattern: Compiled pattern with ``model``, ``seed`` and ``suffix`` groups.
        subdir: Directory inside each run that holds the files.
        **require: Named groups of ``pattern`` restricted to a value, as strings.

    Returns:
        Mapping from variant name to a mapping from seed to file path.

    Raises:
        ValueError: If two files claim the same variant and seed, which would
            make the choice between them silent.
    """
    found = {}
    for path in sorted(Path(runs_dir).glob(f"*/{subdir}/*")):
        match = pattern.match(path.name)
        if match is None or any(match[group] != value for group, value in require.items()):
            continue
        variant = f"{match['model']}{match['suffix']}"
        seed = int(match["seed"])
        if seed in found.get(variant, {}):
            raise ValueError(
                f"{variant} seed {seed} is claimed by both {found[variant][seed]} and {path}"
            )
        found.setdefault(variant, {})[seed] = path
    return found


def read_variants(runs_dir, variants):
    """Per-step frames as ``{variant: {seed: DataFrame}}``, for the variants present.

    A variant the campaign has not run is left out rather than reported empty, so
    a caller can tell "not run yet" from "run and measured".
    """
    available = discover(runs_dir, PER_STEP_NAME, "evaluation_metrics")
    return {
        variant: {seed: pd.read_csv(path) for seed, path in sorted(available[variant].items())}
        for variant in variants
        if variant in available
    }


def seed_curves(frames, column, index="step"):
    """One index column and one measured column stacked over seeds.

    Args:
        frames: ``{seed: DataFrame}`` for one variant, as ``read_variants`` returns.
        column: Column to read.
        index: The column the others are measured against: the rollout step for
            an evaluation, the epoch for a training log.

    Returns:
        ``(index_values, values)`` with ``values`` of shape ``(seed, index)``,
        seeds in ascending order.
    """
    seeds = sorted(frames)
    ordered_frames = [frames[seed] for seed in seeds]
    steps = ordered_frames[0][index].to_numpy()
    for seed, frame in zip(seeds, ordered_frames, strict=True):
        # Seeds measured over different horizons cannot share an x axis, and
        # stacking them would either fail obscurely or line up the wrong steps.
        if not np.array_equal(frame[index].to_numpy(), steps):
            raise ValueError(
                f"seed {seed} covers {index} {frame[index].iloc[0]}"
                f"-{frame[index].iloc[-1]}, not {steps[0]}-{steps[-1]} like seed"
                f" {seeds[0]}"
            )
    return steps, np.stack([frame[column].to_numpy() for frame in ordered_frames])


def seed_spectra(runs_dir, variants, step=None):
    """Wavenumbers and the isotropic energy spectra at the end of the rollout.

    Each archive holds one spectrum per test trajectory, so the trajectory mean
    is taken first and a seed contributes a single curve.

    The end of the rollout is the last step any run under ``runs_dir`` recorded,
    which is the step the arms are compared at. A variant that stopped earlier
    has no archive there and is left out rather than compared at a step of its
    own.

    Args:
        runs_dir: Directory of run directories.
        variants: Variant names to read, in the order wanted.
        step: Rollout step whose archives are compared; ``None`` picks the last
            step any run recorded.

    Returns:
        ``{variant: (k, predicted, truth)}`` with the two arrays shaped
        ``(seed, wavenumber)``, restricted to the variants with an archive there.
    """
    if step is None:
        steps = sorted(
            {
                int(match["step"])
                for path in Path(runs_dir).glob("*/saved_plots/*.npz")
                if (match := SPECTRA_NAME.match(path.name))
            }
        )
        if not steps:
            return {}
        step = steps[-1]
    available = discover(runs_dir, SPECTRA_NAME, "saved_plots", step=str(step))
    spectra = {}
    for variant in variants:
        if variant not in available:
            continue
        predicted, truth, wavenumbers = [], [], None
        for _, path in sorted(available[variant].items()):
            with np.load(path) as archive:
                wavenumbers = archive["k_bins"]
                predicted.append(archive["Ek_pred"].mean(axis=0))
                truth.append(archive["Ek_true"].mean(axis=0))
        spectra[variant] = (wavenumbers, np.stack(predicted), np.stack(truth))
    return spectra


def exit_steps(ratio, guard=BLOWUP_GUARD):
    """First 1-based step at which each trajectory leaves the bounded set, or -1.

    The bounded set is ``1 / guard <= ratio <= guard``, and a non-finite energy
    counts as leaving it at the step it appears. Against the upper bound alone
    this is ``rollout_stability.blowup_steps`` and reproduces its archive exactly;
    the lower bound is what that archive does not count.
    """
    ratio = np.asarray(ratio, dtype=float)
    gone = (ratio > guard) | (ratio < 1.0 / guard) | ~np.isfinite(ratio)
    first = gone.argmax(axis=0) + 1
    first[~gone.any(axis=0)] = -1
    return first


def free_traces(patterns):
    """Free-running energy traces as ``{variant: {seed: (ratio, blowup)}}``.

    A free rollout carries no ground truth, so it is stored beside the campaign
    rather than beside an evaluation, and the two campaigns store it differently:
    E1 writes one archive into each run directory and E3 one archive per arm into
    a directory of its own. Both name the arm and the seed somewhere, so the
    variant is read from the archive's own name where that parses and from its
    directory's name where it does not.

    One pattern supplies every trace, never a mixture: the arms of a figure have to
    share a step axis, and an arm read from a 400-step archive would otherwise be
    counted as bounded over the 1,600 steps it was never rolled for.

    Args:
        patterns: Glob patterns, most preferred first. The one read is the
            earliest that carries every arm and seed any of the others carries,
            so a longer rollout supersedes a shorter one by being named ahead of
            it --- but only once it covers everything the shorter one did. A
            campaign is still writing its archives for most of a day, and a
            pattern that merely has more files than another can still be missing
            four of one arm's five seeds; where no pattern covers the rest, the
            last is read, which is the settled one.

    Returns:
        ``{variant: {seed: (ratio, blowup)}}``: ``ratio`` is kinetic energy over
        its own initial value, shaped ``(step, trajectory)``, and ``blowup`` the
        1-based step at which each trajectory first left the bounded set, or -1
        for one that never did, counted from the trace by ``exit_steps`` rather
        than read from the archive, whose own count knows the upper bound alone.
    """
    scanned = []
    for pattern in patterns:
        found = {}
        for name in sorted(glob.glob(pattern)):
            path = Path(name)
            match = RUN_NAME.match(path.stem) or RUN_NAME.match(path.parent.name)
            if match is not None:
                variant = FREE_ALIASES.get(match["model"], match["model"])
                found[(variant, int(match["seed"]))] = path
        scanned.append(found)
    covering = (
        candidate
        for candidate in scanned
        if candidate and all(candidate.keys() >= other.keys() for other in scanned)
    )
    chosen = next(covering, None) or next((found for found in reversed(scanned) if found), {})
    traces = {}
    for (variant, seed), path in chosen.items():
        with np.load(path) as archive:
            ratio = archive["trace"][..., 0]
        traces.setdefault(variant, {})[seed] = (ratio, exit_steps(ratio))
    return traces


def bounded_fraction(blowup, steps):
    """Fraction of trajectories still bounded at each of ``steps``.

    Bounded here means one thing only: the trajectory's kinetic energy has stayed
    within the guard of its initial value, above and below. ``blowup`` is -1 for a
    trajectory that never left that set and otherwise the step at which it first
    did, which is what ``exit_steps`` counts and what both the stability figure
    and Table 1's stability column read, so the two agree with each other.

    This is not the ``blowup_never_frac`` of a campaign summary, which counts a
    different failure over a different rollout: that column is measured against
    the ground truth and reads 0.00 for every E3 arm at 448 steps, where this one
    reads 0.18 to 1.00. One says the prediction has lost its skill, the other says
    the energy has left the bounded set, and they are hundreds of steps apart.
    """
    first = np.asarray(blowup)[:, None]
    return ((first < 0) | (first > np.asarray(steps)[None, :])).mean(axis=0)


def padded_ratio(ratio, horizon):
    """One energy trace on the common step axis of its experiment.

    A rollout stops recording the moment any trajectory in its batch goes
    non-finite, and the batch it stops for holds healthy trajectories as well as
    the one that failed. Each is therefore held at its last measured value: the
    failed trajectory's is already infinite and stays off the panel, and a healthy
    one keeps the energy it had, which is the same thing ``bounded_fraction`` says
    about it over those steps.
    """
    ratio = np.asarray(ratio, dtype=float)
    if ratio.shape[0] >= horizon:
        return ratio[:horizon]
    tail = np.repeat(ratio[-1:], horizon - ratio.shape[0], axis=0)
    return np.concatenate([ratio, tail])


def free_horizon(traces):
    """The longest free rollout any arm of a campaign recorded."""
    return max(
        (ratio.shape[0] for seeds in traces.values() for ratio, _ in seeds.values()), default=0
    )


def training_logs(runs_dir):
    """Per-epoch training logs as ``{variant: {seed: DataFrame}}``.

    A training log belongs to the run rather than to a variant of it --- its own
    file name carries the architecture --- so here, and only here, the variant is
    read from the run directory's name.
    """
    found = {}
    for path in sorted(Path(runs_dir).glob("*/*_training_log.csv")):
        match = RUN_NAME.match(path.parent.name)
        if match is None:
            continue
        found.setdefault(match["model"], {})[int(match["seed"])] = pd.read_csv(path)
    return found


def selected_epochs(runs_dir):
    """The epoch each run's evaluated checkpoint came from, ``{variant: {seed: epoch}}``.

    The evaluation records the epoch the guarded validation selection kept, so
    the epoch a figure marks is the one every reported number was measured at,
    and not a minimum recomputed from the log.
    """
    available = discover(runs_dir, METRICS_NAME, "evaluation_metrics", suffix="")
    epochs = {}
    for variant, seeds in available.items():
        for seed, path in sorted(seeds.items()):
            frame = pd.read_csv(path)
            if "checkpoint_epoch" in frame:
                epochs.setdefault(variant, {})[seed] = int(frame["checkpoint_epoch"].iloc[0])
    return epochs


def band(values):
    """Median and interquartile band over the seed axis.

    Returns:
        ``(median, lower, upper)``, each one curve over the remaining axis.
    """
    return (
        np.median(values, axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
    )


def seed_range(values):
    """Median and full seed range over the seed axis."""
    return np.median(values, axis=0), values.min(axis=0), values.max(axis=0)


def log_marks(count, marks=MARKERS_PER_CURVE):
    """Indices into ``count`` samples that are evenly spaced on a logarithmic axis.

    ``markevery`` spaces marks evenly in the index, which piles every one of them
    into the right-hand decade of a logarithmic x axis and leaves the first
    decades unnamed.
    """
    if count < 2:
        return list(range(count))
    return np.unique(np.geomspace(1, count, marks).astype(int) - 1).tolist()


def draw_band(ax, x, values, style, label=None, spread=band, marks=None):
    """One arm's median curve inside its band over the seeds."""
    middle, lower, upper = spread(values)
    ax.fill_between(x, lower, upper, color=style["colour"], alpha=BAND_ALPHA, linewidth=0)
    ax.plot(
        x,
        middle,
        color=style["colour"],
        linestyle=style.get("dash", "-"),
        marker=style["marker"],
        markersize=MARKER_SIZE,
        markeredgewidth=0.6,
        markevery=marks if marks is not None else max(1, len(x) // MARKERS_PER_CURVE),
        label=label,
    )


def draw_truth(ax, x, values, label=None):
    """The truth's own curve, as a reference rather than a series."""
    ax.plot(x, np.median(values, axis=0), label=label, **TRUTH_STYLE)


def column_title(ax, text):
    """Name the experiment a column of panels belongs to, above the column's top panel.

    A figure that carries both campaigns side by side names each of them once, over
    the column it owns, where the name reads as a heading rather than as one more
    thing drawn on a crowded panel.
    """
    ax.set_title(text)


def row_label(ax, text):
    """Name the experiment a single-campaign figure belongs to, inside the panel."""
    ax.annotate(
        text,
        xy=(0.03, 0.94),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=6,
        color=MUTED_INK,
    )


def ordered(available, order):
    """The entries of ``available`` that ``order`` names, in ``order``'s order.

    Colour follows the arm and never its rank, so every panel walks the arms in
    the one order the configuration lists them in, whichever of them the campaign
    at hand happens to carry.
    """
    return {key: available[key] for key in order if key in available}


def panel_error(ax, frames, styles):
    """Relative L2 against the truth, by rollout step."""
    for variant, seeds in frames.items():
        steps, values = seed_curves(seeds, "l2_mean")
        draw_band(ax, steps, values, styles[variant], label=styles[variant]["label"])
    ax.set_yscale("log")
    ax.set(xlabel=STEP_LABEL, ylabel="relative $L_2$ error")


def draw_spectra(ax, spectra, styles, name_the_truth=False):
    """Every arm's spectrum and the truth's, on one axes.

    The truth is read off the first arm, because every arm of a campaign is
    compared against the same one.
    """
    for variant, (wavenumbers, predicted, _) in spectra.items():
        draw_band(ax, wavenumbers, predicted, styles[variant])
    first = next(iter(spectra.values()), None)
    if first is not None:
        wavenumbers, _, truth = first
        draw_truth(ax, wavenumbers, truth, label=TRUTH_LABEL if name_the_truth else None)


def inset_limits(spectra, band):
    """Vertical limits for the spectrum inset, from the curves inside ``band``.

    A median curve is read at the wavenumbers where it still reaches
    ``INSET_FLOOR`` of the truth's own energy, which for the truth is everywhere,
    and the extremes over those wavenumbers are the limits, padded so that
    nothing rides the frame. An arm that has shed the band contributes only the
    part of it that it has not shed, and the rest falls outside the range.

    Args:
        spectra: ``{variant: (k, predicted, truth)}`` as ``seed_spectra`` returns.
        band: The pair of wavenumbers the inset spans, inclusive.

    Returns:
        ``(lower, upper)``, or ``None`` where no wavenumber falls in the band.
    """
    first = next(iter(spectra.values()), None)
    if first is None:
        return None
    wavenumbers = first[0]
    inside = (wavenumbers >= band[0]) & (wavenumbers <= band[1])
    if not inside.any():
        return None
    truth = np.median(first[2], axis=0)[inside]
    curves = [truth] + [
        np.median(predicted, axis=0)[inside] for _, predicted, _ in spectra.values()
    ]
    carried = np.concatenate([curve[curve >= truth * INSET_FLOOR] for curve in curves])
    return carried.min() / INSET_PAD, carried.max() * INSET_PAD


def spectrum_inset(ax, spectra, styles, band):
    """A magnified view of ``band`` inside the spectrum panel, and its rectangle.

    The same curves as the panel itself, in the same hues, markers, strokes and
    seed bands, clipped to the band. Naming the arms is the panel's job, so the
    inset carries no legend of its own, and its tick labels are the size the
    panel's are. A campaign whose spectra carry no wavenumber in the band gets
    no inset and no rectangle.
    """
    limits = inset_limits(spectra, band)
    if limits is None:
        return
    inner = ax.inset_axes(INSET_BOX, xscale="log", yscale="log")
    draw_spectra(inner, spectra, styles)
    inner.set(xlim=tuple(band), ylim=limits)
    # The inset's own wavenumbers are named as they are read rather than in the
    # scientific notation the panel's decades are set in.
    inner.xaxis.set_major_formatter(ScalarFormatter())
    inner.xaxis.set_minor_formatter(ScalarFormatter())
    indicator = ax.indicate_inset_zoom(inner, edgecolor=MUTED_INK, alpha=0.9, linewidth=0.9)
    # The rectangle says where the inset is read from and the two connectors only
    # which corners it maps to, so they cross the panel at a weight that does not
    # compete with the curves they cross.
    for connector in indicator.connectors:
        connector.set(linewidth=0.5, alpha=0.4)


def panel_spectrum(ax, spectra, styles, band):
    """Isotropic energy spectrum at the end of the rollout, against the truth.

    ``band`` is the pair of wavenumbers an inset magnifies, because on the panel
    itself the arms that hold the truth's spectrum are one stroke wide.
    """
    draw_spectra(ax, spectra, styles, name_the_truth=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="wavenumber $k$", ylabel="$E(k)$, end of rollout")
    spectrum_inset(ax, spectra, styles, band)


def panel_constraint(ax, frames, styles, metric, ylabel, name_the_arms=False):
    """One constraint diagnostic by rollout step, against the truth's own level.

    Args:
        ax: Axes to draw on.
        frames: ``{variant: {seed: DataFrame}}`` of the plotted arms.
        styles: ``{variant: {colour, marker, label}}``.
        metric: Per-step column, without its ``pred_``/``truth_`` prefix.
        ylabel: Axis label, in the paper's notation.
        name_the_arms: Whether this panel contributes the legend's arm entries.
            Only one panel need do so; the legend is gathered across panels and
            deduplicated, but labelling once keeps the legend's order the arms'.
    """
    for variant, seeds in frames.items():
        steps, values = seed_curves(seeds, f"pred_{metric}")
        label = styles[variant]["label"] if name_the_arms else None
        draw_band(ax, steps, values, styles[variant], label=label)
    first = next(iter(frames.values()), None)
    if first is not None:
        draw_truth(ax, *seed_curves(first, f"truth_{metric}"), label=TRUTH_LABEL)
    ax.set_yscale("log")
    ax.set(xlabel=STEP_LABEL, ylabel=ylabel)


def panel_divergence(ax, frames, styles, name_the_arms=False):
    """Worst pointwise divergence of the predicted velocity, by rollout step."""
    panel_constraint(
        ax,
        frames,
        styles,
        "div_max_mean",
        r"$\max|\nabla\!\cdot\!\boldsymbol{u}|$",
        name_the_arms=name_the_arms,
    )


def panel_mean_velocity(ax, frames, styles):
    """Magnitude of the domain-mean velocity, by rollout step."""
    panel_constraint(
        ax, frames, styles, "domain_mean_speed_mean", r"$|\langle \boldsymbol{u} \rangle_\Omega|$"
    )


def panel_bounded(ax, traces, styles, horizon, name_the_arms=False):
    """Fraction of free-running trajectories still bounded, by rollout step."""
    steps = np.arange(1, horizon + 1)
    for variant, seeds in traces.items():
        values = np.stack(
            [bounded_fraction(blowup, steps) for _, (_, blowup) in sorted(seeds.items())]
        )
        label = styles[variant]["label"] if name_the_arms else None
        draw_band(ax, steps, values, styles[variant], label=label)
    ax.set_ylim(-0.04, 1.04)
    ax.set(xlabel=STEP_LABEL, ylabel="fraction with bounded energy")


def panel_energy_ratio(ax, traces, styles, horizon):
    """Median kinetic energy of a free-running trajectory over its initial energy."""
    steps = np.arange(1, horizon + 1)
    for variant, seeds in traces.items():
        values = np.stack(
            [
                np.median(padded_ratio(ratio, horizon), axis=1)
                for _, (ratio, _) in sorted(seeds.items())
            ]
        )
        # A wholly blown arm's band runs between two infinities, an interval
        # numpy's quantile reaches by an arithmetic that is undefined; the band is
        # then simply absent, which is what an off-scale curve should look like.
        with np.errstate(invalid="ignore"):
            draw_band(ax, steps, values, styles[variant])
    # Both bounds of the bounded set. The panel above counts a trajectory as
    # leaving it at either one, so an arm that drains away is no longer bounded
    # by the letter of the criterion.
    ax.axhline(BLOWUP_GUARD, label=rf"${BLOWUP_GUARD:.0f}\times$", **TRUTH_STYLE)
    ax.axhline(
        1.0 / BLOWUP_GUARD,
        label=rf"${1.0 / BLOWUP_GUARD:.1f}\times$",
        color=MUTED_INK,
        linestyle=(0, (3.5, 1.5)),
        linewidth=0.7,
    )
    ax.set_yscale("log")
    # An arm whose median trajectory has run away reaches 1e30 and would take the
    # axis with it, flattening the approach to the guard that is what this panel
    # is for. It leaves the top of the frame instead, an order past the guard, and
    # the panel above says when it crossed.
    ax.set_ylim(top=BLOWUP_GUARD * 10.0)
    ax.set(xlabel=STEP_LABEL, ylabel="median $E_n / E_0$")


def panel_training(ax, logs, epochs, styles):
    """Training and validation loss by epoch, with the epoch each run was taken from.

    Validation is measured every few epochs and the log carries a blank for the
    epochs in between, so the two curves live on different epoch axes and are
    stacked separately. Validation is the solid curve for every arm because it is
    the one the checkpoint selection reads; training is the same colour, dashed and
    drawn thin, and the ring is the epoch the selection kept.

    The selection is guarded, and reads validation only at the epochs whose input
    noise is at full amplitude. On E1 the ramp runs over the whole schedule, so
    the only eligible epoch is the last, and the ring sits well to the right of
    the curve's minimum. That is the guard working: an unguarded selection would
    take an epoch trained under little or no noise and quietly discard the
    stabilisation the campaign is built on.
    """
    for variant, seeds in logs.items():
        style = styles[variant]
        measured = {seed: frame.dropna(subset=["eval_l2"]) for seed, frame in seeds.items()}
        steps, values = seed_curves(measured, "eval_l2", index="epoch")
        # Solid against dashed is what this panel means by validation against
        # training, so it is the one figure where a stroke says which of the two
        # curves it is rather than which family the arm belongs to. Colour and
        # marker still name the arm.
        draw_band(
            ax, steps, values, {**style, "dash": "-"}, label=style["label"], spread=seed_range
        )
        train_steps, train_values = seed_curves(seeds, "train_l2", index="epoch")
        ax.plot(
            train_steps,
            np.median(train_values, axis=0),
            color=style["colour"],
            linewidth=0.6,
            linestyle=(0, (3, 1.5)),
            alpha=0.75,
        )
        chosen = epochs.get(variant)
        if chosen:
            nearest = np.abs(steps - np.median(list(chosen.values()))).argmin()
            ax.plot(
                steps[nearest],
                np.median(values, axis=0)[nearest],
                marker="o",
                markersize=MARKER_SIZE + 2.4,
                markerfacecolor="none",
                markeredgecolor=style["colour"],
                markeredgewidth=0.8,
                linestyle="none",
            )
    ax.set_yscale("log")
    ax.set(xlabel="epoch", ylabel="one-step relative $L_2$ loss")


def mark_wavenumbers(ax, panel):
    """The forcing scale and the two cutoffs, as references on a flux panel.

    Both are named inside the panel rather than in the shared legend, because the
    forcing sits at one wavenumber on one dataset and over a band on the other,
    and one legend entry cannot say both.
    """
    forcing = panel["forcing"]
    if len(forcing) == 1:
        ax.axvline(forcing[0], color="#1baf7a", linewidth=0.9, alpha=0.7)
    else:
        ax.axvspan(forcing[0], forcing[1], color="#1baf7a", alpha=0.14, linewidth=0)
    annotate_wavenumber(ax, max(forcing), "forcing", 0.02)
    # The two labels are staggered because on the E2 panel the cutoffs sit close
    # together against a wide axis and side by side they would run into each other.
    for index, cutoff in enumerate(FLUX_CUTOFFS):
        if cutoff > panel["k_max"]:
            continue
        ax.axvline(cutoff, color=MUTED_INK, linewidth=0.6, linestyle=(0, (1.2, 1.2)))
        annotate_wavenumber(ax, cutoff, f"$K={cutoff}$", 0.02 + 0.13 * (index + 1))


def annotate_wavenumber(ax, wavenumber, text, drop):
    """Name a vertical reference inside a panel, ``drop`` of the height below its top.

    The label sits on whichever side of its line leaves it inside the panel.
    """
    left, right = ax.get_xlim()
    inside = wavenumber < left + 0.75 * (right - left)
    ax.annotate(
        text,
        xy=(wavenumber, 1.0 - drop),
        xytext=(1.5 if inside else -1.5, 0.0),
        textcoords="offset points",
        xycoords=("data", "axes fraction"),
        ha="left" if inside else "right",
        va="top",
        fontsize=5.5,
        color=MUTED_INK,
    )


def panel_flux(ax, archive, panel, name_the_series=False):
    """Cumulative spectral energy flux and cumulative dissipation, against the cutoff.

    ``Pi(K)`` is the net nonlinear transfer into the band ``k <= K`` and
    ``D_in(K)`` the dissipation inside that band. Where the band carries no
    forcing the two coincide, so the gap between the curves is what the forcing
    puts in below ``K``.
    """
    wavenumbers = archive[f"{panel['prefix']}_k"]
    inside = wavenumbers <= panel["k_max"]
    for key, style in FLUX_SERIES.items():
        ax.plot(
            wavenumbers[inside],
            archive[f"{panel['prefix']}_{key}"][inside],
            color=style["colour"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=MARKER_SIZE,
            markeredgewidth=0.6,
            markevery=max(1, int(inside.sum()) // MARKERS_PER_CURVE),
            label=style["label"] if name_the_series else None,
        )
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.5)
    mark_wavenumbers(ax, panel)
    ax.set(xlabel="cutoff wavenumber $K$", ylabel="energy rate")


def panel_transfer(ax, archive, panel):
    """Shell energy transfer, the wavenumber-by-wavenumber derivative of the flux.

    ``T(k) = -dPi/dK`` by the convention the flux was measured under, so it is
    read off the cumulative curve rather than measured a second time.
    """
    wavenumbers = archive[f"{panel['prefix']}_k"]
    transfer = -np.diff(archive[f"{panel['prefix']}_Pi_mean"], prepend=0.0)
    inside = wavenumbers <= panel["k_max"]
    ax.plot(
        wavenumbers[inside],
        transfer[inside],
        color=FLUX_SERIES["Pi_mean"]["colour"],
        marker=FLUX_SERIES["Pi_mean"]["marker"],
        markersize=MARKER_SIZE,
        markeredgewidth=0.6,
        markevery=max(1, int(inside.sum()) // MARKERS_PER_CURVE),
    )
    ax.axhline(0.0, color=MUTED_INK, linewidth=0.5)
    mark_wavenumbers(ax, panel)
    ax.set(xlabel="wavenumber $k$", ylabel="$T(k)$")


def shared_legend(figure, **kwargs):
    """One legend for the whole figure, over every labelled curve on any panel.

    Identity never rests on colour alone: the legend is always present, and each
    entry carries the arm's marker and dash pattern beside its name. Labels are
    gathered across panels because the truth appears on some of them and not on
    others, and deduplicated so an arm drawn on four panels is named once.
    """
    entries = {}
    for axis in figure.axes:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            entries.setdefault(label, handle)
    return figure.legend(entries.values(), entries.keys(), frameon=False, **kwargs)


def figure_rollout(experiment, styles, series, path, width, band):
    """The rollout figure: accuracy, spectrum, and the two exact constraints.

    Two rows of two. Above, the relative L2 error against the truth by lead time
    and the energy spectrum at the end of the rollout, the latter carrying an
    inset over the ``band`` of wavenumbers the panel's own axis compresses;
    below, the worst pointwise divergence and the magnitude of the domain-mean
    velocity, each against the truth's own level as a dotted reference.
    """
    frames = ordered(experiment["frames"], series)
    spectra = ordered(experiment["spectra"], series)
    with paper_style():
        figure, axes = plt.subplots(2, 2, figsize=panel_grid(width, 2, 2, legend=0.44))
        panel_error(axes[0][0], frames, styles)
        panel_spectrum(axes[0][1], spectra, styles, band)
        panel_divergence(axes[1][0], frames, styles)
        panel_mean_velocity(axes[1][1], frames, styles)
        row_label(axes[0][0], experiment["label"])
        shared_legend(figure, loc="outside lower center", ncol=3)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)


def closed_panel_spectra(experiment, closed_runs, series, step):
    """Per-seed spectra at ``step`` for the arms of the closure figure.

    Three of the four arms are the campaign's own and are already read at that
    step. The closed arm's campaign evaluated under a pinned band cutoff, so its
    archives carry that cutoff's suffix; they are read here and filed under the
    arm's plain name, which is the name its style and the step-448 archive use.
    """
    spectra = dict(experiment["spectra"])
    for arm in series:
        if arm in spectra:
            continue
        suffixed = seed_spectra(closed_runs, [arm + CLOSED_SUFFIX], step=step)
        if suffixed:
            spectra[arm] = suffixed[arm + CLOSED_SUFFIX]
    return ordered(spectra, series)


def archived_median(archive, variant):
    """One arm's median step-448 spectrum out of the six-window archive.

    The archive stacks seeds on one axis and windows on another, with a NaN row
    for a rollout whose final state is non-finite, so the median at each
    wavenumber is taken over the finite rollouts alone.
    """
    key = f"pred_{variant}"
    if key not in archive:
        return None
    rows = archive[key].reshape(-1, archive[key].shape[-1])
    return np.nanmedian(rows, axis=0)


def draw_curve(ax, x, curve, style, label=None):
    """One arm's single curve, styled as its banded counterpart minus the band."""
    ax.plot(
        x,
        curve,
        color=style["colour"],
        linestyle=style.get("dash", "-"),
        marker=style["marker"],
        markersize=MARKER_SIZE,
        markeredgewidth=0.6,
        markevery=max(1, len(x) // MARKERS_PER_CURVE),
        label=label,
    )


def figure_closed_spectra(experiment, closure, styles, path, width):
    """The closure appendix figure: E2 energy spectra at steps 64 and 448.

    Two log-log panels on a shared vertical axis. The left panel is the end of
    the standard rollout, each arm the median over seeds of its trajectory-mean
    spectrum, with the truth's own step-64 spectrum dotted. The right panel is
    the end of the six with-truth 448-step windows, each arm the median over its
    thirty seed-window rollouts, with the truth's stationary mean spectrum ---
    the average over every snapshot of the test split --- as the reference,
    because a single step's truth no longer resembles the prediction there.
    """
    archive_path = Path(closure["archive"])
    closed_runs = Path(closure["runs"])
    if not archive_path.exists() or not closed_runs.exists():
        return False
    spectra = closed_panel_spectra(experiment, closed_runs, closure["series"], closure["step"])
    if not spectra:
        return False
    with np.load(archive_path) as archive, paper_style():
        figure, (early, late) = plt.subplots(
            1, 2, sharey=True, figsize=panel_grid(width, 2, 1, legend=0.38)
        )
        for variant, (wavenumbers, predicted, _) in spectra.items():
            curve = np.median(predicted, axis=0)
            draw_curve(early, wavenumbers, curve, styles[variant], label=styles[variant]["label"])
        first = next(iter(spectra.values()))
        early.plot(first[0], np.median(first[2], axis=0), label=TRUTH_LABEL, **TRUTH_STYLE)
        wavenumbers = archive["k"]
        for variant in spectra:
            curve = archived_median(archive, variant)
            if curve is not None:
                draw_curve(late, wavenumbers, curve, styles[variant])
        late.plot(wavenumbers, archive["truth_stationary_mean"], **TRUTH_STYLE)
        for ax, title in ((early, "step 64"), (late, "step 448")):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("wavenumber $k$")
            column_title(ax, title)
        early.set_ylabel("$E(k)$")
        shared_legend(figure, loc="outside lower center", ncol=3)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)
    return True


def figure_stability(experiments, styles, order, path, width):
    """Free-running stability, one column per experiment.

    The top row is the fraction of trajectories still bounded, which says when an
    arm fails; the bottom row is the median energy of a trajectory over its own
    initial energy, which says how: on E2 some arms leave the bounded set upwards
    and others by draining below a tenth of their initial energy, and only this
    row tells the two apart. The second row earns its space because the
    arms part on it long before the first trajectory reaches the guard: the
    fraction is still flat at one while the energies are already an order apart.

    The step axis is linear, and shared down each column. Linear is what the
    failures ask for:
    on E2 they run from step 96 to step 443 of 448, a fifth of the way across the
    panel to the far end of it, where a logarithmic axis would crowd every one of
    them into the last quarter and collapse the arms into a single cliff.
    """
    with paper_style():
        columns = len(experiments)
        # The rows do not share a y axis. The top one needs none --- a fraction is
        # already pinned to nought and one --- and on the bottom one the two
        # experiments genuinely differ by thirty decades once an arm runs away,
        # where one axis would flatten the other panel into a line.
        figure, axes = plt.subplots(
            2, columns, figsize=panel_grid(width, columns, 2, legend=0.50), squeeze=False
        )
        for column, experiment in enumerate(experiments):
            traces = ordered(experiment["free"], order)
            horizon = free_horizon(traces)
            # Every column names its arms. The legend is gathered across panels and
            # deduplicated, so an arm on both keeps one entry, and an arm that
            # lands on one campaign before the other is still named.
            panel_bounded(axes[0][column], traces, styles, horizon, name_the_arms=True)
            panel_energy_ratio(axes[1][column], traces, styles, horizon)
            column_title(axes[0][column], experiment["label"])
        shared_legend(figure, loc="outside lower center", ncol=4)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)


def style_key(ax, entries):
    """Name a line style in the shared legend, where no series carries it alone.

    The entries are drawn empty: they exist to put a handle in the legend, and a
    curve of no points adds nothing to the panel.
    """
    for label, dash in entries:
        ax.plot([], [], color=MUTED_INK, linewidth=0.9, linestyle=dash, label=label)


def figure_training(experiments, styles, order, path, width):
    """Training and validation loss by epoch, one panel per experiment.

    The ring on a validation curve is the epoch the guarded selection kept, which
    is the checkpoint every other figure and every table entry was measured from.
    """
    with paper_style():
        columns = len(experiments)
        figure, axes = plt.subplots(
            1, columns, figsize=panel_grid(width, columns, 1, legend=0.50), squeeze=False
        )
        for column, experiment in enumerate(experiments):
            panel_training(
                axes[0][column],
                ordered(experiment["logs"], order),
                experiment["epochs"],
                styles,
            )
            column_title(axes[0][column], experiment["label"])
        style_key(axes[0][0], [("validation", "-"), ("training", (0, (3, 1.5)))])
        shared_legend(figure, loc="outside lower center", ncol=4)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)


def figure_flux(archive, labels, path, width):
    """Spectral energy budget of the two datasets, one column each.

    Above, the cumulative flux into the band below the cutoff against the
    dissipation inside that band; below, the shell transfer the cumulative curve
    integrates. The forced wavenumber or forcing band is marked and so are both
    cutoffs, because where the forcing sits relative to the cutoff is the whole
    difference between the two datasets.
    """
    panels = {name: panel for name, panel in FLUX_PANELS.items() if f"{name}_k" in archive}
    with paper_style():
        columns = len(panels)
        figure, axes = plt.subplots(
            2, columns, figsize=panel_grid(width, columns, 2, legend=0.32), squeeze=False
        )
        for column, (name, panel) in enumerate(panels.items()):
            panel_flux(axes[0][column], archive, panel, name_the_series=column == 0)
            panel_transfer(axes[1][column], archive, panel)
            column_title(axes[0][column], labels.get(name, name))
        shared_legend(figure, loc="outside lower center", ncol=2)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)


def figure_residual(experiment, styles, order, path, width):
    """The vorticity residual of a rollout, against the truth's own residual.

    The vorticity formulation carries one residual and no velocity, so this is
    the whole of its constraint picture: how far the prediction sits from the
    vorticity equation, in multiples of the forcing term the residual is
    normalised by, beside the level the published data themselves sit at.
    """
    with paper_style():
        figure, axes = plt.subplots(1, 1, figsize=panel_grid(width, 1, 1, legend=0.34))
        panel_constraint(
            axes,
            ordered(experiment["frames"], order),
            styles,
            "cont_rel_mean",
            r"vorticity residual / $\Delta t\,f$",
            name_the_arms=True,
        )
        # The residual is a ratio of order one, not a quantity spanning decades.
        axes.set_yscale("linear")
        row_label(axes, experiment["label"])
        shared_legend(figure, loc="outside lower center", ncol=3)
        figure.savefig(path, metadata={"CreationDate": None})
        plt.close(figure)


# ------------------------------------------------------------------------------
# Tables.
#
# A column is a summary.csv column stem --- aggregate_seeds.py writes <stem>_mean,
# <stem>_median and <stem>_std for each --- a header, and how the cell is read.
# "spread" prints the mean with the seed standard deviation; "ratio" prints the
# mean over the same column's baseline mean inside the same experiment, which is
# what an overhead is.
#
# `never_blown_frac` is the one stem no summary carries: it is counted from the
# free-rollout archives and merged onto the summary before the table is
# rendered, because stability over a horizon with no ground truth is not an
# evaluation. Its header names both the criterion and the horizon --- the
# criterion because a campaign summary carries a `blowup_never_frac` of its own
# that means something else, and the horizon because it differs by campaign.
HORIZON_FIELD = "<steps>"

COLUMNS = {
    "l2_rollout_mean": (r"rel.\ $L_2$", "spread"),
    "l2_step1": ("step 1", "spread"),
    "l2_step16": ("step 16", "spread"),
    "l2_step32": ("step 32", "spread"),
    "l2_step64": ("step 64", "spread"),
    "div_max": (r"\shortstack{max\\$|\diver \vec{u}|$}", "spread"),
    "mean_speed_final": (r"\shortstack{$|\langle \boldsymbol{u} \rangle_\Omega|$}", "spread"),
    "never_blown_frac": (r"\shortstack{energy bounded\\(<steps> steps)}", "spread"),
    "seconds_per_epoch": ("train", "ratio"),
    "inference_s_per_trajectory": ("infer.", "ratio"),
    "tto_s_per_trajectory": (r"\shortstack{TTO\\(s)}", "spread"),
}

# Overheads are quoted against this arm of the same experiment, as the caption of
# Table 1 says.
COST_BASELINE = "FNO"

# The columns that report how close a rollout stays to the truth. They are the
# ones an arm can be held out of leading, because an arm that trades accuracy for
# a constraint it was built to probe is not competing on them.
ACCURACY_COLUMNS = frozenset({"l2_rollout_mean", "l2_step1", "l2_step16", "l2_step32", "l2_step64"})

# Every other column is an error, a violation or a cost, whose leader is the
# smallest entry; in these the leader is the largest.
HIGHER_IS_BETTER = frozenset({"never_blown_frac"})

# A number outside this window is written with a power of ten rather than as a
# run of leading or trailing zeros.
FIXED_RANGE = (1e-2, 1e3)

MISSING = "--"

# The mark an arm held out of leading an accuracy column carries, and the note
# the table sets under itself to say what the mark means. The note is set in a
# paragraph column of the text block's own width, so that it wraps rather than
# setting a floor under how narrow the table can be.
EXCLUDED_MARK = r"$^{\dagger}$"
EXCLUDED_NOTE = r"$^{\dagger}$Held out of leading the accuracy columns."


def _places_for(spread, decimals, extra=2):
    """Decimal places for a spread, so a small one does not print as zero.

    Starts at the precision the mean is printed to and adds up to ``extra``
    places, which covers a spread two orders below its mean. A spread smaller
    than that still rounds to zero: the cap keeps the column narrow, and a
    reader of a table printed to two significant figures is not owed a third.
    """
    for places in range(decimals, decimals + extra + 1):
        if spread == 0.0 or round(spread, places) != 0.0:
            return places
    return decimals + extra


def _stacked(top, bottom=r"\strut", align=""):
    r"""A two-line table cell.

    Every cell in these tables is stacked, including the one-line ones, because
    ``\shortstack`` puts a cell's baseline on its last line: a plain cell in a row
    of stacks would sit beside the spreads rather than beside the means. A strut
    fills the second line where there is nothing to put there.
    """
    return rf"\shortstack{align}{{{top}\\{bottom}}}"


def _maths(body, bold=False):
    """One maths group, with its digits emboldened when it is the best in its column."""
    return rf"$\mathbf{{{body}}}$" if bold else f"${body}$"


def spread_places(mean, std):
    """Decimal places one fixed-notation entry takes, or ``None`` where it takes none.

    ``None`` covers the entries that do not share a column's decimal place: one
    that is missing, and one whose magnitude puts it in exponent notation.
    """
    if mean is None or std is None or not np.isfinite(mean) or not np.isfinite(std):
        return None
    if not (FIXED_RANGE[0] <= abs(mean) < FIXED_RANGE[1] or mean == 0.0):
        return None
    decimals = 0 if mean == 0.0 else max(0, 1 - int(np.floor(np.log10(abs(mean)))))
    return _places_for(abs(std), decimals)


def column_places(means, spreads):
    """The decimal place a whole column sets, so its entries line up under each other.

    It is the widest any one entry needs. A column that mixed three decimals with
    four would leave a reader comparing digits that do not sit above one another.
    """
    places = [spread_places(mean, std) for mean, std in zip(means, spreads, strict=True)]
    return max((place for place in places if place is not None), default=None)


def printed_value(mean, std, places=None):
    """The number a cell shows, which is the number a bold mark is a claim about.

    Two entries that print the same digits are level however far apart the means
    behind them are. A table that emboldens one of two cells both reading
    ``2.6`` is read as a typesetting slip rather than as a result, and the
    difference it is marking is one the printed precision does not carry.
    """
    if mean is None or not np.isfinite(mean):
        return mean
    if FIXED_RANGE[0] <= abs(mean) < FIXED_RANGE[1] or mean == 0.0:
        decimals = places if places is not None else spread_places(mean, std)
        return float(f"{mean:#.2g}") if decimals is None else round(mean, decimals)
    # Outside the fixed window the cell shows the mean over a shared power of ten,
    # to one decimal place.
    scale = 10.0 ** int(np.floor(np.log10(abs(mean))))
    return round(mean / scale, 1) * scale


def format_spread(mean, std, bold=False, places=None):
    r"""A mean with its seed standard deviation, as one two-line LaTeX cell.

    The mean and the spread sit on the first line and any shared power of ten on
    the second. Both are as narrow as the numbers allow, because the table is
    wide: at the settings Table 1 uses --- \footnotesize, \tabcolsep 2pt, both
    experiments filled --- its ten numeric columns measure 391.9pt against a
    397.5pt text block, which is what lets it be set upright and unscaled. Only the first
    line is emboldened, because a bold power of ten is noise rather than a signal.

    The mean carries at least two significant figures, and as many more as its own
    spread needs before it stops rounding to zero: an arm reported to two figures
    alone loses its ordering at the top of a decade, where a step-64 error of 1.11
    and one of 1.07 both print as 1.1. Quoting the mean to the decade of its
    spread is the precision five seeds actually support, and ``_places_for`` caps
    how far the column can widen. ``places`` overrides that with the place the
    whole column settled on, so the entries line up. The plus-minus is set tight
    because the thick spaces around it cost the table four points a cell.

    The three ways a value can be absent are kept apart, as ``aggregate_seeds.py``
    keeps them apart: a metric never recorded prints the missing marker, one that
    overflowed prints an infinity, and a mean whose spread is unavailable prints
    alone rather than beside a fabricated zero.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return _stacked(MISSING)
    if np.isinf(mean):
        return _stacked(_maths(r"\infty" if mean > 0 else r"-\infty", bold))
    if std is None or not np.isfinite(std):
        return _stacked(_maths(f"{mean:#.2g}", bold))
    spread = abs(std)
    if FIXED_RANGE[0] <= abs(mean) < FIXED_RANGE[1] or mean == 0.0:
        # Both numbers take the same decimal place, so they line up and neither
        # slips into exponent notation: the place its own spread needs, or the
        # place the whole column agreed on where a caller passed one.
        decimals = places if places is not None else spread_places(mean, std)
        top = _maths(f"{mean:.{decimals}f}", bold)
        bottom = _maths(rf"\pm{spread:.{decimals}f}")
    else:
        exponent = int(np.floor(np.log10(abs(mean))))
        scale = 10.0**exponent
        scaled = spread / scale
        top = _maths(rf"{mean / scale:.1f}{{\pm}}{scaled:.{_places_for(scaled, 1)}f}", bold)
        bottom = _maths(rf"\times 10^{{{exponent}}}")
    return _stacked(top, bottom)


def format_ratio(mean, baseline, bold=False):
    """A cost as a multiple of the baseline arm's cost.

    The ratio is taken between the two seed means, because the seeds of two arms
    are not paired for wall-clock time and a spread on the ratio would be read as
    one that is. A baseline of zero has no multiple, so the cell reports nothing.
    """
    if mean is None or baseline is None or not np.isfinite(mean) or not np.isfinite(baseline):
        return _stacked(MISSING)
    if baseline == 0.0:
        return _stacked(MISSING)
    ratio = mean / baseline
    shown = f"{ratio:.0f}" if abs(ratio) >= 100 else f"{ratio:.2f}"
    return _stacked(_maths(rf"{shown}\times", bold))


def summary_value(summary, model, column, statistic):
    """One statistic of one metric for one arm, or ``None`` when it is absent."""
    if summary is None:
        return None
    row = summary[summary["model"] == model]
    name = f"{column}_{statistic}"
    if row.empty or name not in row:
        return None
    value = float(row.iloc[0][name])
    return None if np.isnan(value) else value


def leading_indices(means, spreads, eligible, sign=1.0, strict=True):
    """Which rows lead a column.

    Args:
        means: The value each row shows, one per row, ``None`` where the row
            carries no number.
        spreads: The seed standard deviations, read only by the lenient rule.
        eligible: Which rows may lead at all.
        sign: ``1`` where the leader is the smallest entry, ``-1`` where it is
            the largest.
        strict: Mark the leader alone, together with any row exactly level with
            it. The lenient alternative marks every row within the seed spread of
            the leader --- level meaning the gap between two means is no larger
            than the wider of their two seed standard deviations --- on the
            argument that the paper claims no difference smaller than the seed
            spread, so a column that cannot separate its arms marks all of them
            rather than picks one. Which rule the table runs under is the
            author's call, which is why it is a switch and not a constant.

    Returns:
        The indices to embolden.
    """
    contenders = [
        index
        for index, mean in enumerate(means)
        if eligible[index] and mean is not None and np.isfinite(mean)
    ]
    if not contenders:
        return set()
    best = min(contenders, key=lambda index: sign * means[index])
    if strict:
        return {index for index in contenders if means[index] == means[best]}

    def spread(index):
        value = spreads[index]
        return abs(value) if value is not None and np.isfinite(value) else 0.0

    return {
        index
        for index in contenders
        if sign * (means[index] - means[best]) <= max(spread(best), spread(index))
    }


def table_cells(summary, rows, columns, strict=True, excluded=()):
    """The rendered cells of one column group, as ``{column: [cell per row]}``.

    ``leading_indices`` decides which entries are marked, over the values as they
    print rather than as they are stored. An accuracy column excludes the arms
    ``excluded`` names. An overhead column marks nothing at all: its own baseline
    arm is one by construction, so the mark would land on a number larger than the
    unmarked one beside it, and "the cheapest arm that is not the baseline" is not
    a claim the paper makes.
    """
    rendered = {}
    for column in columns:
        _, kind = COLUMNS[column]

        def metric(model, column=column):
            # TTO inference is the optimiser wall clock, not the unadapted rollout
            # the summary also stores under ``inference_s_per_trajectory``.
            if column == "inference_s_per_trajectory" and model.endswith("_tto"):
                return "tto_s_per_trajectory"
            return column

        means = [summary_value(summary, model, metric(model), "mean") for model in rows]
        spreads = [summary_value(summary, model, metric(model), "std") for model in rows]
        eligible = [
            kind != "ratio" and (column not in ACCURACY_COLUMNS or model not in excluded)
            for model in rows
        ]
        baseline = None
        if kind == "ratio":
            baseline = summary_value(summary, COST_BASELINE, column, "mean")
            spreads = [None] * len(rows)
        places = column_places(means, spreads)
        shown = [printed_value(mean, std, places) for mean, std in zip(means, spreads, strict=True)]
        sign = -1.0 if column in HIGHER_IS_BETTER else 1.0
        leaders = leading_indices(shown, spreads, eligible, sign=sign, strict=strict)
        cells = []
        for index, mean in enumerate(means):
            bold = index in leaders
            if kind == "ratio":
                cells.append(format_ratio(mean, baseline, bold=bold))
            else:
                cells.append(format_spread(mean, spreads[index], bold=bold, places=places))
        rendered[column] = cells
    return rendered


def column_header(column, horizon=None):
    """The printed header of one column, carrying the horizon a stability column names.

    The substitution is a plain replacement rather than a format call, because a
    header is LaTeX and its braces are the argument of a macro.
    """
    return COLUMNS[column][0].replace(HORIZON_FIELD, str(horizon))


def latex_table(spec, summaries, experiment_labels, model_labels, horizons=None):
    r"""One table body: the ``tabular`` environment, ready to ``\input``.

    Args:
        spec: A ``[tables.*]`` entry of the configuration.
        summaries: ``{experiment: DataFrame or None}`` of the seed summaries.
        experiment_labels: ``{experiment: printed name}``.
        model_labels: ``{variant: printed name}``.
        horizons: ``{experiment: free-rollout horizon}``, for the header of a
            stability column.

    Returns:
        The LaTeX source, ending in a newline.
    """
    rows = spec["rows"]
    horizons = horizons or {}
    excluded = tuple(spec.get("exclude_from_leading", []))
    strict = spec.get("bold", "strict") == "strict"
    groups = [
        (name, experiment_labels[name], summaries.get(name), spec["columns"])
        for name in spec["experiments"]
    ]

    widths = [len(columns) for _, _, _, columns in groups]
    header = ["l", *["c" * width for width in widths]]
    rule = (
        "bold marks the strict leader of a column, and any arm exactly level with it"
        if strict
        else "bold marks the leader of a column and every arm within the seed spread of it"
    )
    lines = [
        "% Generated by figures/make_figures.py from the campaign summaries.",
        "% Do not edit by hand: rerun the script when a campaign is re-aggregated.",
        "% Entries are the mean over seeds with the seed standard deviation.",
        "%",
        f"% Bolding rule: {rule}.",
        "% Every column is an error, a constraint violation or a cost, so the leader is the",
        "% smallest mean, and arms level as printed are marked together. An overhead column",
        f"% is not marked at all: it is quoted against {COST_BASELINE} of the same experiment,",
        "% whose own entry is one by construction.",
        "%",
        f"% A metric an experiment does not measure, or has not run, prints {MISSING}; one",
        "% that overflowed prints an infinity; a mean whose spread is unavailable prints",
        "% on its own.",
    ]
    if any(column in HIGHER_IS_BETTER for _, _, _, columns in groups for column in columns):
        lines.append("% The stability column is the exception: there the leader is the largest.")
    if excluded:
        lines.append(
            "% Held out of leading an accuracy column, and marked with a dagger: "
            + ", ".join(model_labels[model] for model in excluded)
        )
    lines += [
        rf"\begin{{tabular}}{{@{{}}{' '.join(header)}@{{}}}}",
        r"  \toprule",
    ]

    spans, first = [], 2
    for (_, title, _, _), width in zip(groups, widths, strict=True):
        spans.append((title, first, first + width - 1))
        first += width
    lines.append(
        "  & "
        + " & ".join(
            rf"\multicolumn{{{stop - start + 1}}}{{c}}{{{title}}}" for title, start, stop in spans
        )
        + r" \\"
    )
    lines.append("  " + " ".join(rf"\cmidrule(lr){{{start}-{stop}}}" for _, start, stop in spans))
    lines.append(
        "  Model & "
        + " & ".join(
            column_header(column, horizons.get(name))
            for name, _, _, columns in groups
            for column in columns
        )
        + r" \\"
    )
    lines.append(r"  \midrule")

    rendered = [
        table_cells(summary, rows, columns, strict=strict, excluded=excluded)
        for _, _, summary, columns in groups
    ]
    for index, model in enumerate(rows):
        cells = [
            group[column][index]
            for group, (_, _, _, columns) in zip(rendered, groups, strict=True)
            for column in columns
        ]
        mark = EXCLUDED_MARK if model in excluded else ""
        name = _stacked(f"{model_labels[model]}{mark}", align="[l]")
        lines.append(f"  {name} & " + " & ".join(cells) + r" \\")
        if model in spec.get("midrule_after", []):
            lines.append(r"  \midrule")

    lines.append(r"  \bottomrule")
    if excluded:
        lines.append(
            rf"  \multicolumn{{{sum(widths) + 1}}}{{@{{}}p{{\textwidth}}@{{}}}}"
            rf"{{\footnotesize {EXCLUDED_NOTE}}} \\"
        )
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def stability_rows(traces, horizon):
    """The never-blown-up fraction of every arm, shaped like a summary frame.

    Stability over a horizon with no ground truth is counted from the free
    rollouts rather than measured by an evaluation, so it reaches the table by
    being merged onto the summary under a column stem of its own.
    """
    records = []
    for variant, seeds in traces.items():
        fractions = np.array(
            [bounded_fraction(blowup, [horizon])[0] for _, (_, blowup) in sorted(seeds.items())]
        )
        records.append(
            {
                "model": variant,
                "never_blown_frac_mean": float(fractions.mean()),
                # One seed has no spread; a zero here would read as seeds that agreed.
                "never_blown_frac_std": float(fractions.std(ddof=1))
                if fractions.size > 1
                else float("nan"),
            }
        )
    return pd.DataFrame(records)


def with_stability(summary, traces, horizon):
    """A summary frame carrying the never-blown-up fraction of each of its arms."""
    counted = stability_rows(traces, horizon)
    if summary is None or counted.empty:
        return summary
    return summary.merge(counted, on="model", how="left")


# ------------------------------------------------------------------------------


def load_experiment(name, config, variants):
    """Everything one campaign contributes, or ``None`` when it has not landed.

    Every arm the configuration knows about is read and each figure takes the
    subset it draws, so an arm that lands later appears wherever it belongs
    without a second pass over the campaign.

    Returns:
        A dictionary with the campaign's printed ``label``, its ``summary``
        frame, the per-step ``frames`` of its arms, their end-of-rollout
        ``spectra``, the ``free`` running energy traces, the per-epoch ``logs``
        and the selected ``epochs``; ``None`` if neither its summary nor its runs
        exist.
    """
    entry = config["experiments"][name]
    summary_path, runs_dir = Path(entry["summary"]), Path(entry["runs"])
    if not summary_path.exists() and not runs_dir.is_dir():
        return None
    frames = read_variants(runs_dir, variants) if runs_dir.is_dir() else {}
    return {
        "name": name,
        "label": entry["label"],
        "rollout_figure": entry.get("rollout_figure"),
        "summary": pd.read_csv(summary_path) if summary_path.exists() else None,
        "frames": frames,
        "spectra": seed_spectra(runs_dir, list(frames), step=entry.get("spectrum_step"))
        if frames
        else {},
        "free": free_traces(entry.get("free", [])),
        "logs": training_logs(runs_dir) if runs_dir.is_dir() else {},
        "epochs": selected_epochs(runs_dir) if runs_dir.is_dir() else {},
    }


def main(argv=None):
    """Write the paper's figures and table bodies, reporting what each one used."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="TOML inputs and styling"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_CONFIG.parent,
        help="directory the PDFs and table bodies are written to",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    figures = config["figures"]
    styles = style_entries(config["models"])
    arms = list(config["models"])
    experiment_labels = {name: entry["label"] for name, entry in config["experiments"].items()}
    model_labels = {key: entry["label"] for key, entry in config["models"].items()}

    experiments = {}
    for name in config["experiments"]:
        loaded = load_experiment(name, config, arms)
        if loaded is None:
            print(f"{name}: no campaign at its configured paths yet -- skipped")
            continue
        experiments[name] = loaded
        print(
            f"{name}: {len(loaded['frames'])} arms measured, spectra for "
            f"{len(loaded['spectra'])}, free rollouts for {len(loaded['free'])}, "
            f"training logs for {len(loaded['logs'])}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    width = figures["width_in"]

    for experiment in experiments.values():
        if not (experiment["rollout_figure"] and experiment["frames"]):
            continue
        path = args.out_dir / experiment["rollout_figure"]
        figure_rollout(
            experiment, styles, figures["series"], path, width, figures["spectrum_inset"]
        )
        print(f"wrote {path.name} from {experiment['name']}")

    stability = [
        experiments[name]
        for name in figures["stability_experiments"]
        if name in experiments and experiments[name]["free"]
    ]
    if stability:
        path = args.out_dir / figures["stability_output"]
        figure_stability(stability, styles, arms, path, width)
        print(f"wrote {path.name} from {', '.join(row['name'] for row in stability)}")

    training = [
        experiments[name]
        for name in figures["training_experiments"]
        if name in experiments and experiments[name]["logs"]
    ]
    if training:
        path = args.out_dir / figures["training_output"]
        figure_training(training, styles, arms, path, width)
        print(f"wrote {path.name} from {', '.join(row['name'] for row in training)}")

    residual = figures["residual_experiment"]
    if residual in experiments and experiments[residual]["frames"]:
        path = args.out_dir / figures["residual_output"]
        figure_residual(experiments[residual], styles, arms, path, figures["narrow_width_in"])
        print(f"wrote {path.name} from {residual}")

    source = Path(figures["flux_source"])
    if source.exists():
        path = args.out_dir / figures["flux_output"]
        with np.load(source) as archive:
            figure_flux(archive, experiment_labels, path, width)
        print(f"wrote {path.name} from {source}")
    else:
        print(f"flux: no archive at {source} yet -- skipped")

    closure = config.get("closure")
    if closure and "E3" in experiments:
        path = args.out_dir / closure["output"]
        if figure_closed_spectra(experiments["E3"], closure, styles, path, width):
            print(f"wrote {path.name} from E3 and {closure['archive']}")
        else:
            print("closure: archive or closed-campaign runs missing -- skipped")

    horizons = {name: free_horizon(entry["free"]) for name, entry in experiments.items()}
    summaries = {
        name: with_stability(entry["summary"], entry["free"], horizons[name])
        for name, entry in experiments.items()
    }
    for spec in config["tables"].values():
        path = args.out_dir / spec["output"]
        path.write_text(
            latex_table(spec, summaries, experiment_labels, model_labels, horizons=horizons)
        )
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
