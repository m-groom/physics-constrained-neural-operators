r"""Vorticity snapshots at the end of the rollout, one figure per experiment.

Reads the vorticity trajectories ``test_operator_AR_2d.py`` already writes for the first
test trajectory of every run --- ``saved_plots/<model>_seed<seed>_vorticity_sample0*.npz``,
holding ``omega = dv/dx - du/dy`` of the prediction and of the truth at every rollout level
in physical units --- and draws the truth beside every arm at one rollout step. Nothing here
re-runs a model, so a panel is the same field the run's reported numbers were measured on.

E1 shows the end of the T = 64 rollout. E3 shows step 64 and step 256 of the 448-step
with-truth window, whose evaluation wrote its own archive: step 256 is where the projected
arms are still inside the truth's vorticity range and the two unprojected ones are not. The
paper calls E3 "E2", so the E3 figure is written under that name.

An arm the campaign archived no vorticity for is left out and named on the console, because
every row of a figure carries the same panels in the same order.

Every panel of a figure shares one diverging colour scale, symmetric about zero and set from
the truth's own vorticity on that figure, at the 99.5th percentile of its magnitude. An arm
that has left that range is drawn saturated rather than given a scale of its own, because the
point of the panel is how far it has left the truth; the extremes themselves are quoted in
the caption.

Run from the repository root:

    uv run --no-sync python figures/make_snapshots.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent))

# The paper's figure styling and the arm labels belong to the figure pipeline and are
# already written down once, in make_figures.py and figures.toml beside it.
from make_figures import DEFAULT_CONFIG, MUTED_INK, load_config, paper_style

# Vorticity is signed, so the ramp is diverging: two hues about a near-neutral midpoint,
# never a rainbow and never a hue in the middle.
COLOURMAP = "RdBu_r"

# The archive one evaluation writes per run, for the first test trajectory.
ARCHIVE_STEM = "{model}_seed{seed}_vorticity_sample0{suffix}"

# The marker an arm's name carries when it is the test-time-optimised twin of
# another arm. The evaluation puts the same marker at the end of the file name,
# after the evaluation's own suffix.
TTO_MARKER = "_tto"

TRUTH_LABEL = "truth"


def archive_path(runs_dir, arm, seed, suffix):
    """Where the evaluation wrote one arm's vorticity trajectory, or ``None``.

    A run directory is named for the checkpoint and a file for the variant that
    used it, and the two differ: a test-time-optimised rollout is written into the
    run directory of the checkpoint it adapted, under a name carrying the
    evaluation's own suffix and then the ``_tto`` marker. The archive is therefore
    found by its own name wherever it sits, which is the rule the rest of the
    figure pipeline reads run files by.

    Args:
        runs_dir: Directory of run directories.
        arm: Variant name as the configuration lists it.
        seed: Seed whose run is read.
        suffix: The evaluation's file suffix, which the arm's own marker follows.

    Returns:
        The archive's path, or ``None`` where the campaign never wrote one.

    Raises:
        ValueError: If two run directories claim the same archive, which would
            make the choice between them silent.
    """
    model, marker = (arm[: -len(TTO_MARKER)], TTO_MARKER) if arm.endswith(TTO_MARKER) else (arm, "")
    stem = ARCHIVE_STEM.format(model=model, seed=seed, suffix=f"{suffix}{marker}")
    found = sorted(Path(runs_dir).glob(f"*/saved_plots/{stem}.npz"))
    if len(found) > 1:
        raise ValueError(f"{stem}.npz is claimed by {', '.join(str(path) for path in found)}")
    return found[0] if found else None


def available_arms(runs_dir, arms, seed, steps):
    """Split the arms into those archived at every step of a figure and those not.

    Every row of a figure carries the same panels in the same order, so an arm
    missing one of the steps is left out of all of them rather than leaving a hole
    in one row. An arm a campaign never archived is reported rather than drawn
    empty, so "not run" is told apart from "run and measured".

    Returns:
        ``(present, missing)``, each in the order ``arms`` gives.
    """
    present = [
        arm
        for arm in arms
        if all(archive_path(runs_dir, arm, seed, suffix) is not None for _, suffix in steps)
    ]
    return present, [arm for arm in arms if arm not in present]


def read_snapshot(path, step, key):
    """One time level of an archived vorticity trajectory, shape ``(S, S)``.

    Args:
        path: The archive written beside the run.
        step: Rollout step to read; level 0 is the initial condition.
        key: ``"vorticity_pred"`` or ``"vorticity_truth"``.

    Returns:
        np.ndarray of shape ``(S, S)``, indexed ``(x, y)``, in physical units.

    Raises:
        ValueError: The rollout in that archive never reached ``step``.
    """
    with np.load(path) as archive:
        field = archive[key]
    levels = field.shape[-1]
    if not 0 <= step < levels:
        raise ValueError(f"{path} covers steps 0-{levels - 1}, not {step}")
    return field[..., step]


def symmetric_limit(fields, quantile=1.0):
    """A colour limit about zero, from the finite magnitudes over ``fields``.

    Vorticity is heavy-tailed, so the extreme of a field is a poor scale for the field:
    it spends the whole ramp on a handful of cells and leaves the rest near the neutral
    midpoint. ``quantile`` below one clips that tail instead.

    Args:
        fields: The reference fields, here the truth panels of one figure.
        quantile: Quantile of ``|omega|`` the limit sits at; ``1.0`` is the extreme.

    Returns:
        The limit, so the scale runs from ``-limit`` to ``+limit``.

    Raises:
        ValueError: No field carries a finite value to set a scale from.
    """
    finite = [
        np.abs(values[np.isfinite(values)])
        for values in (np.asarray(field, dtype=float) for field in fields)
    ]
    populated = [values for values in finite if values.size]
    if not populated:
        raise ValueError("no finite vorticity to set a colour scale from")
    return float(np.quantile(np.concatenate([v.ravel() for v in populated]), quantile))


def wrap_label(label):
    """An arm label broken so it fits over a panel at the typeset width.

    With both adapted arms archived a row carries nine panels, and the longest label
    is three parts: the family, its qualifier and the ``+ TTO`` marker. Each part
    takes a line of its own, so no title reaches into its neighbour's panel.
    """
    return label.replace(" (", "\n(").replace(" + ", "\n+ ")


def draw_snapshots(rows, path, width_in, limit):
    """Draw one row of panels per rollout step and save the figure.

    Args:
        rows: ``[(row_label, [(panel_label, field), ...]), ...]``; every row carries the
            same panels in the same order, the truth first.
        path: Where the PDF is written.
        width_in: Width of the typeset figure in inches, so the 7pt labels arrive on the
            page at 7pt.
        limit: Colour limit; the scale runs from ``-limit`` to ``+limit``.

    Returns:
        The figure, so a caller can inspect it before it is closed.
    """
    n_columns = len(rows[0][1])
    panel = width_in / n_columns
    # Titles run to two lines, and the colour bar and its label sit under the grid.
    furniture = 0.30 + 0.42
    with paper_style():
        figure, axes = plt.subplots(
            len(rows),
            n_columns,
            figsize=(width_in, len(rows) * panel + furniture),
            squeeze=False,
        )
        # Panels of one figure are one picture: they touch, so the eye compares fields
        # rather than crossing gutters.
        figure.set_layout_engine("constrained", w_pad=0.01, h_pad=0.01, wspace=0.012, hspace=0.02)
        # The bar describes the scale every panel is drawn on, not the last panel drawn.
        scale = ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=COLOURMAP)
        for row_index, (row_label, panels) in enumerate(rows):
            for column, (panel_label, field) in enumerate(panels):
                axis = axes[row_index][column]
                axis.imshow(
                    np.asarray(field, dtype=float).T,
                    origin="lower",
                    cmap=COLOURMAP,
                    norm=scale.norm,
                    interpolation="nearest",
                )
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                if row_index == 0:
                    axis.set_title(wrap_label(panel_label))
                if column == 0:
                    axis.set_ylabel(row_label, color=MUTED_INK)
        bar = figure.colorbar(
            scale,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            fraction=0.09,
            aspect=45,
            shrink=0.55,
            pad=0.01,
        )
        bar.set_label("vorticity $\\omega$")
        bar.outline.set_linewidth(0.4)
        # Two-line titles overrun a canvas sized for the panels alone, and a clipped
        # glyph is not in the file for LaTeX to recover: the box follows the artwork.
        # No creation date, so a regeneration that changes nothing leaves the file
        # alone rather than a timestamp diff, as make_figures.py already does.
        figure.savefig(
            path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.01,
            metadata={"CreationDate": None},
        )
    return figure


def build_rows(runs_dir, arms, labels, seed, steps):
    """The panels of one experiment's figure, truth first in every row."""
    rows = []
    for step, suffix in steps:
        panels = [
            (
                TRUTH_LABEL,
                read_snapshot(
                    archive_path(runs_dir, arms[0], seed, suffix), step, "vorticity_truth"
                ),
            )
        ]
        for arm in arms:
            path = archive_path(runs_dir, arm, seed, suffix)
            panels.append((labels.get(arm, arm), read_snapshot(path, step, "vorticity_pred")))
        rows.append((f"step {step}", panels))
    return rows


def main():
    """Write one snapshot figure per experiment named in the configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    config = load_config(args.config)
    snapshots = config["snapshots"]
    labels = {key: entry["label"] for key, entry in config["models"].items()}
    for experiment, settings in sorted(snapshots.items()):
        if not isinstance(settings, dict):
            continue
        runs_dir = config["experiments"][experiment]["runs"]
        if not Path(runs_dir).exists():
            print(f"{experiment}: no runs at {runs_dir}, skipped")
            continue
        steps = [tuple(step) for step in settings["steps"]]
        arms, missing = available_arms(runs_dir, snapshots["arms"], snapshots["seed"], steps)
        if missing:
            print(f"{experiment}: no vorticity archive for {', '.join(missing)}, left out")
        if not arms:
            print(f"{experiment}: no arm archived at every step, skipped")
            continue
        rows = build_rows(runs_dir, arms, labels, snapshots["seed"], steps)
        # The scale belongs to the truth: an arm that has left it is the finding, not the
        # reason to rescale every other panel.
        limit = symmetric_limit([panels[0][1] for _, panels in rows], quantile=0.995)
        path = args.outdir / settings["output"]
        figure = draw_snapshots(rows, path, snapshots["width_in"], limit)
        plt.close(figure)
        print(f"{experiment}: wrote {path} (scale +/-{limit:.3g})")


if __name__ == "__main__":
    main()
