import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def append_file_suffix(stem, suffix):
    return f"{stem}{suffix}" if suffix else stem


def find_single_file(root_dir, pattern):
    matches = sorted(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} found under {root_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous match for {pattern!r} under {root_dir}: {matches}")
    return matches[0]


def extract_spectra_step(path):
    stem = Path(path).stem
    match = re.search(r"_spectra_t(\d+)", stem)
    if match is None:
        raise ValueError(f"Could not parse spectra step from {path}")
    return int(match.group(1))


def update_bad_mask_from_array(bad_mask, arr):
    if arr.ndim == 0:
        return
    if arr.shape[0] != bad_mask.shape[0]:
        return
    if not np.issubdtype(arr.dtype, np.number):
        return
    axis = tuple(range(1, arr.ndim))
    finite_rows = np.isfinite(arr) if not axis else np.isfinite(arr).all(axis=axis)
    bad_mask |= ~finite_rows


def build_bad_sample_mask(per_sample_npz, spectra_paths):
    num_samples = per_sample_npz["l2"].shape[0]
    bad_mask = np.zeros(num_samples, dtype=bool)

    for key in per_sample_npz.files:
        if key == "step":
            continue
        update_bad_mask_from_array(bad_mask, per_sample_npz[key])

    for path in spectra_paths:
        spectra_npz = np.load(path)
        for key in spectra_npz.files:
            if key == "k_bins":
                continue
            update_bad_mask_from_array(bad_mask, spectra_npz[key])

    return bad_mask


def save_filtered_per_sample_npz(per_sample_npz, keep_mask, output_path):
    filtered = {}
    for key in per_sample_npz.files:
        arr = per_sample_npz[key]
        if arr.ndim > 0 and arr.shape[0] == keep_mask.shape[0] and key != "step":
            filtered[key] = arr[keep_mask]
        else:
            filtered[key] = arr
    np.savez(output_path, **filtered)


def save_filtered_per_step_csv(per_sample_npz, keep_mask, output_path):
    valid = {key: per_sample_npz[key][keep_mask] for key in per_sample_npz.files if key != "step"}
    num_steps = valid["l2"].shape[1]
    df_per_step = pd.DataFrame(
        {
            "step": np.arange(1, num_steps + 1),
            "l2_mean": valid["l2"].mean(axis=0),
            "l2_std": valid["l2"].std(axis=0),
            "pred_cont_abs_mean": valid["pred_cont_abs"].mean(axis=0),
            "pred_cont_abs_std": valid["pred_cont_abs"].std(axis=0),
            "pred_momx_abs_mean": valid["pred_momx_abs"].mean(axis=0),
            "pred_momx_abs_std": valid["pred_momx_abs"].std(axis=0),
            "pred_momy_abs_mean": valid["pred_momy_abs"].mean(axis=0),
            "pred_momy_abs_std": valid["pred_momy_abs"].std(axis=0),
            "pred_cont_rel_mean": valid["pred_cont_rel"].mean(axis=0),
            "pred_cont_rel_std": valid["pred_cont_rel"].std(axis=0),
            "pred_momx_rel_mean": valid["pred_momx_rel"].mean(axis=0),
            "pred_momx_rel_std": valid["pred_momx_rel"].std(axis=0),
            "pred_momy_rel_mean": valid["pred_momy_rel"].mean(axis=0),
            "pred_momy_rel_std": valid["pred_momy_rel"].std(axis=0),
            "truth_cont_abs_mean": valid["truth_cont_abs"].mean(axis=0),
            "truth_cont_abs_std": valid["truth_cont_abs"].std(axis=0),
            "truth_momx_abs_mean": valid["truth_momx_abs"].mean(axis=0),
            "truth_momx_abs_std": valid["truth_momx_abs"].std(axis=0),
            "truth_momy_abs_mean": valid["truth_momy_abs"].mean(axis=0),
            "truth_momy_abs_std": valid["truth_momy_abs"].std(axis=0),
            "truth_cont_rel_mean": valid["truth_cont_rel"].mean(axis=0),
            "truth_cont_rel_std": valid["truth_cont_rel"].std(axis=0),
            "truth_momx_rel_mean": valid["truth_momx_rel"].mean(axis=0),
            "truth_momx_rel_std": valid["truth_momx_rel"].std(axis=0),
            "truth_momy_rel_mean": valid["truth_momy_rel"].mean(axis=0),
            "truth_momy_rel_std": valid["truth_momy_rel"].std(axis=0),
        }
    )
    df_per_step.to_csv(output_path, index=False)


def save_filtered_spectra_npzs(
    spectra_paths, keep_mask, output_dir, experiment_name, seed, output_suffix
):
    for path in spectra_paths:
        spectra_npz = np.load(path)
        filtered = {}
        for key in spectra_npz.files:
            arr = spectra_npz[key]
            if arr.ndim > 0 and arr.shape[0] == keep_mask.shape[0] and key != "k_bins":
                filtered[key] = arr[keep_mask]
            else:
                filtered[key] = arr
        step = extract_spectra_step(path)
        output_path = os.path.join(
            output_dir,
            append_file_suffix(f"{experiment_name}_seed{seed}_spectra_t{step}", output_suffix)
            + ".npz",
        )
        np.savez(output_path, **filtered)


def main():
    parser = argparse.ArgumentParser(
        description="Recover finite TTO artifacts from an existing evaluation run."
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing evaluation_metrics/ and saved_plots/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where recovered artifacts will be written",
    )
    parser.add_argument("--experiment_name", type=str, default="PINO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_suffix", type=str, default="_tto")
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Filename suffix for recovered artifacts; defaults to --input_suffix",
    )
    args = parser.parse_args()

    output_suffix = args.input_suffix if args.output_suffix is None else args.output_suffix

    per_sample_path = find_single_file(
        args.source_dir,
        f"**/evaluation_metrics/{args.experiment_name}_seed{args.seed}_per_sample_metrics{args.input_suffix}.npz",
    )
    per_sample_npz = np.load(per_sample_path)

    spectra_paths = sorted(
        glob.glob(
            os.path.join(
                args.source_dir,
                f"**/saved_plots/{args.experiment_name}_seed{args.seed}_spectra_t*{args.input_suffix}.npz",
            ),
            recursive=True,
        ),
        key=extract_spectra_step,
    )
    if not spectra_paths:
        raise FileNotFoundError("No TTO spectra files were found to recover.")

    bad_mask = build_bad_sample_mask(per_sample_npz, spectra_paths)
    keep_mask = ~bad_mask
    bad_indices_0 = np.where(bad_mask)[0]
    bad_indices_1 = (bad_indices_0 + 1).tolist()

    eval_dir = os.path.join(args.output_dir, "evaluation_metrics")
    plot_dir = os.path.join(args.output_dir, "saved_plots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    per_sample_out = os.path.join(
        eval_dir,
        append_file_suffix(
            f"{args.experiment_name}_seed{args.seed}_per_sample_metrics", output_suffix
        )
        + ".npz",
    )
    save_filtered_per_sample_npz(per_sample_npz, keep_mask, per_sample_out)

    per_step_out = os.path.join(
        eval_dir,
        append_file_suffix(
            f"{args.experiment_name}_seed{args.seed}_per_step_metrics", output_suffix
        )
        + ".csv",
    )
    save_filtered_per_step_csv(per_sample_npz, keep_mask, per_step_out)

    save_filtered_spectra_npzs(
        spectra_paths,
        keep_mask,
        plot_dir,
        args.experiment_name,
        args.seed,
        output_suffix,
    )

    vorticity_path = find_single_file(
        args.source_dir,
        f"**/saved_plots/{args.experiment_name}_seed{args.seed}_vorticity_sample0{args.input_suffix}.npz",
    )
    vorticity_out = os.path.join(
        plot_dir,
        append_file_suffix(
            f"{args.experiment_name}_seed{args.seed}_vorticity_sample0", output_suffix
        )
        + ".npz",
    )
    shutil.copy2(vorticity_path, vorticity_out)

    report = {
        "source_dir": args.source_dir,
        "output_dir": args.output_dir,
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "input_suffix": args.input_suffix,
        "output_suffix": output_suffix,
        "num_total_samples": int(keep_mask.shape[0]),
        "num_kept_samples": int(keep_mask.sum()),
        "num_dropped_samples": int(bad_mask.sum()),
        "dropped_samples_0_based": bad_indices_0.tolist(),
        "dropped_samples_1_based": bad_indices_1,
    }
    report_path = os.path.join(
        eval_dir,
        append_file_suffix(f"{args.experiment_name}_seed{args.seed}_recovery_report", output_suffix)
        + ".json",
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Kept {report['num_kept_samples']}/{report['num_total_samples']} samples.")
    print(f"Dropped samples (1-based): {bad_indices_1}")
    print(f"Saved recovered per-sample metrics to {per_sample_out}")
    print(f"Saved recovered per-step metrics to {per_step_out}")
    print(f"Saved recovered spectra files to {plot_dir}")
    print(f"Copied vorticity sample 0 to {vorticity_out}")
    print(f"Saved recovery report to {report_path}")


if __name__ == "__main__":
    main()
