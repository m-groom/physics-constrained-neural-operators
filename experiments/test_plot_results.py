import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import plot_results


def test_build_parser_excludes_pino_tto_by_default():
    parser = plot_results.build_parser()
    args = parser.parse_args([])
    assert args.include_pino_tto is False


def test_build_parser_includes_pino_tto_when_flag_is_set():
    parser = plot_results.build_parser()
    args = parser.parse_args(["--include_pino_tto"])
    assert args.include_pino_tto is True


def test_vorticity_column_titles_omit_tto_by_default():
    assert plot_results.get_vorticity_column_titles() == ["FNO", "FNOC", "PINO", "Ground truth"]


def test_vorticity_column_titles_include_tto_when_requested():
    assert plot_results.get_vorticity_column_titles(include_pino_tto=True) == [
        "FNO",
        "FNOC",
        "PINO",
        "PINO+TTO",
        "Ground truth",
    ]


# The spectra glob is "*_spectra_t*{suffix}.npz", and with an empty suffix the
# wildcard after the step number also swallows a variant suffix: a long-horizon
# evaluation sharing a directory with the default one would otherwise supply the
# default figure's final spectra.
SPECTRA_NAMES = [
    "FNO_seed1_spectra_t1.npz",
    "FNO_seed1_spectra_t64.npz",
    "FNO_seed1_spectra_t448_T448.npz",
    "FNO_seed1_spectra_t64_test_time.npz",
]


def test_the_default_suffix_keeps_only_unsuffixed_spectra():
    assert plot_results.spectra_files_for_suffix(SPECTRA_NAMES, "") == [
        "FNO_seed1_spectra_t1.npz",
        "FNO_seed1_spectra_t64.npz",
    ]


def test_a_variant_suffix_keeps_only_its_own_spectra():
    assert plot_results.spectra_files_for_suffix(SPECTRA_NAMES, "_T448") == [
        "FNO_seed1_spectra_t448_T448.npz"
    ]
