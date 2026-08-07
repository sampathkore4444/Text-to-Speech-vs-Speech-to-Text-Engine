"""Pure-logic tests for the fine-tuning loop helpers.

These don't require the ``finetune`` extra - the torch-dependent checkpoint
round-trip tests live in ``test_finetune.py`` (skipped without torch).
"""

from __future__ import annotations

from speechai.finetune.train import _early_stop_update


def test_early_stop_first_eval_is_improvement() -> None:
    best, stall, stop, improved = _early_stop_update(
        None, 0.41, min_delta=0.0, patience=3, stall_count=0
    )
    assert (best, stall, stop, improved) == (0.41, 0, False, True)


def test_early_stop_improvement_resets_stall() -> None:
    best, stall, stop, improved = _early_stop_update(
        0.41, 0.30, min_delta=0.0, patience=3, stall_count=2
    )
    assert (best, stall, stop, improved) == (0.30, 0, False, True)


def test_early_stop_plateau_counts_stall() -> None:
    best, stall, stop, improved = _early_stop_update(
        0.30, 0.31, min_delta=0.0, patience=3, stall_count=1
    )
    assert (best, stall, stop, improved) == (0.30, 2, False, False)


def test_early_stop_patience_one_stops_on_first_stall() -> None:
    best, stall, stop, improved = _early_stop_update(
        0.30, 0.31, min_delta=0.0, patience=1, stall_count=0
    )
    assert (best, stall, stop, improved) == (0.30, 1, True, False)


def test_early_stop_triggers_after_patience() -> None:
    best, stall, stop, improved = _early_stop_update(
        0.30, 0.32, min_delta=0.0, patience=2, stall_count=1
    )
    assert (best, stall, stop, improved) == (0.30, 2, True, False)


def test_early_stop_min_delta_requires_real_improvement() -> None:
    # +0.003 is not enough to count as progress when min_delta=0.01
    best, stall, stop, improved = _early_stop_update(
        0.300, 0.297, min_delta=0.01, patience=2, stall_count=0
    )
    assert (best, stall, stop, improved) == (0.300, 1, False, False)
    # ...but a 0.02 gain resets the counter and updates the best.
    best, stall, stop, improved = _early_stop_update(
        0.300, 0.280, min_delta=0.01, patience=2, stall_count=1
    )
    assert (best, stall, stop, improved) == (0.280, 0, False, True)
