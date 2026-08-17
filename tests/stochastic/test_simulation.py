# -*- coding: utf-8 -*-
"""
Module test_simulation.py
=========================

Sanity checks for the SimPy DES engine in `src.stochastic.simulation`.

    - **TestMM1Convergence**: a single M/M/1 node run converges to the closed-form rho / L / W / Wq within Monte-Carlo noise; `_std` companion columns are populated for the CI machinery downstream.
    - **TestSeededReproducibility**: passing the same `seed` twice produces a bit-identical summary frame.
    - **TestBlockingBoundary**: an M/M/1/K node under saturation records a non-zero `Blocking_Prob`; under light load it is ~ 0.
    - **TestModelString**: the `M/M/c[/K]` label string helper.

# TODO: add a 2-node feed-forward regression once the orchestrator's end-to-end test (`tests/methods/test_stochastic.py`) is wired up; the engine-level test should stay focused on single-node invariants to keep the runtime tight.
"""
# data types
from typing import Optional

# scientific stack
import numpy as np
import pandas as pd

# testing framework
import pytest

# module under test
from src.stochastic.simulation import (
    format_model_string,
    simulate_net,
)


def _run_single_node(*,
                     lam_z: float,
                     K: Optional[int],
                     horizon: float,
                     warmup: float,
                     reps: int,
                     seed: int = 42) -> pd.DataFrame:
    """*_run_single_node()* drive `simulate_net` on a single mu=10, c=1 node (no onward routing) at the given lambda / capacity / run length."""
    _sim = simulate_net(
        mu=[10.0],
        lam_z=[lam_z],
        c=[1],
        K=[K],
        P=np.array([[0.0]]),
        horizon=horizon,
        warmup=warmup,
        reps=reps,
        seed=seed,
    )
    return _sim


class TestMM1Convergence:
    """**TestMM1Convergence** a single M/M/1 node at rho = 0.5 converges to the textbook formulas (rho = 0.5, L = 1.0, W = 0.2, Wq = 0.1) within Monte-Carlo noise; `_std` companion columns are non-NaN and non-negative when reps > 1."""

    def test_rho_L_W_Wq_match_textbook(self) -> None:
        """*test_rho_L_W_Wq_match_textbook()* lambda=5, mu=10, 3 x 1500 s gives ~21 k samples (CLT-safe); rho / L / W match within 12 %, Wq within 20 % (the usual wider band)."""
        _summary = _run_single_node(lam_z=5.0,
                                    K=None,
                                    horizon=1500.0,
                                    warmup=150.0,
                                    reps=3)
        _row = _summary.iloc[0]
        assert _row["rho_mean"] == pytest.approx(0.5, rel=0.12)
        assert _row["L_mean"] == pytest.approx(1.0, rel=0.12)
        assert _row["W_mean"] == pytest.approx(0.2, rel=0.12)
        assert _row["Wq_mean"] == pytest.approx(0.1, rel=0.20)

    def test_std_cols_populated(self) -> None:
        """*test_std_cols_populated()* `_std` columns for rho / L / W / Wq are non-NaN and `>= 0` when `reps > 1`."""
        _summary = _run_single_node(lam_z=5.0,
                                    K=None,
                                    horizon=500.0,
                                    warmup=50.0,
                                    reps=2)
        _row = _summary.iloc[0]
        for _m in ("rho", "L", "W", "Wq"):
            _s = _row[f"{_m}_std"]
            assert not np.isnan(_s), f"{_m}_std should not be NaN"
            assert _s >= 0.0, f"{_m}_std must be non-negative"


class TestSeededReproducibility:
    """**TestSeededReproducibility** seeding `random` and `numpy.random` with the same value produces a bit-identical summary frame across calls."""

    def test_same_seed_same_summary(self) -> None:
        """*test_same_seed_same_summary()* two seeded `_run_single_node` calls return frames that compare equal cell-for-cell on every numeric column."""
        _first = _run_single_node(lam_z=5.0, K=None, horizon=1000.0, warmup=100.0, reps=2)
        _second = _run_single_node(lam_z=5.0, K=None, horizon=1000.0, warmup=100.0, reps=2)
        for _col in _first.columns:
            if _col == "node":
                continue
            assert (_first[_col].to_numpy() == pytest.approx(_second[_col].to_numpy()))


class TestBlockingBoundary:
    """**TestBlockingBoundary** the finite-capacity M/M/1/K boundary records non-zero `Blocking_Prob` under saturation and ~0 under light load."""

    def test_saturated_system_blocks(self) -> None:
        """*test_saturated_system_blocks()* `lam_z=15, mu=10, K=5` (rho ~ 1.5) yields `Blocking_Prob_mean > 0`."""
        _summary = _run_single_node(lam_z=15.0,
                                    K=5,
                                    horizon=500.0,
                                    warmup=50.0,
                                    reps=3)
        assert _summary.iloc[0]["Blocking_Prob_mean"] > 0.0

    def test_unloaded_system_no_blocks(self) -> None:
        """*test_unloaded_system_no_blocks()* `lam_z=0.5, mu=10, K=5` (rho ~ 0.05) yields `Blocking_Prob_mean == 0`."""
        _summary = _run_single_node(lam_z=0.5,
                                    K=5,
                                    horizon=500.0,
                                    warmup=50.0,
                                    reps=3)
        assert _summary.iloc[0]["Blocking_Prob_mean"] == pytest.approx(0.0)


class TestModelString:
    """**TestModelString** `format_model_string(c, K)` produces `M/M/c[/K]` labels matching the analytic method's notation."""

    def test_mm1_unbounded(self) -> None:
        """*test_mm1_unbounded()* `format_model_string(1, None) == "M/M/1"`."""
        assert format_model_string(1, None) == "M/M/1"

    def test_mm1_with_K(self) -> None:
        """*test_mm1_with_K()* `format_model_string(1, 10) == "M/M/1/10"`."""
        assert format_model_string(1, 10) == "M/M/1/10"

    def test_mmc_unbounded(self) -> None:
        """*test_mmc_unbounded()* `format_model_string(3, None) == "M/M/3"`."""
        assert format_model_string(3, None) == "M/M/3"

    def test_mmc_with_K(self) -> None:
        """*test_mmc_with_K()* `format_model_string(2, 20) == "M/M/2/20"`."""
        assert format_model_string(2, 20) == "M/M/2/20"
