# -*- coding: utf-8 -*-
"""
Module test_metrics.py
======================

Sanity checks for the network-wide aggregator and the R1 / R2 verdict logic in `src.analytic.metrics`.

    - **TestAggregateNetwork** aggregation math on small, hand-computable per-node frames: per-visit weighted means, the flow-based end-to-end availability via the FAIL sink, the FAIL-sink exclusion from the real-network aggregates, and the legacy (exposure) path kept for not-yet-propagated callers.
    - **TestCheckRequirements** R1 / R2 verdicts under the loss-network flow model; the `lam_z` + `routing` requirement, override kwargs, and the contributions breakdown.
    - **TestThresholdsFromReference** verdict's threshold / operator / units come from `data/reference/baseline.json` (single source of truth).
"""
# data types
from typing import Any, Dict, List

# scientific stack
import numpy as np
import pandas as pd

# testing framework
import pytest

# modules under test
from src.analytic.metrics import (aggregate_net,
                                  check_reqs)
from src.io import load_reference

_FAIL = "FAIL_{1}"


def _make_nodes(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """*_make_nodes()* per-node DataFrame from a list of dicts; sensible defaults fill any column the caller omits.

    Args:
        rows (List[Dict[str, Any]]): per-node column overrides (include `key` to mark the FAIL sink).

    Returns:
        pd.DataFrame: per-node metrics frame ready for aggregation.
    """
    _defaults = {
        "key": "N",
        "lambda": 0.0,
        "mu": 1.0,
        "c": 1,
        "K": None,
        "rho": 0.0,
        "L": 0.0,
        "Lq": 0.0,
        "W": 0.0,
        "Wq": 0.0,
    }
    _filled = [{**_defaults, **_r} for _r in rows]
    return pd.DataFrame(_filled)


def _fail_routing(n: int, from_idx: int, leak: float) -> np.ndarray:
    """*_fail_routing()* an `n`-node routing whose row `from_idx` sends `leak` to the FAIL sink (last index); only used to trigger the flow path and feed the R1 contributions."""
    _R = np.zeros((n, n))
    _R[from_idx, n - 1] = leak
    return _R


class TestAggregateNetwork:
    """**TestAggregateNetwork** verifies that the per-node to network-wide reduction matches hand-computed expectations on small frames."""

    def test_totals_and_max(self) -> None:
        """*test_totals_and_max()* `nodes == 2`, `total_throughput == 40.0`, `avg_rho == 0.5`, `max_rho == 0.8`, `avg_mu == 50.0`, `L_net == 3.0`, `Lq_net == 1.4`."""
        _nodes = _make_nodes([
            {"lambda": 10.0, "mu": 50.0, "rho": 0.2, "L": 1.0, "Lq": 0.4},
            {"lambda": 30.0, "mu": 50.0, "rho": 0.8, "L": 2.0, "Lq": 1.0},
        ])
        _agg = aggregate_net(_nodes).iloc[0]
        assert _agg["nodes"] == 2
        assert _agg["total_throughput"] == pytest.approx(40.0)
        assert _agg["avg_rho"] == pytest.approx(0.5)
        assert _agg["max_rho"] == pytest.approx(0.8)
        assert _agg["avg_mu"] == pytest.approx(50.0)
        assert _agg["L_net"] == pytest.approx(3.0)
        assert _agg["Lq_net"] == pytest.approx(1.4)

    def test_weighted_w_net(self) -> None:
        """*test_weighted_w_net()* per-visit weighted-mean: `W_net == (10*0.1 + 30*0.5) / 40 == 0.4`; `Wq_net == 0.2`."""
        _nodes = _make_nodes([
            {"lambda": 10.0, "W": 0.1, "Wq": 0.05},
            {"lambda": 30.0, "W": 0.5, "Wq": 0.25},
        ])
        _agg = aggregate_net(_nodes).iloc[0]
        assert _agg["W_net"] == pytest.approx(0.40)
        assert _agg["Wq_net"] == pytest.approx(0.20)

    def test_zero_lambda_branch(self) -> None:
        """*test_zero_lambda_branch()* every-zero-lambda short-circuits the weighted-mean formula; no `0 / 0` raised."""
        _nodes = _make_nodes([
            {"lambda": 0.0, "W": 1.0, "Wq": 1.0},
            {"lambda": 0.0, "W": 2.0, "Wq": 2.0},
        ])
        _agg = aggregate_net(_nodes).iloc[0]
        assert _agg["total_throughput"] == pytest.approx(0.0)
        assert _agg["W_net"] == pytest.approx(0.0)
        assert _agg["Wq_net"] == pytest.approx(0.0)

    def test_e2e_columns_nan_without_inputs(self) -> None:
        """*test_e2e_columns_nan_without_inputs()* without `routing` / `eps_vec`, the end-to-end columns are NaN; per-visit columns still populated."""
        _nodes = _make_nodes([{"lambda": 10.0, "W": 0.1, "L": 1.0}])
        _agg = aggregate_net(_nodes).iloc[0]
        assert np.isnan(_agg["eps_e2e"])
        assert np.isnan(_agg["chi_out"])
        assert np.isnan(_agg["W_e2e"])
        assert _agg["W_net"] == pytest.approx(0.1)

    def test_flow_availability_from_fail_sink(self) -> None:
        """*test_flow_availability_from_fail_sink()* with `routing`, `eps_e2e == lambda_FAIL/lam_z`, `chi_out == lam_z - lambda_FAIL`, `W_e2e == L_net/chi_out`; the FAIL sink is excluded from the real-network aggregates."""
        _nodes = _make_nodes([
            {"key": "SVC", "lambda": 100.0, "mu": 100.0, "rho": 0.5, "L": 2.0, "W": 0.02},
            {"key": _FAIL, "lambda": 15.0, "mu": 1e9, "rho": 0.0, "L": 0.0, "W": 0.0},
        ])
        _routing = _fail_routing(2, from_idx=0, leak=0.15)
        _agg = aggregate_net(_nodes, lam_z=100.0, routing=_routing).iloc[0]
        assert _agg["eps_e2e"] == pytest.approx(0.15)
        assert _agg["chi_out"] == pytest.approx(85.0)
        assert _agg["W_e2e"] == pytest.approx(2.0 / 85.0)
        # FAIL sink excluded: only the real service is counted, its mu not the 1e9 sink
        assert _agg["nodes"] == 1
        assert _agg["avg_mu"] == pytest.approx(100.0)
        assert _agg["L_net"] == pytest.approx(2.0)

    def test_legacy_exposure_path(self) -> None:
        """*test_legacy_exposure_path()* with `eps_vec` (no routing), `eps_e2e == 1 - prod((1-eps_j)^V_j)` (kept for not-yet-propagated callers)."""
        _nodes = _make_nodes([
            {"lambda": 50.0, "L": 1.0, "W": 0.02},
            {"lambda": 50.0, "L": 1.0, "W": 0.02},
        ])
        _agg = aggregate_net(_nodes, lam_z=100.0, eps_vec=np.array([0.10, 0.20])).iloc[0]
        _expected = 1.0 - (0.9 ** 0.5) * (0.8 ** 0.5)
        assert _agg["eps_e2e"] == pytest.approx(_expected)


class TestCheckRequirements:
    """**TestCheckRequirements** verifies the R1 / R2 verdicts under the locked thresholds (R1 <= 1% availability; R2 <= 26 ms performance) using the loss-network flow model."""

    def test_missing_inputs_raise(self) -> None:
        """*test_missing_inputs_raise()* calling `check_reqs` without `routing`/`eps_vec` (and no overrides) raises ValueError."""
        _nodes = _make_nodes([{"lambda": 10.0, "W": 0.001}])
        with pytest.raises(ValueError, match="end-to-end"):
            check_reqs(_nodes)

    def test_r1_fail_from_fail_sink(self) -> None:
        """*test_r1_fail_from_fail_sink()* a 10% leak to FAIL drives `eps_e2e` to 0.10 (>1%); R1 fails, R2 still passes."""
        _nodes = _make_nodes([
            {"key": "SVC", "lambda": 100.0, "L": 0.1, "W": 0.001},
            {"key": _FAIL, "lambda": 10.0, "mu": 1e9, "L": 0.0},
        ])
        _routing = _fail_routing(2, from_idx=0, leak=0.10)
        _req = check_reqs(_nodes, lam_z=100.0, routing=_routing)
        assert _req["R1"]["pass"] is False
        assert _req["R1"]["value"] == pytest.approx(0.10)
        assert _req["R2"]["pass"] is True

    def test_r2_fail_from_high_L(self) -> None:
        """*test_r2_fail_from_high_L()* L_net well above the threshold per chi_out -> W_e2e > 26 ms -> R2 fail."""
        _nodes = _make_nodes([
            {"key": "SVC", "lambda": 100.0, "L": 5.0, "W": 0.05},
            {"key": _FAIL, "lambda": 0.0, "mu": 1e9, "L": 0.0},
        ])
        _routing = _fail_routing(2, from_idx=0, leak=0.0)
        _req = check_reqs(_nodes, lam_z=100.0, routing=_routing)
        assert _req["R2"]["pass"] is False
        assert _req["R2"]["value"] == pytest.approx(5.0 / 100.0)

    def test_override_kwargs_win(self) -> None:
        """*test_override_kwargs_win()* explicit `failure_rate` and `response_time` overrides win over the frame-derived defaults."""
        _nodes = _make_nodes([{"lambda": 10.0, "W": 0.001, "L": 0.5}])
        _req = check_reqs(_nodes, failure_rate=0.05, response_time=0.100)
        assert _req["R1"]["value"] == pytest.approx(0.05)
        assert _req["R1"]["pass"] is False
        assert _req["R2"]["value"] == pytest.approx(0.100)
        assert _req["R2"]["pass"] is False

    def test_contributions_present(self) -> None:
        """*test_contributions_present()* R1 entries (flow path) expose `node`, `epsilon`, `V`, `contribution` (leak-to-FAIL share); R2 entries expose `node`, `L`, `W`, `V`, `share`."""
        _nodes = _make_nodes([
            {"key": "A", "lambda": 100.0, "L": 0.5, "W": 0.005},
            {"key": "B", "lambda": 100.0, "L": 1.5, "W": 0.015},
            {"key": _FAIL, "lambda": 15.0, "mu": 1e9, "L": 0.0},
        ])
        _R = np.zeros((3, 3))
        _R[0, 2] = 0.05   # A -> FAIL
        _R[1, 2] = 0.10   # B -> FAIL (larger contributor)
        _req = check_reqs(_nodes, lam_z=100.0, routing=_R)
        _r1_top = _req["R1"]["contributions"][0]
        assert set(_r1_top.keys()) == {"node", "epsilon", "V", "contribution"}
        assert _r1_top["node"] == "B"
        assert _r1_top["contribution"] == pytest.approx(0.10)   # 100*0.10/100
        _r2_top = _req["R2"]["contributions"][0]
        assert set(_r2_top.keys()) == {"node", "L", "W", "V", "share"}
        assert _r2_top["node"] == "B"
        assert _r2_top["share"] == pytest.approx(1.5 / 2.0)

    def test_verdict_schema(self) -> None:
        """*test_verdict_schema()* every verdict's keys are exactly `{metric, value, threshold, operator, units, pass, contributions, notes}`."""
        _nodes = _make_nodes([
            {"key": "SVC", "lambda": 100.0, "L": 0.1, "W": 0.001},
            {"key": _FAIL, "lambda": 0.0, "mu": 1e9, "L": 0.0},
        ])
        _req = check_reqs(_nodes, lam_z=100.0, routing=_fail_routing(2, 0, 0.0))
        _expected_keys = {"metric", "value", "threshold", "operator",
                          "units", "pass", "contributions", "notes"}
        for _k in ("R1", "R2"):
            assert set(_req[_k].keys()) == _expected_keys


class TestThresholdsFromReference:
    """**TestThresholdsFromReference** the R1 / R2 thresholds and metadata in every verdict come from `data/reference/baseline.json` (single source of truth)."""

    def _verdict(self) -> Dict[str, Dict[str, Any]]:
        """*_verdict()* a verdict computed via the flow path on a trivial FAIL-sink frame."""
        _nodes = _make_nodes([
            {"key": "SVC", "lambda": 100.0, "L": 0.1, "W": 0.001},
            {"key": _FAIL, "lambda": 0.0, "mu": 1e9, "L": 0.0},
        ])
        return check_reqs(_nodes, lam_z=100.0, routing=_fail_routing(2, 0, 0.0))

    def test_r1_threshold_matches_reference_json(self) -> None:
        """*test_r1_threshold_matches_reference_json()* R1 threshold / operator / units / notes match the reference JSON verbatim."""
        _ref = load_reference("baseline")["requirements"]
        _req = self._verdict()
        assert _req["R1"]["threshold"] == pytest.approx(_ref["R1"]["threshold"])
        assert _req["R1"]["operator"] == _ref["R1"]["operator"]
        assert _req["R1"]["units"] == _ref["R1"]["units"]
        assert _req["R1"]["notes"] == _ref["R1"]["notes"]

    def test_r2_threshold_matches_reference_json(self) -> None:
        """*test_r2_threshold_matches_reference_json()* R2 threshold / operator / units flow through."""
        _ref = load_reference("baseline")["requirements"]
        _req = self._verdict()
        assert _req["R2"]["threshold"] == pytest.approx(_ref["R2"]["threshold"])
        assert _req["R2"]["operator"] == _ref["R2"]["operator"]
        assert _req["R2"]["units"] == _ref["R2"]["units"]

    def test_reference_has_only_r1_r2(self) -> None:
        """*test_reference_has_only_r1_r2()* the reference JSON exposes exactly `{R1, R2}` (R3 retired); `check_reqs` returns that same set."""
        _ref = load_reference("baseline")["requirements"]
        assert set(_ref.keys()) == {"R1", "R2"}
        assert set(self._verdict().keys()) == {"R1", "R2"}
