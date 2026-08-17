# -*- coding: utf-8 -*-
"""
Module test_stochastic.py
=========================

End-to-end sanity checks for the stochastic-method orchestrator in `src.methods.stochastic`.

Tests use `_QUICK_CFG` (2 reps x 300 invocations, warmup 30) instead of production fidelity (10 reps x 10 000 invocations). This keeps each per-adaptation run under ~1 s with wider CIs; rho / W agreement tolerances are loosened to match. To re-run at full fidelity, swap `method_cfg=_QUICK_CFG` for `method_cfg=None` in the call.

    - **TestStochasticEndToEnd**: each adaptation solves end-to-end via SimPy DES, produces 13 stable nodes and a full R1 / R2 verdict with the expected schema; `_std` columns are present for every stochastic metric.
    - **TestAnalyticAgreement**: the DES baseline agrees with the closed-form analytic baseline on the network-wide averages (avg_rho, W_net) within Monte-Carlo tolerance, making the two methods mutually validating.
"""
# data types
from typing import Any, Dict

# testing framework
import pytest

# modules under test
from src.methods.analytic import run as run_analytic
from src.methods.stochastic import run as run_stochastic


# 2 reps x 300 invocations is enough for end-to-end pipeline coverage (reps > 1 populates `_std`; warmup > 0 exercises the interval-straddling branch); production fidelity lives in `stochastic.json`
_QUICK_CFG = {
    "method": "stochastic",
    "seed": 42,
    "horizon_invocations": 300,
    "warmup_invocations": 30,
    "replications": 2,
    "confidence_level": 0.95,
    "window_invocations": 30,
    "rsem_target": 0.05,
}


@pytest.fixture(scope="module")
def _result_baseline() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """*_result_baseline()* `run_stochastic(adp="baseline", wrt=False, method_cfg=_QUICK_CFG)` once per module."""
    return run_stochastic(adp="baseline", wrt=False, method_cfg=_QUICK_CFG)


@pytest.fixture(scope="module")
def _result_s1() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """*_result_s1()* `run_stochastic(adp="s1", wrt=False, method_cfg=_QUICK_CFG)` once per module."""
    return run_stochastic(adp="s1", wrt=False, method_cfg=_QUICK_CFG)


@pytest.fixture(scope="module")
def _analytic_baseline() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """*_analytic_baseline()* `run_analytic(adp="baseline", wrt=False)`, fixture-cached so cross-method tests share one solve."""
    return run_analytic(adp="baseline", wrt=False)


class TestStochasticEndToEnd:
    """**TestStochasticEndToEnd** the DES pipeline solves end-to-end across `baseline` / `s1`, produces 13 stable nodes, exposes the full R1 / R2 verdict, and attaches `_std` columns for every stochastic metric. Two adaptations are enough to cover both profiles (`dflt` / `opti`)."""

    @pytest.fixture(params=["baseline", "s1"])
    def _result(self,
                request: pytest.FixtureRequest,
                _result_baseline: Dict[str, Any],
                _result_s1: Dict[str, Any]) -> Dict[str, Any]:
        """*_result()* dispatch the right per-adaptation result by `request.param`."""
        if request.param == "baseline":
            return _result_baseline
        return _result_s1

    def test_runs_and_stable(self, _result: Dict[str, Any]) -> None:
        """*test_runs_and_stable()* 13 real nodes (plus the `FAIL_{1}` sink) and `rho < 1.0` on every real node."""
        _nds = _result["nodes"]
        _real = _nds[_nds["key"] != "FAIL_{1}"]
        assert len(_real) == 13
        assert "FAIL_{1}" in _nds["key"].values
        _max_rho = _real["rho"].max()
        assert _max_rho < 1.0, f"max rho={_max_rho:.4f}"

    def test_reqs_shape(self, _result: Dict[str, Any]) -> None:
        """*test_reqs_shape()* `set(_req.keys()) == {"R1", "R2"}` and each entry carries `pass` / `value` / `metric` fields."""
        _req = _result["requirements"]
        assert set(_req.keys()) == {"R1", "R2"}
        for _k in ("R1", "R2"):
            assert "pass" in _req[_k]
            assert "value" in _req[_k]
            assert "metric" in _req[_k]

    def test_r2_verdict_matches_W_e2e_threshold(self, _result: Dict[str, Any]) -> None:
        """*test_r2_verdict_matches_W_e2e_threshold()* the R2 verdict equals `W_e2e <= threshold`; the verdict pipeline is internally consistent. Pass/fail per scenario varies per the dissertation's findings; only the consistency is asserted here."""
        _r2 = _result["requirements"]["R2"]
        _w_e2e = _result["network"].iloc[0]["W_e2e"]
        assert _r2["value"] == pytest.approx(_w_e2e, rel=1e-6)
        assert _r2["pass"] == (_w_e2e <= _r2["threshold"])

    def test_std_cols_present(self, _result: Dict[str, Any]) -> None:
        """*test_std_cols_present()* `f"{m}_std" in _nds.columns` for every stochastic metric (CI plotters depend on these)."""
        _nds = _result["nodes"]
        for _m in ("rho", "L", "Lq", "W", "Wq"):
            assert f"{_m}_std" in _nds.columns


class TestAnalyticAgreement:
    """**TestAnalyticAgreement** the DES baseline matches the closed-form analytic baseline on network-wide averages within Monte-Carlo tolerance. Bands are wide because `_QUICK_CFG` only runs 2 reps x 300 invocations."""

    def test_avg_rho_close_to_analytic(self,
                                       _result_baseline: Dict[str, Any],
                                       _analytic_baseline: Dict[str, Any]) -> None:
        """*test_avg_rho_close_to_analytic()* stochastic `avg_rho` matches analytic within 25 % at the `_QUICK_CFG` horizon (300 x 2)."""
        _st_rho = float(_result_baseline["network"].iloc[0]["avg_rho"])
        _an_rho = float(_analytic_baseline["network"].iloc[0]["avg_rho"])
        assert _st_rho == pytest.approx(_an_rho, rel=0.25), (
            f"stochastic avg_rho={_st_rho:.4f} vs "
            f"analytic avg_rho={_an_rho:.4f}"
        )

    def test_W_net_close_to_analytic(self,
                                     _result_baseline: Dict[str, Any],
                                     _analytic_baseline: Dict[str, Any]) -> None:
        """*test_W_net_close_to_analytic()* stochastic `W_net` matches analytic within 30 % at the `_QUICK_CFG` horizon (300 x 2)."""
        _st_w = float(_result_baseline["network"].iloc[0]["W_net"])
        _an_w = float(_analytic_baseline["network"].iloc[0]["W_net"])
        assert _st_w == pytest.approx(_an_w, rel=0.30), (
            f"stochastic W_net={_st_w:.6f} vs analytic W_net={_an_w:.6f}"
        )
