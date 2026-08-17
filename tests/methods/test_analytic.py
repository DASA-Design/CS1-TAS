# -*- coding: utf-8 -*-
"""
Module test_analytic.py
=======================

End-to-end sanity checks for the analytic-method orchestrator in `src.methods.analytic`, exercised across all four adaptations (`baseline`, `s1`, `s2`, `aggregate`) under the loss-network FAIL-sink model.

    - **TestAnalyticEndToEnd** every adaptation solves to 13 stable real nodes plus the absorbing `FAIL_{1}` sink, exposes a full R1 / R2 verdict, and satisfies the flow identity `eps_e2e = lambda_FAIL/lambda_z` with conservation `chi_out + lambda_FAIL = lambda_z`.
    - **TestTradeoff** the performance-vs-availability trade-off has the expected shape: baseline fails availability, retry (s1) buys availability at the cost of performance, select-reliable (s2) buys performance but not availability, and only the aggregate satisfies both.
"""
# testing framework
import pytest

# module under test
from src.methods.analytic import run

_FAIL = "FAIL_{1}"

# expected R1 (availability <= 1%) / R2 (performance <= 26 ms) verdict per adaptation
_EXPECTED = {
    "baseline":  {"R1": False, "R2": True},
    "s1":        {"R1": True,  "R2": False},
    "s2":        {"R1": False, "R2": True},
    "aggregate": {"R1": True,  "R2": True},
}


@pytest.mark.parametrize("adp", ["baseline", "s1", "s2", "aggregate"])
class TestAnalyticEndToEnd:
    """**TestAnalyticEndToEnd** every adaptation solves end-to-end: 13 stable real nodes plus the FAIL sink, a full R1 / R2 verdict, and the flow-based availability identity."""

    def test_runs_and_stable(self, adp: str) -> None:
        """*test_runs_and_stable()* 13 real nodes (plus the `FAIL_{1}` sink) and `rho < 1.0` on every real node."""
        _nds = run(adp=adp, wrt=False)["nodes"]
        _real = _nds[_nds["key"] != _FAIL]
        assert len(_real) == 13
        assert _FAIL in _nds["key"].values
        _max_rho = _real["rho"].max()
        assert _max_rho < 1.0, f"{adp}: max rho={_max_rho:.4f}"

    def test_requirements_shape(self, adp: str) -> None:
        """*test_requirements_shape()* `set(req) == {"R1", "R2"}` and every verdict carries `pass`, `value`, `metric`, `contributions`."""
        _req = run(adp=adp, wrt=False)["requirements"]
        assert set(_req.keys()) == {"R1", "R2"}
        for _k in ("R1", "R2"):
            assert {"pass", "value", "metric", "contributions"} <= set(_req[_k])

    def test_e2e_columns_populated(self, adp: str) -> None:
        """*test_e2e_columns_populated()* the aggregate frame carries finite `eps_e2e`, positive `chi_out`, and positive `W_e2e`."""
        _net = run(adp=adp, wrt=False)["network"].iloc[0]
        assert _net["eps_e2e"] == _net["eps_e2e"]  # NaN-check via reflexivity
        assert _net["chi_out"] > 0
        assert _net["W_e2e"] > 0

    def test_availability_is_fail_flow(self, adp: str) -> None:
        """*test_availability_is_fail_flow()* `eps_e2e == lambda_FAIL/lambda_z` and `chi_out == lambda_z - lambda_FAIL` (loss-network conservation)."""
        _result = run(adp=adp, wrt=False)
        _nds = _result["nodes"]
        _net = _result["network"].iloc[0]
        _lam_z = float(_result["config"].build_lam_z_vec()[0])
        _lam_fail = float(_nds[_nds["key"] == _FAIL]["lambda"].sum())
        assert _net["eps_e2e"] == pytest.approx(_lam_fail / _lam_z)
        assert _net["chi_out"] == pytest.approx(_lam_z - _lam_fail)

    def test_r2_verdict_matches_W_e2e(self, adp: str) -> None:
        """*test_r2_verdict_matches_W_e2e()* the R2 verdict equals `W_e2e <= threshold`; the verdict pipeline is internally consistent."""
        _result = run(adp=adp, wrt=False)
        _r2 = _result["requirements"]["R2"]
        _w_e2e = _result["network"].iloc[0]["W_e2e"]
        assert _r2["value"] == pytest.approx(_w_e2e)
        assert _r2["pass"] == (_w_e2e <= _r2["threshold"])

    def test_verdict_matches_expected(self, adp: str) -> None:
        """*test_verdict_matches_expected()* the R1 / R2 PASS / FAIL pattern matches the trade-off table for this adaptation."""
        _req = run(adp=adp, wrt=False)["requirements"]
        assert _req["R1"]["pass"] is _EXPECTED[adp]["R1"], f"{adp} R1"
        assert _req["R2"]["pass"] is _EXPECTED[adp]["R2"], f"{adp} R2"


class TestTradeoff:
    """**TestTradeoff** the performance-vs-availability trade-off across adaptations has the expected shape: retry buys availability at a performance cost, selection buys performance, only the aggregate clears both requirements."""

    def _eps_w(self, adp: str) -> tuple:
        """*_eps_w()* return `(eps_e2e, W_e2e)` for an adaptation."""
        _net = run(adp=adp, wrt=False)["network"].iloc[0]
        return float(_net["eps_e2e"]), float(_net["W_e2e"])

    def test_only_aggregate_passes_both(self) -> None:
        """*test_only_aggregate_passes_both()* the aggregate is the only adaptation whose R1 and R2 both pass."""
        _both = {}
        for _adp in ("baseline", "s1", "s2", "aggregate"):
            _req = run(adp=_adp, wrt=False)["requirements"]
            _both[_adp] = _req["R1"]["pass"] and _req["R2"]["pass"]
        assert _both == {"baseline": False, "s1": False, "s2": False, "aggregate": True}

    def test_adaptations_cut_availability_failure(self) -> None:
        """*test_adaptations_cut_availability_failure()* every adaptation lowers `eps_e2e` below baseline; the aggregate is the lowest."""
        _b, _ = self._eps_w("baseline")
        _s1, _ = self._eps_w("s1")
        _s2, _ = self._eps_w("s2")
        _ag, _ = self._eps_w("aggregate")
        assert _s1 < _b and _s2 < _b
        assert _ag < _s1 and _ag < _s2

    def test_retry_costs_performance_selection_saves_it(self) -> None:
        """*test_retry_costs_performance_selection_saves_it()* retry (s1) raises `W_e2e` above baseline (re-dispatch overload); selection (s2) lowers it (faster reliable services)."""
        _, _wb = self._eps_w("baseline")
        _, _w1 = self._eps_w("s1")
        _, _w2 = self._eps_w("s2")
        assert _w1 > _wb, "retry should raise W_e2e (overload)"
        assert _w2 < _wb, "selection should lower W_e2e (faster services)"
