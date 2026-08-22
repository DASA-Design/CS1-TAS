# -*- coding: utf-8 -*-
"""
Module test_reshape.py
======================

Shape + semantic checks for the orchestrator-output reshapers:

    - **TestNodeShape**: `coefs_to_nodes` produces one row per artifact with the expected column set.
    - **TestNetworkShape**: `coefs_to_net` produces a single-row frame with the aggregate coefficients.
    - **TestDeltaSemantics**: `compute_coefs_delta` gives zero delta on identical inputs and handles mismatched node sets (e.g. 13-node baseline vs 16-node aggregate).
    - **TestArchitectureAggregation**: `aggregate_arch_coefs` sum-first-divide-after rule produces single-row architecture-level coefficients (theta, sigma, eta, phi, epsilon).
    - **TestAggregateSweepToArch**: `aggregate_sweep_to_arch` collapses per-artifact sweep arrays into flat architecture-level arrays; aligned across artifacts via `sweep_arch`.
"""
# data types
from typing import Any, Dict

# scientific stack
import numpy as np

# testing framework
import pytest

# modules under test
from src.dimensional import (aggregate_arch_coefs,
                             aggregate_sweep_to_arch,
                             coefs_to_net,
                             coefs_to_nodes,
                             compute_coefs_delta,
                             compute_net_delta)
from src.methods.dimensional import run

# synthetic absorbing failure-sink node; excluded from architecture-level aggregation
_FAIL = "FAIL_{1}"


@pytest.fixture(scope="module")
def _dim_baseline() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """*_dim_baseline()* cached dimensional run for baseline (consumed by every test method via pytest's name-based fixture injection)."""
    return run(adp="baseline", wrt=False)


@pytest.fixture(scope="module")
def _dim_aggregate() -> Dict[str, Any]:  # pyright: ignore[reportUnusedFunction]
    """*_dim_aggregate()* cached dimensional run for aggregate (16 nodes); consumed by every test that compares against baseline."""
    return run(adp="aggregate", wrt=False)


class TestNodeShape:
    """**TestNodeShape** `coefs_to_nodes` produces one row per artifact with the expected column set."""

    def test_thirteen_rows_on_baseline(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_thirteen_rows_on_baseline()* baseline has 13 artifacts; the per-node frame has 13 rows."""
        _nds = coefs_to_nodes(_dim_baseline)
        assert len(_nds) == 13

    def test_thirteen_rows_on_aggregate(self, _dim_aggregate: Dict[str, Any]) -> None:
        """*test_thirteen_rows_on_aggregate()* aggregate has the same artifact count as baseline; the 3 swap slots replace 3 originals."""
        _nds = coefs_to_nodes(_dim_aggregate)
        assert len(_nds) == 13

    def test_expected_columns_present(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_expected_columns_present()* frame carries `key`, `name`, `type`, and one column per derived coefficient (`theta`, `sigma`, `eta`, `phi`)."""
        _nds = coefs_to_nodes(_dim_baseline)
        assert {"key", "name", "type", "theta", "sigma", "eta", "phi"} <= set(_nds.columns)

    def test_theta_varies_per_art(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_theta_varies_per_art()* `theta.max() - theta.min() > 0.05` after analytic seeding (Jackson-solved L varies per artifact, so theta is not uniform)."""
        _nds = coefs_to_nodes(_dim_baseline)
        _range = _nds["theta"].max() - _nds["theta"].min()
        _msg = f"theta range {_range} too small; seed may have failed"
        assert _range > 0.05, _msg


class TestNetworkShape:
    """**TestNetworkShape** `coefs_to_net` produces a single-row frame with one column per aggregated coefficient plus a `nodes` count."""

    def test_single_row(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_single_row()* the network aggregate collapses to exactly one row."""
        _net = coefs_to_net(_dim_baseline)
        assert len(_net) == 1

    def test_nodes_count_matches(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_nodes_count_matches()* the `nodes` column carries the artifact count."""
        _net = coefs_to_net(_dim_baseline)
        assert _net["nodes"].iloc[0] == 13

    def test_rejects_unknown_aggregator(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_rejects_unknown_aggregator()* a bogus `agg` name raises `ValueError` before any work is done."""
        with pytest.raises(ValueError, match="unknown aggregator"):
            coefs_to_net(_dim_baseline, agg="bogus")

    def test_median_differs_from_mean(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_median_differs_from_mean()* eta has a skewed distribution on baseline; median and mean diverge."""
        _mean = coefs_to_net(_dim_baseline, agg="mean")
        _med = coefs_to_net(_dim_baseline, agg="median")
        assert _med["eta"].iloc[0] != pytest.approx(_mean["eta"].iloc[0])


class TestDeltaSemantics:
    """**TestDeltaSemantics** `compute_coefs_delta` gives zero delta on identical inputs and handles mismatched node sets (13-node baseline vs 16-node aggregate)."""

    def test_zero_delta_on_identical_inputs(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_zero_delta_on_identical_inputs()* `compute_coefs_delta(x, x)` returns all-zero columns for every derived coefficient."""
        _nds = coefs_to_nodes(_dim_baseline)
        _d = compute_coefs_delta(_nds, _nds)
        for _m in ("theta", "sigma", "eta", "phi"):
            assert all(abs(_d[_m]) < 1e-12)

    def test_aggregate_delta_uses_intersection(self, _dim_baseline: Dict[str, Any], _dim_aggregate: Dict[str, Any]) -> None:
        """*test_aggregate_delta_uses_intersection()* baseline and aggregate each have 13 artifacts; 2 are swapped (MAS_{3}/AS_{3} -> MAS_{4}/AS_{4}; DS_{1} is constant). The delta frame keeps the 11 non-swap nodes."""
        _b = coefs_to_nodes(_dim_baseline)
        _a = coefs_to_nodes(_dim_aggregate)
        _d = compute_coefs_delta(_b, _a)
        assert len(_d) == 11

    def test_network_delta_single_row(self, _dim_baseline: Dict[str, Any], _dim_aggregate: Dict[str, Any]) -> None:
        """*test_network_delta_single_row()* `compute_net_delta` collapses two single-row network frames into one row with the shared coefficient columns."""
        _nb = coefs_to_net(_dim_baseline)
        _na = coefs_to_net(_dim_aggregate)
        _d = compute_net_delta(_nb, _na)
        assert len(_d) == 1
        assert {"theta", "sigma", "eta", "phi"} <= set(_d.columns)


class TestArchitectureAggregation:
    """**TestArchitectureAggregation** sum-first variable-level aggregation (sum raw vars first, divide after) produces one architecture-level coefficient per metric."""

    def test_single_row_output(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_single_row_output()* `aggregate_arch_coefs` returns exactly one row."""
        _arch = aggregate_arch_coefs(_dim_baseline)
        assert len(_arch) == 1

    def test_default_tag_is_TAS(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_default_tag_is_TAS()* default tag `"TAS"` produces `\\theta_{TAS}` / `\\sigma_{TAS}` / ... columns."""
        _arch = aggregate_arch_coefs(_dim_baseline)
        for _coef in ("theta", "sigma", "eta", "phi", "epsilon"):
            assert f"\\{_coef}_{{TAS}}" in _arch.columns

    def test_custom_tag_flows_through(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_custom_tag_flows_through()* a caller-supplied `tag` becomes the subscript on every output column."""
        _arch = aggregate_arch_coefs(_dim_baseline, tag="TAS_base")
        assert "\\theta_{TAS_base}" in _arch.columns

    def test_theta_equals_sumL_over_sumK(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_theta_equals_sumL_over_sumK()* theta_arch = (sum L_i) / (sum K_i) across every artifact, recomputed by hand from the config setpoints."""
        _cfg = _dim_baseline["config"]
        _sum_L = sum(float(_a.vars[f"L_{{{_a.key}}}"]["_setpoint"])
                     for _a in _cfg.artifacts if _a.key != _FAIL)
        _sum_K = sum(float(_a.vars[f"K_{{{_a.key}}}"]["_setpoint"])
                     for _a in _cfg.artifacts if _a.key != _FAIL)
        _arch = aggregate_arch_coefs(_dim_baseline)
        _theta = float(_arch["\\theta_{TAS}"].iloc[0])
        assert _theta == pytest.approx(_sum_L / _sum_K, rel=1e-9)

    def test_epsilon_uses_cumulative_probability(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_epsilon_uses_cumulative_probability()* epsilon_arch = 1 - prod(1 - epsilon_i); value derives from the actual per-artifact epsilons."""
        _cfg = _dim_baseline["config"]
        _prod = 1.0
        for _a in _cfg.artifacts:
            if _a.key == _FAIL:
                continue
            _eps_i = float(_a.vars[f"\\epsilon_{{{_a.key}}}"]["_setpoint"])
            _prod *= (1.0 - _eps_i)
        _expected = 1.0 - _prod

        _arch = aggregate_arch_coefs(_dim_baseline)
        _eps = float(_arch["\\epsilon_{TAS}"].iloc[0])
        assert _eps == pytest.approx(_expected, rel=1e-9)

    def test_eta_matches_sum_chi_K_over_sum_mu_c(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_eta_matches_sum_chi_K_over_sum_mu_c()* eta_arch = (sum chi_i * K_i) / (sum mu_i * c_i); recomputed by hand from the config setpoints."""
        _cfg = _dim_baseline["config"]
        _num = 0.0
        _den = 0.0
        for _a in _cfg.artifacts:
            if _a.key == _FAIL:
                continue
            _chi = float(_a.vars[f"\\chi_{{{_a.key}}}"]["_setpoint"])
            _K = float(_a.vars[f"K_{{{_a.key}}}"]["_setpoint"])
            _mu = float(_a.vars[f"\\mu_{{{_a.key}}}"]["_setpoint"])
            _c = float(_a.vars[f"c_{{{_a.key}}}"]["_setpoint"])
            _num += _chi * _K
            _den += _mu * _c
        _expected = _num / _den

        _arch = aggregate_arch_coefs(_dim_baseline)
        _eta = float(_arch["\\eta_{TAS}"].iloc[0])
        assert _eta == pytest.approx(_expected, rel=1e-9)

    def test_architecture_coefficients_in_range(self, _dim_baseline: Dict[str, Any]) -> None:
        """*test_architecture_coefficients_in_range()* every architecture-level coefficient must sit in a sensible range for a stable baseline."""
        _arch = aggregate_arch_coefs(_dim_baseline)
        _theta = float(_arch["\\theta_{TAS}"].iloc[0])
        _phi = float(_arch["\\phi_{TAS}"].iloc[0])
        _eps = float(_arch["\\epsilon_{TAS}"].iloc[0])
        # occupancy + memory use are ratios in [0, 1]; compound epsilon also in [0, 1]
        assert 0.0 <= _theta <= 1.0
        assert 0.0 <= _phi <= 1.0
        assert 0.0 <= _eps <= 1.0


class TestAggregateSweepToArch:
    """**TestAggregateSweepToArch** `aggregate_sweep_to_arch` collapses per-artifact sweep arrays (from `sweep_arch`) into flat architecture-level arrays via sum-first aggregation point-by-point. Tests build a synthetic 2-artifact sweep so the contract can be checked without a real `sweep_arch` round-trip."""

    def _synthetic_sweep(self) -> Dict[str, Dict[str, np.ndarray]]:
        """*_synthetic_sweep()* build a 2-artifact x 3-point nested sweep dict with hand-controllable values; uniform (mu, c, K) per row so the architecture aggregate has a known closed form."""
        def _build(key: str) -> Dict[str, np.ndarray]:
            return {
                f"\\theta_{{{key}}}": np.array([0.1, 0.2, 0.3]),
                f"\\sigma_{{{key}}}": np.array([0.1, 0.2, 0.3]),
                f"\\eta_{{{key}}}": np.array([1.0, 2.0, 3.0]),
                f"\\phi_{{{key}}}": np.array([0.1, 0.2, 0.3]),
                f"c_{{{key}}}": np.array([1.0, 1.0, 1.0]),
                f"\\mu_{{{key}}}": np.array([10.0, 10.0, 10.0]),
                f"K_{{{key}}}": np.array([10.0, 10.0, 10.0]),
                f"\\lambda_{{{key}}}": np.array([1.0, 2.0, 3.0]),
            }
        return {"A": _build("A"), "B": _build("B")}

    def test_returns_empty_arrays_on_empty_input(self) -> None:
        """*test_returns_empty_arrays_on_empty_input()* empty sweep -> all output keys present with zero-length arrays."""
        _arch = aggregate_sweep_to_arch({})
        for _prefix in ("\\theta", "\\sigma", "\\eta", "\\phi",
                        "c", "\\mu", "K", "\\lambda"):
            _key = f"{_prefix}_{{TAS}}"
            assert _key in _arch
            assert len(_arch[_key]) == 0

    def test_keys_match_yoly_chart_shape(self) -> None:
        """*test_keys_match_yoly_chart_shape()* output carries the LaTeX-subscripted keys `plot_yoly_chart` looks up via prefix match."""
        _arch = aggregate_sweep_to_arch(self._synthetic_sweep())
        for _prefix in ("\\theta", "\\sigma", "\\eta", "\\phi",
                        "c", "\\mu", "K", "\\lambda"):
            assert f"{_prefix}_{{TAS}}" in _arch

    def test_lambda_arch_sums_per_artifact(self) -> None:
        """*test_lambda_arch_sums_per_artifact()* lambda_arch[k] = sum_i lambda_i[k]; with 2 artifacts both at [1, 2, 3], the arch sum is [2, 4, 6]."""
        _arch = aggregate_sweep_to_arch(self._synthetic_sweep())
        _expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(_arch["\\lambda_{TAS}"], _expected)

    def test_phi_equals_theta_under_constant_payload(self) -> None:
        """*test_phi_equals_theta_under_constant_payload()* in the CS-01 TAS schema, phi = M_act/M_buf = (L*delta)/(K*delta) = L/K = theta; aggregate_sweep_to_arch enforces phi_arch = theta_arch by construction."""
        _arch = aggregate_sweep_to_arch(self._synthetic_sweep())
        np.testing.assert_allclose(_arch["\\phi_{TAS}"], _arch["\\theta_{TAS}"])

    def test_custom_tag_flows_through(self) -> None:
        """*test_custom_tag_flows_through()* a caller-supplied `tag` becomes the subscript on every output key."""
        _arch = aggregate_sweep_to_arch(self._synthetic_sweep(), tag="ARCH")
        assert "\\theta_{ARCH}" in _arch
        assert "\\lambda_{ARCH}" in _arch
