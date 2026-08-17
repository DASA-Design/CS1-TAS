# -*- coding: utf-8 -*-
"""
Module test_results.py
======================

Sanity checks for `src.io.results.load_qn_result` and `load_requirements`. The loaders are thin wrappers over `json.load` against `data/results/<method>/<adp>/...`; the tests verify path resolution, the returned dict shape, and the clear-error contract for missing files / unknown method or adaptation names.
"""
# testing framework
import pytest

# module under test
from src.io.results import load_qn_result, load_requirements


class TestLoadQnResult:
    """**TestLoadQnResult** the per-run envelope loader resolves paths correctly and returns the expected top-level keys for each method's persisted JSON."""

    def test_loads_analytic_baseline(self) -> None:
        """*test_loads_analytic_baseline()* `network` and `routing` and `lambda_z` keys are present; `network["W_e2e"]` is a positive float."""
        _doc = load_qn_result("analytic", "baseline")
        assert "network" in _doc
        assert "routing" in _doc
        assert "lambda_z" in _doc
        assert float(_doc["network"]["W_e2e"]) > 0

    def test_loads_stochastic_s2(self) -> None:
        """*test_loads_stochastic_s2()* stochastic envelope carries `network` + `nodes` + `method_config`."""
        _doc = load_qn_result("stochastic", "s2")
        assert "network" in _doc
        assert "nodes" in _doc
        assert "method_config" in _doc

    def test_loads_dimensional_aggregate(self) -> None:
        """*test_loads_dimensional_aggregate()* dimensional envelope carries `artifacts` alongside the shared `network` + `nodes` keys."""
        _doc = load_qn_result("dimensional", "aggregate")
        assert "network" in _doc
        assert "nodes" in _doc
        assert "artifacts" in _doc

    def test_rejects_unknown_method(self) -> None:
        """*test_rejects_unknown_method()* `method='experimental'` raises ValueError listing the allowed methods."""
        with pytest.raises(ValueError, match="unknown method"):
            load_qn_result("experimental", "baseline")

    def test_rejects_unknown_adaptation(self) -> None:
        """*test_rejects_unknown_adaptation()* `adp='s3'` raises ValueError listing the allowed adaptations."""
        with pytest.raises(ValueError, match="unknown adaptation"):
            load_qn_result("analytic", "s3")


class TestLoadRequirements:
    """**TestLoadRequirements** the verdict loader returns `{R1, R2}` with the expected per-requirement keys (including the per-node `contributions` list)."""

    def test_shape(self) -> None:
        """*test_shape()* `R1` and `R2` are top-level keys; each carries `value`, `threshold`, `pass`, `contributions`."""
        _req = load_requirements("analytic", "baseline")
        assert set(_req.keys()) == {"R1", "R2"}
        for _k in ("R1", "R2"):
            assert "value" in _req[_k]
            assert "threshold" in _req[_k]
            assert "pass" in _req[_k]
            assert "contributions" in _req[_k]

    def test_contributions_list(self) -> None:
        """*test_contributions_list()* every verdict's `contributions` is a non-empty list (top-K per-node drivers)."""
        _req = load_requirements("analytic", "baseline")
        for _k in ("R1", "R2"):
            _contribs = _req[_k]["contributions"]
            assert isinstance(_contribs, list)
            assert len(_contribs) > 0

    def test_rejects_unknown_method(self) -> None:
        """*test_rejects_unknown_method()* `method='experimental'` raises ValueError."""
        with pytest.raises(ValueError, match="unknown method"):
            load_requirements("experimental", "baseline")
