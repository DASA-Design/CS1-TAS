# -*- coding: utf-8 -*-
"""
Module test_networks.py
=======================

Shape + invariant checks for the configuration-sweep helpers in `src.dimensional.networks`. The two sweep shapes (independent per-artifact vs Jackson-propagated) carry different semantics; each test class pins one contract of one helper.

    - **TestSetpoint** `read_setpoint()` resolves the PACS-form variable key and raises on misses.
    - **TestSweepArtifact** per-artifact independent sweep returns aligned arrays with finite numeric values.
    - **TestSweepArtifacts** walks every artifact, honours `artifact_filter`, and returns a dict keyed by artifact.
    - **TestFindMaxStableLambdaFactor** the binary-search floor respects the utilisation cap.
    - **TestSweepArchitecture** Jackson-propagated whole-network sweep keeps rho < util_threshold at every stable point and aligns arrays across artifacts.

Sweeps can be slow; every test uses a trimmed `_QUICK_GRID` (one `mu_factor`, one `c`, one `K`, few `lambda_steps`) to keep the module under ~2 s.
"""
# data types
from typing import Any, Dict

# scientific stack
import numpy as np

# testing framework
import pytest

# modules under test
from src.analytic.jackson import solve_jackson_lams
from src.dimensional.networks import (_find_max_stable_lam_factor,
                                      read_setpoint,
                                      sweep_arch,
                                      sweep_artifact,
                                      sweep_artifacts)
from src.io import NetCfg, load_profile


# trimmed sweep grid: 4 stable points instead of the production ~30+ combos
_QUICK_GRID = {
    "mu_factor": [1.0],
    "c": [1],
    "K": [10],
    "lambda_steps": 4,
    "lambda_factor_min": 0.10,
    "util_threshold": 0.95,
}


# list of the symbol-prefix keys every sweep must emit per artifact
_SWEEP_PREFIXES = ("\\theta", "\\sigma", "\\eta", "\\phi",
                   "c", "\\mu", "K", "\\lambda")


@pytest.fixture(scope="module")
def _cfg() -> NetCfg:
    """*_cfg()* module-cached baseline profile used by every sweep test."""
    return load_profile(adaptation="baseline")


@pytest.fixture(scope="module")
def _tas1_vars(_cfg: NetCfg) -> Dict[str, Dict[str, Any]]:
    """*_tas1_vars()* per-artifact `vars` dict for TAS_{1} (pulled from the resolved config)."""
    _art = next(_a for _a in _cfg.artifacts if _a.key == "TAS_{1}")
    return _art.vars


class TestSetpoint:
    """**TestSetpoint** `read_setpoint()` resolves the PACS-form variable key and raises on misses."""

    def test_reads_setpoint_value(self, _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_reads_setpoint_value()* the lambda setpoint for TAS_{1} baseline is positive (entry external arrival rate)."""
        _lam = read_setpoint(_tas1_vars, "\\lambda", "TAS_{1}")
        assert _lam > 0

    def test_missing_prefix_raises(self, _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_missing_prefix_raises()* asking for a non-existent LaTeX prefix raises KeyError with the full symbol in the message."""
        with pytest.raises(KeyError, match=r"\\bogus_\{TAS_\{1\}\}"):
            read_setpoint(_tas1_vars, "\\bogus", "TAS_{1}")


class TestSweepArtifact:
    """**TestSweepArtifact** per-artifact independent sweep returns aligned arrays with finite numeric values."""

    def test_keys_cover_every_prefix(self, _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_keys_cover_every_prefix()* the returned dict carries one key per `{prefix}_{TAS_{1}}` symbol."""
        _out = sweep_artifact("TAS_{1}", _tas1_vars, _QUICK_GRID)
        for _p in _SWEEP_PREFIXES:
            assert f"{_p}_{{TAS_{{1}}}}" in _out

    def test_arrays_all_same_length(self, _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_arrays_all_same_length()* every per-symbol array has the same number of stable sweep points."""
        _out = sweep_artifact("TAS_{1}", _tas1_vars, _QUICK_GRID)
        _lengths = {len(_v) for _v in _out.values()}
        assert len(_lengths) == 1, f"ragged arrays: {_lengths}"

    def test_values_are_finite(self, _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_values_are_finite()* no NaN / inf leaks through from unstable points (the sweep drops those)."""
        _out = sweep_artifact("TAS_{1}", _tas1_vars, _QUICK_GRID)
        for _k, _v in _out.items():
            assert np.all(np.isfinite(_v)), f"{_k} has non-finite values"


class TestSweepArtifacts:
    """**TestSweepArtifacts** walks every artifact, honours `artifact_filter`, and returns a dict keyed by artifact."""

    def test_walks_every_artifact(self, _cfg: NetCfg) -> None:
        """*test_walks_every_artifact()* `set(out.keys()) == {a.key for a in cfg.artifacts}` (no-filter case)."""
        _out = sweep_artifacts(_cfg, _QUICK_GRID)
        _expected_keys = {_a.key for _a in _cfg.artifacts}
        assert set(_out.keys()) == _expected_keys

    def test_filter_limits_keys(self, _cfg: NetCfg) -> None:
        """*test_filter_limits_keys()* `set(out.keys()) == artifact_filter` when a non-empty filter is passed."""
        _want = {"TAS_{1}", "MAS_{1}"}
        _out = sweep_artifacts(_cfg, _QUICK_GRID, artifact_filter=_want)
        assert set(_out.keys()) == _want

    def test_block_matches_solo_shape(self,
                                      _cfg: NetCfg,
                                      _tas1_vars: Dict[str, Dict[str, Any]]) -> None:
        """*test_block_matches_solo_shape()* `set(multi["TAS_{1}"].keys()) == set(sweep_artifact(...).keys())` (the multi-artifact wrapper preserves the per-artifact key set)."""
        _multi = sweep_artifacts(_cfg, _QUICK_GRID,
                                 artifact_filter={"TAS_{1}"})
        _solo = sweep_artifact("TAS_{1}", _tas1_vars, _QUICK_GRID)
        assert set(_multi["TAS_{1}"].keys()) == set(_solo.keys())


class TestFindMaxStableLambdaFactor:
    """**TestFindMaxStableLambdaFactor** the binary-search floor respects the utilisation cap."""

    def test_factor_keeps_every_node_below_cap(self, _cfg: NetCfg) -> None:
        """*test_factor_keeps_every_node_below_cap()* at the returned factor every Jackson-propagated rho is below util_threshold."""
        _util = 0.95
        _mu = np.array([float(_a.mu) for _a in _cfg.artifacts])
        _c_int = 1
        _f = _find_max_stable_lam_factor(_cfg, _mu, _c_int, _util, iters=30)
        assert _f > 0
        _lams = solve_jackson_lams(_cfg.routing,
                                   _f * _cfg.build_lam_z_vec())
        _rhos = _lams / (_mu * float(_c_int))
        # binary search converges from below, so every node sits strictly under the cap at _f
        assert np.all(_rhos < _util)

    def test_zero_floor_on_empty_stable_region(self, _cfg: NetCfg) -> None:
        """*test_zero_floor_on_empty_stable_region()* a nonsensically small mu vector has no stable region; factor should bottom out near zero."""
        # mu = 1e-6 everywhere; any positive lambda saturates instantly
        _mu = np.full(len(_cfg.artifacts), 1e-6)
        _f = _find_max_stable_lam_factor(_cfg, _mu, 1, 0.95, iters=20)
        # converges toward 0 (binary search from [0, 100] narrows fast)
        assert _f < 1e-3


class TestSweepArchitecture:
    """**TestSweepArchitecture** Jackson-propagated whole-network sweep keeps rho < util_threshold at every stable point and aligns arrays across artifacts."""

    @pytest.fixture(scope="class")
    def _arch(self, _cfg: NetCfg) -> Dict[str, Dict[str, np.ndarray]]:
        """*_arch()* class-cached architecture sweep; expensive enough to share across tests."""
        return sweep_arch(_cfg, _QUICK_GRID)

    def test_walks_every_artifact(self,
                                  _cfg: NetCfg,
                                  _arch: Dict[str, Dict[str, np.ndarray]]) -> None:
        """*test_walks_every_artifact()* one top-level entry per artifact in the resolved config, except the synthetic FAIL_{1} sink (not a sweepable service)."""
        _expected_keys = {_a.key for _a in _cfg.artifacts if _a.key != "FAIL_{1}"}
        assert set(_arch.keys()) == _expected_keys

    def test_arrays_aligned_across_artifacts(self,
                                             _arch: Dict[str, Dict[str, np.ndarray]]) -> None:
        """*test_arrays_aligned_across_artifacts()* sample i across every artifact came from the same whole-network solve, so every artifact's array length must match."""
        _lengths = set()
        for _k, _block in _arch.items():
            for _sym_arr in _block.values():
                _lengths.add(len(_sym_arr))
        assert len(_lengths) == 1, f"ragged cross-artifact arrays: {_lengths}"

    def test_all_rhos_under_util_threshold(self,
                                           _arch: Dict[str, Dict[str, np.ndarray]]) -> None:
        """*test_all_rhos_under_util_threshold()* every stable sweep point has `lambda_i < util_threshold * c * mu_i` for every artifact."""
        _util = 0.95
        for _k, _block in _arch.items():
            _lam = _block[f"\\lambda_{{{_k}}}"]
            _mu = _block[f"\\mu_{{{_k}}}"]
            _c = _block[f"c_{{{_k}}}"]
            _cap = _util * _c * _mu
            _msg = f"{_k}: lambda exceeded util cap at some sweep point"
            assert np.all(_lam < _cap), _msg


class TestSweepNoise:
    """**TestSweepNoise** seeded white noise on the independent inputs jitters the coefficient cloud while keeping `noise=None` bit-identical, the run reproducible, the design knobs nominal, and `phi == theta`."""

    def test_no_noise_identical(self, _cfg: NetCfg) -> None:
        """*test_no_noise_identical()* `noise=None` reproduces the plain sweep exactly."""
        _a = sweep_arch(_cfg, _QUICK_GRID)
        _b = sweep_arch(_cfg, _QUICK_GRID, noise=None)
        _k = next(iter(_a))
        assert np.array_equal(_a[_k][f"\\theta_{{{_k}}}"], _b[_k][f"\\theta_{{{_k}}}"])

    def test_seeded_reproducible(self, _cfg: NetCfg) -> None:
        """*test_seeded_reproducible()* same seed gives identical noisy coefficients."""
        _a = sweep_arch(_cfg, _QUICK_GRID, noise=0.05, seed=42)
        _b = sweep_arch(_cfg, _QUICK_GRID, noise=0.05, seed=42)
        _k = next(iter(_a))
        assert np.array_equal(_a[_k][f"\\theta_{{{_k}}}"], _b[_k][f"\\theta_{{{_k}}}"])

    def test_noise_moves_coefs(self, _cfg: NetCfg) -> None:
        """*test_noise_moves_coefs()* a 5 % level shifts theta away from the clean values."""
        _clean = sweep_arch(_cfg, _QUICK_GRID)
        _noisy = sweep_arch(_cfg, _QUICK_GRID, noise=0.05, seed=42)
        _k = next(iter(_clean))
        _ct = _clean[_k][f"\\theta_{{{_k}}}"]
        _nt = _noisy[_k][f"\\theta_{{{_k}}}"]
        _m = min(len(_ct), len(_nt))
        assert not np.allclose(_ct[:_m], _nt[:_m])

    def test_phi_eq_theta_noisy(self, _cfg: NetCfg) -> None:
        """*test_phi_eq_theta_noisy()* phi stays equal to theta under noise (shared recomputed L)."""
        _noisy = sweep_arch(_cfg, _QUICK_GRID, noise=0.05, seed=42)
        for _k, _block in _noisy.items():
            assert np.allclose(_block[f"\\theta_{{{_k}}}"], _block[f"\\phi_{{{_k}}}"])

    def test_mu_metadata_nominal(self, _cfg: NetCfg) -> None:
        """*test_mu_metadata_nominal()* the recorded mu knob keeps its nominal grouping under noise (not jittered per point)."""
        _clean = sweep_arch(_cfg, _QUICK_GRID)
        _noisy = sweep_arch(_cfg, _QUICK_GRID, noise=0.05, seed=42)
        _k = next(iter(_clean))
        _uc = np.unique(np.round(_clean[_k][f"\\mu_{{{_k}}}"], 6))
        _un = np.unique(np.round(_noisy[_k][f"\\mu_{{{_k}}}"], 6))
        assert len(_un) == len(_uc)
