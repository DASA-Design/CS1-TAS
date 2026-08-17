# -*- coding: utf-8 -*-
"""
Module test_jackson.py
======================

Sanity checks for the Jackson traffic-equation solver and rho-indexed helpers in `src.analytic.jackson`. The `solve_network()` wrapper is exercised end-to-end by the analytic method test suite.

    - **TestJacksonSolver** small worked examples where the expected per-node arrival rates are obvious from flow conservation.
    - **TestPerArtifactLambdas** the per-artifact lambda vector is linear in `lam_z` (Jackson linearity).
    - **TestPerArtifactRhos** `rho_i = lam_i / (c_i * mu_i)` finite for well-formed profiles; monotone in `lam_z`.
    - **TestLambdaZForRho** FR-3.5 closed-form inversion; bottleneck hits target rho exactly.
    - **TestBuildRhoGrid** rho-grid to lambda-grid map for the experiment orchestrator.
"""
# scientific stack
import numpy as np

# testing framework
import pytest

# modules under test
from src.analytic.jackson import (build_rho_grid,
                                  compute_lams_per_artifact,
                                  compute_rhos_per_artifact,
                                  invert_rho_to_lam_z,
                                  solve_jackson_lams)
from src.io import NetCfg, load_profile


class TestJacksonSolver:
    """**TestJacksonSolver** `solve_jackson_lams()` returns per-node arrival rates expected from flow conservation on small, hand-checkable topologies."""

    def test_two_node_feedforward(self) -> None:
        """*test_two_node_feedforward()* node 0 routes everything to node 1 (p=1). External arrivals at node 0 only. Expected: `lam_0 = lam_ext`, `lam_1 = p * lam_0`."""
        _P = np.array([[0.0, 1.0],
                       [0.0, 0.0]])
        _lams = solve_jackson_lams(_P, [10.0, 0.0])
        assert _lams[0] == pytest.approx(10.0)
        assert _lams[1] == pytest.approx(10.0)

    def test_two_node_split(self) -> None:
        """*test_two_node_split()* node 0 routes 70% to node 1 and 30% to exit. External arrivals only at node 0."""
        _P = np.array([[0.0, 0.7],
                       [0.0, 0.0]])
        _lams = solve_jackson_lams(_P, [100.0, 0.0])
        assert _lams[0] == pytest.approx(100.0)
        assert _lams[1] == pytest.approx(70.0)

    def test_no_external_no_flow(self) -> None:
        """*test_no_external_no_flow()* with zero external arrivals every node's effective arrival rate is zero regardless of routing."""
        _P = np.array([[0.0, 0.5],
                       [0.5, 0.0]])
        _lams = solve_jackson_lams(_P, [0.0, 0.0])
        assert np.allclose(_lams, 0.0)

    def test_shape_preserved(self) -> None:
        """*test_shape_preserved()* solver output shape matches input dimensionality (13-node TAS baseline)."""
        _P = np.zeros((13, 13))
        _lams = solve_jackson_lams(_P, np.zeros(13))
        assert _lams.shape == (13,)


# ---- rho-indexed helpers ----


@pytest.fixture(scope="module")
def _cfg() -> NetCfg:
    """*_cfg()* module-cached baseline profile used by every rho-indexed helper test."""
    return load_profile(adaptation="baseline")


class TestPerArtifactLambdas:
    """**TestPerArtifactLambdas** Jackson lambda vector is linear in `lam_z`."""

    def test_zero_entry_gives_zero_everywhere(self, _cfg: NetCfg) -> None:
        """*test_zero_entry_gives_zero_everywhere()* `lam_z=0` makes every artifact's lambda zero."""
        _lams = compute_lams_per_artifact(_cfg, 0.0)
        assert np.allclose(_lams, 0.0)

    def test_linear_in_lam_z(self, _cfg: NetCfg) -> None:
        """*test_linear_in_lam_z()* doubling `lam_z` doubles every `lam_i` (Jackson linearity)."""
        _a = compute_lams_per_artifact(_cfg, 10.0)
        _b = compute_lams_per_artifact(_cfg, 20.0)
        assert np.allclose(_b, 2.0 * _a, rtol=1e-9)


class TestPerArtifactRhos:
    """**TestPerArtifactRhos** `rho_i = lam_i / (c_i * mu_i)`; finite for well-formed profiles."""

    def test_all_finite_at_reasonable_lambda(self, _cfg: NetCfg) -> None:
        """*test_all_finite_at_reasonable_lambda()* every artifact's rho is finite at a plausible entry rate (no zero-capacity edge cases)."""
        _rhos = compute_rhos_per_artifact(_cfg, 10.0)
        assert np.all(np.isfinite(_rhos))

    def test_monotone_in_lam_z(self, _cfg: NetCfg) -> None:
        """*test_monotone_in_lam_z()* every component's rho is non-decreasing in `lam_z`."""
        _lo = compute_rhos_per_artifact(_cfg, 5.0)
        _hi = compute_rhos_per_artifact(_cfg, 50.0)
        assert np.all(_hi >= _lo - 1e-12)


class TestLambdaZForRho:
    """**TestLambdaZForRho** FR-3.5 closed-form inversion; bottleneck hits target rho exactly."""

    @pytest.mark.parametrize("_rho_target", [0.05, 0.20, 0.50, 0.80, 0.95])
    def test_bottleneck_hits_target(self, _cfg: NetCfg, _rho_target: float) -> None:
        """*test_bottleneck_hits_target()* inverted `lam_z` makes the identified bottleneck hit `rho_target` exactly and no other artifact exceeds it."""
        _lam_z, _bottleneck, _per_unit = invert_rho_to_lam_z(_cfg, _rho_target)
        assert _lam_z > 0
        assert _per_unit > 0
        _rhos = compute_rhos_per_artifact(_cfg, _lam_z)
        assert _rhos[_bottleneck] == pytest.approx(_rho_target, rel=1e-9)
        assert _rhos.max() == pytest.approx(_rho_target, rel=1e-9)

    def test_rejects_invalid_targets(self, _cfg: NetCfg) -> None:
        """*test_rejects_invalid_targets()* `rho_target` outside `(0, 1)` raises `ValueError`; endpoints included in the rejection set."""
        with pytest.raises(ValueError, match="must be in"):
            invert_rho_to_lam_z(_cfg, 0.0)
        with pytest.raises(ValueError, match="must be in"):
            invert_rho_to_lam_z(_cfg, 1.0)
        with pytest.raises(ValueError, match="must be in"):
            invert_rho_to_lam_z(_cfg, 1.5)


class TestBuildRhoGrid:
    """**TestBuildRhoGrid** rho-grid to lambda-grid map for the experiment orchestrator."""

    def test_full_grid_produces_increasing_lam_z(self, _cfg: NetCfg) -> None:
        """*test_full_grid_produces_increasing_lam_z()* the proof's rho-grid maps to a monotonically increasing `lam_z` sequence, with every tuple's rho matching its input."""
        _rho_grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.55,
                     0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        _grid = build_rho_grid(_cfg, _rho_grid)
        assert len(_grid) == len(_rho_grid)
        _lams = [_l for _, _l, _ in _grid]
        assert _lams == sorted(_lams)
        for (_rho_got, _lam, _b), _rho_expected in zip(_grid, _rho_grid):
            assert _rho_got == pytest.approx(_rho_expected)
            assert _lam > 0
            assert isinstance(_b, int)
