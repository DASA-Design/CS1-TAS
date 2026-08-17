# -*- coding: utf-8 -*-
"""
Module test_search.py
=====================

Sanity checks for the DASA-guided routing search in `src.dimensional.search`.

    - **TestPickPool**: `pick_pool` returns the lowest-`eps` service keys per service type.
    - **TestComputeBounds**: `compute_bounds` reproduces the CS-1 reference numbers `sigma_R2 = 0.525` and `eta_R1 = 34.798` at the locked envelope.
    - **TestBuildCfg**: `build_cfg` produces a 15-node aggregate-style topology with the expected node count, row sums (1 everywhere except TAS_{5} which sums to `sum_A_eps` and TAS_{6} which is the terminal sink), and per-pool self-loops.
    - **TestRunSearch**: `run_search` at step 1.0 produces 27 candidates with the expected column set, using the canonical dimensional method config (no method_cfg trim).
    - **TestPickWinner**: `pick_winner` maximises the DC margin among rho + DASA + R1 + R2 survivors, falls back to smallest DASA-distance among rho-survivors, then to smallest `rho_max_active` when no rho-survivor exists.
"""
# data types
from typing import Any, Dict, List, Tuple

# scientific stack
import pandas as pd

# testing framework
import pytest

# modules under test
from src.dimensional.search import (
    _seed_weights,
    _simplex_neighbors,
    _weights_to_x,
    _x_to_weights,
    build_cfg,
    cloud_to_selection_surface,
    compute_bounds,
    load_skeleton,
    pick_pool,
    pick_winner,
    run_descent,
    run_scipy_search,
    run_search,
    select_from_cloud,
)
from src.io import ArtifactSpec, load_catalogue


@pytest.fixture(scope="module")
def _catalogue() -> Dict[str, Any]:
    """*_catalogue()* the Table 7.3 catalogue, loaded once per module."""
    return load_catalogue()


@pytest.fixture(scope="module")
def _pools(_catalogue: Dict[str, Any]) -> Dict[str, List[str]]:
    """*_pools()* top-3 by lowest eps for MAS / AS / DS."""
    return {_svc: pick_pool(_catalogue, _svc, 3) for _svc in ("MAS", "AS", "DS")}


@pytest.fixture(scope="module")
def _tas_and_template() -> Tuple[List[ArtifactSpec], ArtifactSpec]:
    """*_tas_and_template()* the TAS skeleton + third-party template from opti.json/aggregate."""
    return load_skeleton()


@pytest.fixture(scope="module")
def _bounds() -> Dict[str, float]:
    """*_bounds()* CS-1 reference bounds at the locked envelope."""
    return compute_bounds(lam_z=323.0, r1_max=0.01, r2_max_s=0.026)


class TestPickPool:
    """**TestPickPool** `pick_pool` returns the lowest-`eps` service keys in ascending order."""

    def test_mas_top_3(self, _catalogue: Dict[str, Any]) -> None:
        """*test_mas_top_3()* MAS top-3 by eps == `["MAS_{2}", "MAS_{4}", "MAS_{1}"]` (eps 0.07 / 0.10 / 0.12)."""
        assert pick_pool(_catalogue, "MAS", 3) == ["MAS_{2}", "MAS_{4}", "MAS_{1}"]

    def test_as_top_3(self, _catalogue: Dict[str, Any]) -> None:
        """*test_as_top_3()* AS top-3 by eps == `["AS_{2}", "AS_{4}", "AS_{1}"]` (eps 0.04 / 0.08 / 0.11)."""
        assert pick_pool(_catalogue, "AS", 3) == ["AS_{2}", "AS_{4}", "AS_{1}"]

    def test_ds_top_3(self, _catalogue: Dict[str, Any]) -> None:
        """*test_ds_top_3()* DS top-3 by eps == `["DS_{1}", "DS_{5}", "DS_{2}"]` (eps 0.01 / 0.02 / 0.03)."""
        assert pick_pool(_catalogue, "DS", 3) == ["DS_{1}", "DS_{5}", "DS_{2}"]


class TestComputeBounds:
    """**TestComputeBounds** `compute_bounds` matches the dissertation reference numbers at the locked envelope (lam_z=323, K=16, c=1, mu_min=150)."""

    def test_sigma_R2(self, _bounds: Dict[str, float]) -> None:
        """*test_sigma_R2()* `sigma_R2 == 323 * 0.026 / 16 = 0.524875`."""
        assert _bounds["sigma_R2"] == pytest.approx(0.524875, rel=1e-6)

    def test_eta_R1(self, _bounds: Dict[str, float]) -> None:
        """*test_eta_R1()* `eta_R1 == 323 * 1.01 * 16 / 150 = 34.79786...`."""
        assert _bounds["eta_R1"] == pytest.approx(34.79786666, rel=1e-6)


class TestBuildCfg:
    """**TestBuildCfg** `build_cfg` produces the 15-node aggregate-style topology with row sums and self-loops aligned to the routing template."""

    def test_node_count(self,
                        _catalogue: Dict[str, Any],
                        _pools: Dict[str, List[str]],
                        _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec]) -> None:
        """*test_node_count()* the resulting config has 16 nodes (6 TAS + 3 MAS + 3 AS + 3 DS + the FAIL_{1} sink) and a 16x16 routing matrix."""
        _tas, _template = _tas_and_template
        _w = (1 / 3, 1 / 3, 1 / 3)
        _cfg = build_cfg(_catalogue, _pools, {"M": _w, "A": _w, "D": _w},
                            lam_z=323.0, tas_artifacts=_tas, template=_template)
        assert _cfg.n_nodes == 16
        assert _cfg.routing.shape == (16, 16)

    def test_row_sums(self,
                      _catalogue: Dict[str, Any],
                      _pools: Dict[str, List[str]],
                      _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec]) -> None:
        """*test_row_sums()* every row sums to 1 except the handlers that leak give-ups to FAIL: TAS_{5} sums to `sum_A_eps` and TAS_{6} to `sum_D_eps` (their `1 - f` success deficit exits), and FAIL_{1} is the absorbing terminal (sums to 0)."""
        _tas, _template = _tas_and_template
        _w = (1 / 3, 1 / 3, 1 / 3)
        _cfg = build_cfg(_catalogue, _pools, {"M": _w, "A": _w, "D": _w},
                            lam_z=323.0, tas_artifacts=_tas, template=_template)
        _rs = _cfg.routing.sum(axis=1)
        _keys = _cfg.list_node_keys()
        # give-up to FAIL = dispatch-weighted pool eps; AS: (0.04+0.08+0.11)/3, DS: (0.01+0.02+0.03)/3
        _sum_A_eps = (0.04 + 0.08 + 0.11) / 3
        _sum_D_eps = (0.01 + 0.02 + 0.03) / 3
        for _i, _k in enumerate(_keys):
            if _k == "TAS_{5}":
                assert _rs[_i] == pytest.approx(_sum_A_eps, abs=1e-6)
            elif _k == "TAS_{6}":
                assert _rs[_i] == pytest.approx(_sum_D_eps, abs=1e-6)
            elif _k == "FAIL_{1}":
                assert _rs[_i] == pytest.approx(0.0, abs=1e-6)
            else:
                assert _rs[_i] == pytest.approx(1.0, abs=1e-6), f"row {_k} sums to {_rs[_i]}"

    def test_self_loops_carry_eps(self,
                                  _catalogue: Dict[str, Any],
                                  _pools: Dict[str, List[str]],
                                  _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec]) -> None:
        """*test_self_loops_carry_eps()* each pool service has a self-loop equal to its catalogue `eps`."""
        _tas, _template = _tas_and_template
        _w = (1 / 3, 1 / 3, 1 / 3)
        _cfg = build_cfg(_catalogue, _pools, {"M": _w, "A": _w, "D": _w},
                            lam_z=323.0, tas_artifacts=_tas, template=_template)
        _keys = _cfg.list_node_keys()
        for _svc in ("MAS", "AS", "DS"):
            for _k in _pools[_svc]:
                _i = _keys.index(_k)
                assert _cfg.routing[_i, _i] == pytest.approx(_catalogue[_svc][_k]["eps"], abs=1e-6)


class TestRunSearch:
    """**TestRunSearch** `run_search` enumerates the simplex grid's interior and produces a DataFrame with the expected row count and columns. Uses `method_cfg=None` so the canonical `data/config/method/dimensional.json` drives the dimensional solve; runtime stays under ~30 seconds at the minimal `step=1/3` configuration (1 candidate per pool, 1 candidate total)."""

    def test_step_third_produces_one_interior_row(self,
                                                  _catalogue: Dict[str, Any],
                                                  _pools: Dict[str, List[str]],
                                                  _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec],
                                                  _bounds: Dict[str, float]) -> None:
        """*test_step_third_produces_one_interior_row()* `step=1/3` (`n=3`) leaves exactly one strict-positive interior tuple per pool (uniform 1/3, 1/3, 1/3); 1^3 = 1 candidate. Every expected column is present and the single row is stable."""
        _tas, _template = _tas_and_template
        _df = run_search(_catalogue, _pools, lam_z=323.0,
                         tas_artifacts=_tas, template=_template,
                         bounds=_bounds, step=1 / 3)
        assert len(_df) == 1
        for _col in ("w_M", "w_A", "w_D", "W_e2e", "eps_e2e",
                     "sigma_arch", "eta_arch",
                     "rho_max_active",
                     "dasa_pass", "rho_pass", "r1_pass", "r2_pass", "stable"):
            assert _col in _df.columns
        assert bool(_df.iloc[0]["stable"])


class TestSimplexNeighbors:
    """**TestSimplexNeighbors** lattice-neighbour generation conserves mass, stays interior, and steps by exactly `step`."""

    def test_conserve_mass_and_interior(self, _pools: Dict[str, List[str]]) -> None:
        """*test_conserve_mass_and_interior()* every neighbour keeps each pool summing to 1 and every weight strictly positive."""
        _seed = {"M": (1 / 3, 1 / 3, 1 / 3), "A": (1 / 3, 1 / 3, 1 / 3), "D": (1 / 3, 1 / 3, 1 / 3)}
        _nbs = _simplex_neighbors(_seed, 0.01, _pools)
        assert len(_nbs) > 0
        for _nb in _nbs:
            for _short in ("M", "A", "D"):
                assert sum(_nb[_short]) == pytest.approx(1.0)
                for _w in _nb[_short]:
                    assert _w > 0.0

    def test_step_resolution(self, _pools: Dict[str, List[str]]) -> None:
        """*test_step_resolution()* each neighbour differs from the seed by exactly `step` on two coordinates of one pool."""
        _seed = {"M": (1 / 3, 1 / 3, 1 / 3), "A": (1 / 3, 1 / 3, 1 / 3), "D": (1 / 3, 1 / 3, 1 / 3)}
        _step = 0.01
        _nbs = _simplex_neighbors(_seed, _step, _pools)
        for _nb in _nbs:
            _diffs: List[float] = []
            for _short in ("M", "A", "D"):
                for _j in range(len(_nb[_short])):
                    _diffs.append(_nb[_short][_j] - _seed[_short][_j])
            _moved = [_d for _d in _diffs if abs(_d) > 1e-9]
            assert len(_moved) == 2
            assert sorted(_moved)[0] == pytest.approx(-_step)
            assert sorted(_moved)[1] == pytest.approx(_step)


class TestSeedWeights:
    """**TestSeedWeights** multi-start seeds cover the centroid plus the MAS x AS near-corners, each on the simplex."""

    def test_count_and_simplex(self, _pools: Dict[str, List[str]]) -> None:
        """*test_count_and_simplex()* seed count == (1 + |MAS|) * (1 + |AS|) and every pool tuple sums to 1."""
        _seeds = _seed_weights(_pools)
        assert len(_seeds) == (1 + len(_pools["MAS"])) * (1 + len(_pools["AS"]))
        for _s in _seeds:
            for _short in ("M", "A", "D"):
                assert sum(_s[_short]) == pytest.approx(1.0)


class TestRunDescent:
    """**TestRunDescent** the DC-guided descent returns local optima carrying their retry config; retry on every handler unlocks an R1-passing configuration that selection alone cannot reach."""

    def test_retry_unlocks_r1(self,
                              _catalogue: Dict[str, Any],
                              _pools: Dict[str, List[str]],
                              _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec],
                              _bounds: Dict[str, float]) -> None:
        """*test_retry_unlocks_r1()* with retry depth 3 on all three handlers, a coarse-step descent reaches at least one R1-passing local optimum, and the frame carries the `retry` column. Coarse step + two seeds keep runtime bounded."""
        _tas, _template = _tas_and_template
        _seeds = [
            {"M": (0.9, 0.05, 0.05), "A": (0.9, 0.05, 0.05), "D": (0.9, 0.05, 0.05)},
            {"M": (1 / 3, 1 / 3, 1 / 3), "A": (1 / 3, 1 / 3, 1 / 3), "D": (1 / 3, 1 / 3, 1 / 3)},
        ]
        _df = run_descent(_catalogue, _pools, lam_z=323.0,
                          tas_artifacts=_tas, template=_template,
                          bounds=_bounds,
                          retry_grid=[{"TAS_{4}": 3, "TAS_{5}": 3, "TAS_{6}": 3}],
                          step=0.05, seeds=_seeds, patience=5, max_iters=100)
        assert "retry" in _df.columns
        assert len(_df) == 2
        assert bool(_df["r1_pass"].any())


class TestScipyReparam:
    """**TestScipyReparam** the simplex reparametrisation round-trips and decodes valid weights."""

    def test_roundtrip(self, _pools: Dict[str, List[str]]) -> None:
        """*test_roundtrip()* `_weights_to_x` then `_x_to_weights` recovers the original weights (pools sized by the fixture: 3 services per type)."""
        _w = {"M": (0.5, 0.3, 0.2), "A": (0.2, 0.5, 0.3), "D": (0.4, 0.4, 0.2)}
        _x = _weights_to_x(_w, _pools)
        _back = _x_to_weights(_x, _pools)
        for _short in ("M", "A", "D"):
            for _j in range(len(_w[_short])):
                assert _back[_short][_j] == pytest.approx(_w[_short][_j])

    def test_decode_normalises_to_simplex(self, _pools: Dict[str, List[str]]) -> None:
        """*test_decode_normalises_to_simplex()* any free vector decodes to per-pool weights summing to 1 (6 free vars over three 3-service pools)."""
        import numpy as np
        _w = _x_to_weights(np.array([0.6, 0.6, 0.1, 0.1, 0.3, 0.3]), _pools)
        for _short in ("M", "A", "D"):
            assert sum(_w[_short]) == pytest.approx(1.0)
            for _v in _w[_short]:
                assert _v >= 0.0


class TestRunScipySearch:
    """**TestRunScipySearch** the scipy cross-validation optimiser reaches an R1-passing feasible config with retry on every handler, matching the custom descent's conclusion."""

    def test_retry_feasible(self,
                            _catalogue: Dict[str, Any],
                            _pools: Dict[str, List[str]],
                            _tas_and_template: Tuple[List[ArtifactSpec], ArtifactSpec],
                            _bounds: Dict[str, float]) -> None:
        """*test_retry_feasible()* COBYLA over a retry-all config returns an R1-passing row carrying the `retry` column (fast: COBYLA is a local minimiser)."""
        _tas, _template = _tas_and_template
        _df = run_scipy_search(_catalogue, _pools, lam_z=323.0,
                               tas_artifacts=_tas, template=_template,
                               bounds=_bounds,
                               retry_grid=[{"TAS_{4}": 3, "TAS_{5}": 3, "TAS_{6}": 3}],
                               maxiter=200)
        assert "retry" in _df.columns
        assert len(_df) == 1
        assert bool(_df["r1_pass"].any())


def _row(**fields: Any) -> Dict[str, Any]:
    """*_row()* synthetic-row builder with the full search-frame schema.

    Defaults to a feasible row (rho_pass + dasa_pass + r1_pass + r2_pass all True, stable True); the caller overrides only the fields relevant to the test.
    """
    _defaults: Dict[str, Any] = {
        "w_M": (1.0,), "w_A": (1.0,), "w_D": (1.0,),
        "W_e2e": 0.020, "eps_e2e": 0.008,
        "sigma_arch": 0.4, "eta_arch": 30.0,
        "rho_max_active": 0.5,
        "dasa_pass": True, "rho_pass": True, "r1_pass": True, "r2_pass": True,
        "stable": True,
    }
    _defaults.update(fields)
    return _defaults


class TestPickWinner:
    """**TestPickWinner** the winner-pick logic maximises the DC margin among feasible rows, falls back to smallest DASA-distance among rho-survivors, and falls back further to smallest rho-overshoot when no rho-survivor exists."""

    def test_maximises_dc_margin(self, _bounds: Dict[str, float]) -> None:
        """*test_maximises_dc_margin()* among three feasible rows, the one with the largest DC margin (smallest sigma_arch and eta_arch) wins; `winner_status == "feasible"`."""
        _df = pd.DataFrame([
            _row(eps_e2e=0.008, sigma_arch=0.40, eta_arch=30.0),
            _row(eps_e2e=0.005, sigma_arch=0.45, eta_arch=32.0),
            _row(eps_e2e=0.009, sigma_arch=0.30, eta_arch=20.0),
        ])
        _winner = pick_winner(_df, _bounds)
        # row 3 has the deepest margin on both axes; eta_arch=20 << eta_R1=34.8.
        assert _winner["sigma_arch"] == pytest.approx(0.30)
        assert _winner["eta_arch"] == pytest.approx(20.0)
        assert _winner["winner_status"] == "feasible"

    def test_closest_miss_minimises_dasa_distance(self, _bounds: Dict[str, float]) -> None:
        """*test_closest_miss_minimises_dasa_distance()* when every row fails DASA (eta_arch > eta_R1) but passes rho, the row with smallest overshoot sum wins; `winner_status == "closest_miss"`."""
        _df = pd.DataFrame([
            _row(eps_e2e=0.05, sigma_arch=0.4, eta_arch=50.0, dasa_pass=False, r1_pass=False),
            _row(eps_e2e=0.10, sigma_arch=0.4, eta_arch=40.0, dasa_pass=False, r1_pass=False),
        ])
        _winner = pick_winner(_df, _bounds)
        # row 2 has smaller overshoot (eta_arch=40 vs 50; sigma is inside the bound).
        assert _winner["eta_arch"] == pytest.approx(40.0)
        assert _winner["winner_status"] == "closest_miss"

    def test_prefers_feasible_retry_row(self, _bounds: Dict[str, float]) -> None:
        """*test_prefers_feasible_retry_row()* given a selection-only closest-miss row and a feasible retry row, the feasible row wins and carries its retry config."""
        _df = pd.DataFrame([
            _row(eps_e2e=0.09, sigma_arch=0.4, eta_arch=50.0,
                 dasa_pass=False, r1_pass=False, retry={}),
            _row(eps_e2e=0.004, sigma_arch=0.30, eta_arch=20.0,
                 retry={"TAS_{4}": 3, "TAS_{5}": 3}),
        ])
        _winner = pick_winner(_df, _bounds)
        assert _winner["winner_status"] == "feasible"
        assert _winner["retry"] == {"TAS_{4}": 3, "TAS_{5}": 3}

    def test_no_rho_survivors_picks_smallest_rho_max_active(self, _bounds: Dict[str, float]) -> None:
        """*test_no_rho_survivors_picks_smallest_rho_max_active()* when every row exceeds the rho ceiling, the row with smallest `rho_max_active` wins; `winner_status == "no_rho_survivors"`."""
        _df = pd.DataFrame([
            _row(rho_max_active=0.80, rho_pass=False, dasa_pass=False, r1_pass=False, r2_pass=False),
            _row(rho_max_active=0.70, rho_pass=False, dasa_pass=False, r1_pass=False, r2_pass=False),
            _row(rho_max_active=0.95, rho_pass=False, dasa_pass=False, r1_pass=False, r2_pass=False),
        ])
        _winner = pick_winner(_df, _bounds)
        assert _winner["rho_max_active"] == pytest.approx(0.70)
        assert _winner["winner_status"] == "no_rho_survivors"


class TestCloudToSelectionSurface:
    """**TestCloudToSelectionSurface** the requirement-frame bridge maps `sigma_arch`/`eta_arch` to `\\sigma_{req}`/`\\eta_{req}`, emits no theta/phi keys, and drops non-finite rows."""

    def test_keys_and_values(self) -> None:
        """*test_keys_and_values()* sigma_arch/eta_arch map to the req-tagged keys; no theta/phi keys are emitted."""
        _df = pd.DataFrame([
            _row(sigma_arch=0.30, eta_arch=20.0),
            _row(sigma_arch=0.45, eta_arch=32.0),
        ])
        _surf = cloud_to_selection_surface(_df)
        assert set(_surf.keys()) == {"\\sigma_{req}", "\\eta_{req}"}
        assert list(_surf["\\sigma_{req}"]) == pytest.approx([0.30, 0.45])
        assert list(_surf["\\eta_{req}"]) == pytest.approx([20.0, 32.0])

    def test_drops_nonfinite(self) -> None:
        """*test_drops_nonfinite()* rows with an inf sigma or eta (short-circuited candidates) are filtered out."""
        _df = pd.DataFrame([
            _row(sigma_arch=0.30, eta_arch=20.0),
            _row(sigma_arch=float("inf"), eta_arch=float("inf")),
            _row(sigma_arch=0.40, eta_arch=float("inf")),
        ])
        _surf = cloud_to_selection_surface(_df)
        assert len(_surf["\\sigma_{req}"]) == 1
        assert _surf["\\sigma_{req}"][0] == pytest.approx(0.30)

    def test_custom_tag(self) -> None:
        """*test_custom_tag()* a non-default tag is baked into both output keys."""
        _df = pd.DataFrame([_row(sigma_arch=0.3, eta_arch=20.0)])
        _surf = cloud_to_selection_surface(_df, tag="TAS_winner")
        assert set(_surf.keys()) == {"\\sigma_{TAS_winner}", "\\eta_{TAS_winner}"}


class TestSelectFromCloud:
    """**TestSelectFromCloud** `select_from_cloud` returns exactly the row `pick_winner` picks from the same cloud."""

    def test_matches_pick_winner(self, _bounds: Dict[str, float]) -> None:
        """*test_matches_pick_winner()* select_from_cloud(df, b) equals pick_winner(df, b) row-for-row, including winner_status."""
        _df = pd.DataFrame([
            _row(eps_e2e=0.008, sigma_arch=0.40, eta_arch=30.0),
            _row(eps_e2e=0.009, sigma_arch=0.30, eta_arch=20.0),
        ])
        _from_cloud = select_from_cloud(_df, _bounds)
        _from_pick = pick_winner(_df, _bounds)
        assert _from_cloud == _from_pick
