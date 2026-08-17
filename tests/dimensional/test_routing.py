# -*- coding: utf-8 -*-
"""
Module test_routing.py
======================

Sanity checks for the loss-network routing builder in `src.dimensional.routing`.

    - **TestRetrySplit**: `_retry_split` splits failure mass into a retry edge + give-up that sum to `f`, reduces to selection-only at `k=1`, and reproduces the baked s1 config values.
    - **TestBuildRouting**: `build_routing` reproduces the selection-only matrix at `retry=None`, adds the retry edge to the handler's dispatcher when retry is on (success forwards untouched, handler row still sums to 1), and redistributes a DS-handler retry over the DS pool weights.
"""
# data types
from typing import Dict, List

# scientific stack
import numpy as np

# testing framework
import pytest

# modules under test
from src.dimensional.routing import _retry_split, build_routing


# fixed mesh: 2 MAS + 1 AS + 1 DS + 6 TAS stages + FAIL sink
_MAS = ["MAS_{2}", "MAS_{4}"]
_AS = ["AS_{2}"]
_DS = ["DS_{1}"]
_EPS = {"MAS_{2}": 0.07, "MAS_{4}": 0.10, "AS_{2}": 0.04, "DS_{1}": 0.01}
_NODES = (["TAS_{1}", "TAS_{2}", "TAS_{3}"] + _MAS + _AS
          + ["TAS_{4}"] + _DS + ["TAS_{5}", "TAS_{6}", "FAIL_{1}"])


def _idx(key: str) -> int:
    """*_idx()* index of a node key in the fixed mesh."""
    return _NODES.index(key)


def _route(weights: Dict[str, tuple], retry=None) -> np.ndarray:
    """*_route()* build a routing matrix over the fixed mesh."""
    return build_routing(_NODES, _EPS, weights, _MAS, _AS, _DS, retry=retry)


class TestRetrySplit:
    """**TestRetrySplit** the per-handler failure split."""

    def test_k1_is_selection_only(self) -> None:
        """*test_k1_is_selection_only()* depth 1 leaks the full failure, no retry edge."""
        assert _retry_split(0.2, 1) == (0.0, 0.2)

    def test_zero_failure(self) -> None:
        """*test_zero_failure()* zero failure yields no edges."""
        assert _retry_split(0.0, 3) == (0.0, 0.0)

    def test_split_sums_to_f(self) -> None:
        """*test_split_sums_to_f()* retry edge + give-up == f at depth 3."""
        _re, _gu = _retry_split(0.12, 3)
        assert _re + _gu == pytest.approx(0.12)

    def test_matches_baked_s1(self) -> None:
        """*test_matches_baked_s1()* f=0.1228, k=3 reproduces the devlog s1 edges (retry 0.1212, give-up 0.0016)."""
        _re, _gu = _retry_split(0.1228, 3)
        assert _re == pytest.approx(0.12117, abs=1e-4)
        assert _gu == pytest.approx(0.00163, abs=1e-4)

    def test_full_failure_guarded(self) -> None:
        """*test_full_failure_guarded()* f>=1 leaks fully without a divide-by-zero."""
        assert _retry_split(1.0, 3) == (0.0, 1.0)


class TestBuildRouting:
    """**TestBuildRouting** selection-only and retry encodings of the handler rows."""

    def test_selection_only_handlers(self) -> None:
        """*test_selection_only_handlers()* retry=None leaks the raw pool failure at each handler and sets no retry edge."""
        _w = {"M": (0.5, 0.5), "A": (1.0,), "D": (1.0,)}
        _r = _route(_w, retry=None)
        _f_med = 0.5 * 0.07 + 0.5 * 0.10
        assert _r[_idx("TAS_{4}"), _idx("FAIL_{1}")] == pytest.approx(_f_med)
        assert _r[_idx("TAS_{4}"), _idx("TAS_{2}")] == pytest.approx(0.0)
        assert _r[_idx("TAS_{5}"), _idx("FAIL_{1}")] == pytest.approx(0.04)
        assert _r[_idx("TAS_{5}"), _idx("TAS_{3}")] == pytest.approx(0.0)
        assert _r[_idx("TAS_{6}"), _idx("FAIL_{1}")] == pytest.approx(0.01)

    def test_selection_only_tas4_row_sums_to_one(self) -> None:
        """*test_selection_only_tas4_row_sums_to_one()* the medical handler row conserves flow at retry=None."""
        _w = {"M": (0.5, 0.5), "A": (1.0,), "D": (1.0,)}
        _r = _route(_w, retry=None)
        assert _r[_idx("TAS_{4}")].sum() == pytest.approx(1.0)

    def test_retry_adds_dispatcher_edge(self) -> None:
        """*test_retry_adds_dispatcher_edge()* retry on TAS_{4} routes f_med*r to TAS_{2} and f_med*(1-r) to FAIL, leaving the success forwards unchanged."""
        _w = {"M": (0.5, 0.5), "A": (1.0,), "D": (1.0,)}
        _sel = _route(_w, retry=None)
        _ret = _route(_w, retry={"TAS_{4}": 3})
        _f_med = 0.5 * 0.07 + 0.5 * 0.10
        _re, _gu = _retry_split(_f_med, 3)
        assert _ret[_idx("TAS_{4}"), _idx("TAS_{2}")] == pytest.approx(_re)
        assert _ret[_idx("TAS_{4}"), _idx("FAIL_{1}")] == pytest.approx(_gu)
        # success forwards (to alarm + drug) are identical with and without retry
        assert _ret[_idx("TAS_{4}"), _idx("TAS_{3}")] == pytest.approx(_sel[_idx("TAS_{4}"), _idx("TAS_{3}")])
        assert _ret[_idx("TAS_{4}"), _idx("DS_{1}")] == pytest.approx(_sel[_idx("TAS_{4}"), _idx("DS_{1}")])

    def test_retry_tas4_row_still_conserves(self) -> None:
        """*test_retry_tas4_row_still_conserves()* the medical handler row sums to 1 with retry on."""
        _w = {"M": (0.5, 0.5), "A": (1.0,), "D": (1.0,)}
        _r = _route(_w, retry={"TAS_{4}": 3})
        assert _r[_idx("TAS_{4}")].sum() == pytest.approx(1.0)

    def test_ds_retry_redistributes_over_pool(self) -> None:
        """*test_ds_retry_redistributes_over_pool()* a DS-handler retry splits the retry mass over the DS dispatch weights."""
        _ds = ["DS_{1}", "DS_{5}"]
        _eps = {"MAS_{2}": 0.07, "AS_{2}": 0.04, "DS_{1}": 0.01, "DS_{5}": 0.02}
        _nodes = (["TAS_{1}", "TAS_{2}", "TAS_{3}", "MAS_{2}", "AS_{2}", "TAS_{4}"]
                  + _ds + ["TAS_{5}", "TAS_{6}", "FAIL_{1}"])
        _w = {"M": (1.0,), "A": (1.0,), "D": (0.6, 0.4)}
        _r = build_routing(_nodes, _eps, _w, ["MAS_{2}"], ["AS_{2}"], _ds,
                           retry={"TAS_{6}": 3})
        _f_drg = 0.6 * 0.01 + 0.4 * 0.02
        _re, _gu = _retry_split(_f_drg, 3)
        _i6 = _nodes.index("TAS_{6}")
        assert _r[_i6, _nodes.index("DS_{1}")] == pytest.approx(_re * 0.6)
        assert _r[_i6, _nodes.index("DS_{5}")] == pytest.approx(_re * 0.4)
        assert _r[_i6, _nodes.index("FAIL_{1}")] == pytest.approx(_gu)
