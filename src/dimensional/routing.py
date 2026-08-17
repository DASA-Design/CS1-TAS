# -*- coding: utf-8 -*-
"""
Module routing.py
=================

Loss-network routing-matrix builder for the CS-1 DASA-guided search.

Assembles the TAS routing matrix from chosen pool services, dispatch weights, and an optional per-handler retry-depth map. Failures leak to an absorbing `FAIL_{1}` sink; availability is read from flow downstream (`eps_e2e = lambda_FAIL / lambda_z`). With `retry=None` the matrix is selection-only (dispatch lever in isolation, no recovery), which reproduces the search's original encoding byte-for-byte. With a retry depth `k >= 2` on a handler, the per-attempt failure mass splits into a retry edge (re-dispatch to the handler's dispatcher) and a give-up edge (to FAIL) via `_retry_split`.

This module imports only numpy + typing, so it is acyclic and is re-exported from `src.dimensional.__init__`. The search module (`src.dimensional.search`) calls `build_routing` through a thin `_build_routing` wrapper.

Public API:
    - `build_routing(node_keys, eps_by_key, weights, mas_keys, as_keys, ds_keys, *, retry)` -> the routing ndarray.
    - `pool_eps(weights, keys, eps_by_key)` -> dispatch-weighted pool failure (used by the search to report `f_med` etc.).
"""
# native python modules
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# scientific stack
import numpy as np


# absorbing failure-sink node (loss-network); candidates leak give-ups here and
# availability is read downstream as eps_e2e = lambda_FAIL / lambda_z
_FAIL_KEY = "FAIL_{1}"

# workflow split at TAS_{1}: 75% follow the normal analysis path through TAS_{2};
# 25% bypass via TAS_{3} (panic button -> alarm)
_W_ANALYSE = 0.75
_W_ALARM = 0.25

# TAS_{4} medical-handler success forwards: of the calls that succeed (1 - f_med),
# 34% loop back to the alarm path (TAS_{3}) and 66% dispatch a drug order (DS pool)
_W_MED_TO_ALARM = 0.34
_W_MED_TO_DRUG = 0.66


def pool_eps(weights: Tuple[float, ...],
             keys: List[str],
             eps_by_key: Dict[str, float]) -> float:
    """*pool_eps()* compute `sum(w_i * eps_i)` over a pool.

    The dispatch-weighted failure rate seen at a TAS handler: each pool service contributes its catalogue `eps` scaled by the fraction of traffic the dispatch weights send to it.

    Args:
        weights (Tuple[float, ...]): per-pool dispatch weights.
        keys (List[str]): pool service keys aligned with `weights`.
        eps_by_key (Dict[str, float]): catalogue eps lookup.

    Returns:
        float: weighted-eps sum.
    """
    _total = 0.0
    for _j, _k in enumerate(keys):
        _total = _total + weights[_j] * eps_by_key[_k]
    return _total


def _retry_split(f: float, k: int) -> Tuple[float, float]:
    """*_retry_split()* split a handler's per-attempt failure mass into a retry edge and a give-up edge.

    For a split-loop handler with per-attempt failure `f` and `MAX_TRIES = k`, the retry fraction of failures is `r = (1 - f**(k-1)) / (1 - f**k)`; the retry edge carries `f * r` (re-dispatch) and the give-up edge carries `f * (1 - r)` (leak to FAIL). The two always sum to `f`, so the handler row's success-forward weight `1 - f` is untouched. `k = 1` (or a missing depth) is the selection-only case: zero retry, full `f` leaks. Guards cover `f <= 0` (no failure) and `f >= 1` (everything leaks, avoids the `1 - f**k = 0` divide).

    Args:
        f (float): dispatch-weighted per-attempt failure rate at the handler.
        k (int): retry depth (MAX_TRIES); 1 means no retry.

    Returns:
        Tuple[float, float]: `(retry_edge, give_up)`, summing to `f`.
    """
    if f <= 0.0:
        return (0.0, 0.0)
    if k <= 1:
        return (0.0, f)
    if f >= 1.0:
        return (0.0, 1.0)
    _r = (1.0 - f ** (k - 1)) / (1.0 - f ** k)
    _retry_edge = f * _r
    _give_up = f * (1.0 - _r)
    return (_retry_edge, _give_up)


def _retry_depth(retry: Optional[Dict[str, int]], handler: str) -> int:
    """*_retry_depth()* read the retry depth for one handler from the depth map.

    A `None` map or a missing handler key both mean selection-only (depth 1).

    Args:
        retry (Optional[Dict[str, int]]): per-handler retry-depth map keyed by node (e.g. `{"TAS_{4}": 3}`).
        handler (str): the handler node key.

    Returns:
        int: the retry depth for `handler`; 1 when unspecified.
    """
    if retry is None:
        return 1
    return int(retry.get(handler, 1))


def build_routing(node_keys: List[str],
                  eps_by_key: Dict[str, float],
                  weights: Dict[str, Tuple[float, ...]],
                  mas_keys: List[str],
                  as_keys: List[str],
                  ds_keys: List[str],
                  *,
                  retry: Optional[Dict[str, int]] = None) -> np.ndarray:
    """*build_routing()* assemble the loss-network routing matrix (selection + optional retry, FAIL sink).

    Edges:

        - `TAS_{1} -> TAS_{2} = 0.75`, `TAS_{1} -> TAS_{3} = 0.25`.
        - `TAS_{2} -> mas_i = w_M[i]`; `TAS_{3} -> as_i = w_A[i]`.
        - service self-loops `svc_i -> svc_i = eps_i` (error overload, conserves flow), `svc_i -> handler = 1 - eps_i`; for MAS / AS / DS pools.
        - `TAS_{4}` success forwards: `-> TAS_{3} = 0.34 * (1 - f_med)`, `-> ds_i = 0.66 * (1 - f_med) * w_D[i]`; failure mass `f_med` splits via `_retry_split(f_med, k_4)` into `-> TAS_{2} = retry_edge` and `-> FAIL = give_up`.
        - `TAS_{5}` failure mass `f_alm` splits via `_retry_split(f_alm, k_5)` into `-> TAS_{3} = retry_edge` and `-> FAIL = give_up`; success deficit `1 - f_alm` exits.
        - `TAS_{6}` failure mass `f_drg` splits via `_retry_split(f_drg, k_6)`; the retry edge re-dispatches over the DS pool (`-> ds_i += retry_edge * w_D[i]`) and `-> FAIL = give_up`; success deficit exits.
        - `FAIL` row is all-zero (absorbing sink).

    With `retry=None` every `_retry_split` returns `(0, f)`, so the handler give-ups equal the raw pool failure and the matrix is selection-only (identical to the search's original encoding).

    Args:
        node_keys (List[str]): ordered node keys (matches the artifact order in `build_cfg`).
        eps_by_key (Dict[str, float]): catalogue eps per pool service key.
        weights (Dict[str, Tuple[float, ...]]): dispatch weights keyed `M` / `A` / `D`, each aligned with the pool key order.
        mas_keys (List[str]): MAS pool keys (dispatch order).
        as_keys (List[str]): AS pool keys (dispatch order).
        ds_keys (List[str]): DS pool keys (dispatch order).
        retry (Optional[Dict[str, int]]): per-handler retry-depth map keyed by node (`TAS_{4}` / `TAS_{5}` / `TAS_{6}`); missing handlers and `None` mean selection-only.

    Returns:
        np.ndarray: the `n x n` routing matrix.
    """
    _n = len(node_keys)
    _idx: Dict[str, int] = {}
    for _i, _k in enumerate(node_keys):
        _idx[_k] = _i
    _r = np.zeros((_n, _n))

    _w_M = weights["M"]
    _w_A = weights["A"]
    _w_D = weights["D"]
    _fi = _idx[_FAIL_KEY]

    # workflow split at the entry handler
    _r[_idx["TAS_{1}"], _idx["TAS_{2}"]] = _W_ANALYSE
    _r[_idx["TAS_{1}"], _idx["TAS_{3}"]] = _W_ALARM

    # per-pool dispatch: the search varies the weights independently in each pool
    for _j, _mk in enumerate(mas_keys):
        _r[_idx["TAS_{2}"], _idx[_mk]] = _w_M[_j]
    for _j, _ak in enumerate(as_keys):
        _r[_idx["TAS_{3}"], _idx[_ak]] = _w_A[_j]

    # service self-loops = eps: error overload (throughput ~ (1+eps)*lambda); conserves flow
    # (forwards 1 - eps), so it only loads the station and never changes availability
    for _mk in mas_keys:
        _eps = eps_by_key[_mk]
        _r[_idx[_mk], _idx[_mk]] = _eps
        _r[_idx[_mk], _idx["TAS_{4}"]] = 1.0 - _eps
    for _ak in as_keys:
        _eps = eps_by_key[_ak]
        _r[_idx[_ak], _idx[_ak]] = _eps
        _r[_idx[_ak], _idx["TAS_{5}"]] = 1.0 - _eps
    for _dk in ds_keys:
        _eps = eps_by_key[_dk]
        _r[_idx[_dk], _idx[_dk]] = _eps
        _r[_idx[_dk], _idx["TAS_{6}"]] = 1.0 - _eps

    # TAS_{4} medical handler: success forwards then failure split (retry edge + give-up)
    _f_med = pool_eps(_w_M, mas_keys, eps_by_key)
    _r[_idx["TAS_{4}"], _idx["TAS_{3}"]] = _W_MED_TO_ALARM * (1.0 - _f_med)
    for _j, _dk in enumerate(ds_keys):
        _r[_idx["TAS_{4}"], _idx[_dk]] = _W_MED_TO_DRUG * (1.0 - _f_med) * _w_D[_j]
    _med_retry, _med_give = _retry_split(_f_med, _retry_depth(retry, "TAS_{4}"))
    _r[_idx["TAS_{4}"], _idx["TAS_{2}"]] = _med_retry
    _r[_idx["TAS_{4}"], _fi] = _med_give

    # TAS_{5} alarm handler: retry re-dispatches to its alarm dispatcher (TAS_{3})
    _f_alm = pool_eps(_w_A, as_keys, eps_by_key)
    _alm_retry, _alm_give = _retry_split(_f_alm, _retry_depth(retry, "TAS_{5}"))
    _r[_idx["TAS_{5}"], _idx["TAS_{3}"]] = _alm_retry
    _r[_idx["TAS_{5}"], _fi] = _alm_give

    # TAS_{6} drug handler: retry re-dispatches across the DS pool weights
    _f_drg = pool_eps(_w_D, ds_keys, eps_by_key)
    _drg_retry, _drg_give = _retry_split(_f_drg, _retry_depth(retry, "TAS_{6}"))
    for _j, _dk in enumerate(ds_keys):
        _r[_idx["TAS_{6}"], _idx[_dk]] = _r[_idx["TAS_{6}"], _idx[_dk]] + _drg_retry * _w_D[_j]
    _r[_idx["TAS_{6}"], _fi] = _drg_give

    return _r
