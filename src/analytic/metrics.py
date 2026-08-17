# -*- coding: utf-8 -*-
"""
Module metrics.py
=================

Network-wide aggregates and R1 / R2 validation for the analytic method of the TAS case study.

Two layers, kept separate so the aggregation math can be unit-tested independently of the validation thresholds:

    - `aggregate_net(nodes, lam_z, routing=...)` reduces the per-node DataFrame into a single network-wide row.
    - `check_reqs(nodes, ...)` evaluates the R1 / R2 targets against the thresholds declared in a reference file (default: `data/reference/baseline.json`):

        R1: failure rate <= 0.01 (1%) (Availability)
        R2: response time <= 26 ms (Performance)

Thresholds are NOT hardcoded; they live in `data/reference/<name>.json`.

**Loss-network (flow) availability.** Under the analytic model, failures are routing leaks that
terminate at an absorbing FAIL sink node (key `FAIL_{1}`). When the caller passes `routing`, the
end-to-end figures are read directly from flow:

    eps_e2e = lambda_FAIL / lambda_z              (availability: failed fraction)
    chi_out = lambda_z - lambda_FAIL              (successful-completion rate)
    W_e2e   = L_net / chi_out                     (per-request end-to-end response time)

The FAIL sink (a synthetic absorbing node with huge mu, L ~ 0) is excluded from the real-network
aggregates so it does not skew `avg_mu` / `avg_rho` / `W_net`.

**Legacy exposure path (kept for not-yet-propagated callers).** When the caller passes `eps_vec`
instead of `routing`, the older exposure estimate is used: `eps_e2e = 1 - prod((1-eps_j)^V_j)`,
`chi_out = lam_z * (1 - eps_e2e)`. Stochastic / dimensional / search still use this path until
they are migrated to the flow model; it is FAIL-masked so the added sink node does not break them.
"""
# native python modules
from __future__ import annotations

from typing import Any, Dict, List, Optional

# scientific stack
import numpy as np
import pandas as pd

# local modules
from src.io import load_reference

# absorbing failure-sink node key (loss-network model)
_FAIL_KEY = "FAIL_{1}"


def _fail_mask(nodes: pd.DataFrame) -> np.ndarray:
    """*_fail_mask()* boolean mask (positional) of rows that are the FAIL sink; all-False when absent."""
    if "key" in nodes.columns:
        return (nodes["key"] == _FAIL_KEY).to_numpy()
    return np.zeros(len(nodes), dtype=bool)


def aggregate_net(nodes: pd.DataFrame,
                  lam_z: Optional[float] = None,
                  eps_vec: Optional[np.ndarray] = None,
                  routing: Optional[np.ndarray] = None) -> pd.DataFrame:
    """*aggregate_net()* reduces the per-node metrics frame into a single network-wide row.

    `W_net` is the throughput-weighted mean of per-node `W` (per-visit average). `W_e2e` is the per-request end-to-end response time the R2 threshold gauges. End-to-end availability is computed from flow when `routing` is supplied (loss-network: `eps_e2e = lambda_FAIL/lambda_z`), or from the legacy exposure estimate when `eps_vec` is supplied. Without either, the end-to-end columns are NaN. The FAIL sink node is excluded from the real-network aggregates.

    Args:
        nodes (pd.DataFrame): per-node metrics as produced by `solve_network()`. Required columns: `lambda`, `mu`, `rho`, `L`, `Lq`, `W`, `Wq` (and `key` to locate the FAIL sink).
        lam_z (Optional[float]): external arrival rate at TAS_{1} (`cfg.build_lam_z_vec()[0]`).
        eps_vec (Optional[np.ndarray]): per-node failure rate (legacy exposure path), aligned with `nodes`.
        routing (Optional[np.ndarray]): `(n, n)` routing matrix aligned with `nodes` (flow path); availability is read from the FAIL sink's solved arrival rate.

    Returns:
        pd.DataFrame: single-row frame with `nodes`, `total_throughput`, `avg_mu`, `avg_rho`, `max_rho`, `L_net`, `Lq_net`, `W_net`, `Wq_net`, `eps_e2e`, `chi_out`, `W_e2e`, `Wq_e2e`.
    """
    _is_fail = _fail_mask(nodes)
    _keep = ~_is_fail
    # real-network aggregates exclude the synthetic FAIL sink
    _lams = nodes["lambda"].to_numpy(dtype=float)[_keep]
    _total_lam = float(_lams.sum())
    _l_net = float(nodes["L"].to_numpy(dtype=float)[_keep].sum())
    _lq_net = float(nodes["Lq"].to_numpy(dtype=float)[_keep].sum())
    if _total_lam > 0:
        _w_arr = nodes["W"].to_numpy(dtype=float)[_keep]
        _wq_arr = nodes["Wq"].to_numpy(dtype=float)[_keep]
        _w_net = float(np.sum(_w_arr * _lams) / _total_lam)
        _wq_net = float(np.sum(_wq_arr * _lams) / _total_lam)
    else:
        _w_net = 0.0
        _wq_net = 0.0
    _mu_arr = nodes["mu"].to_numpy(dtype=float)[_keep]
    _rho_arr = nodes["rho"].to_numpy(dtype=float)[_keep]
    # end-to-end figures
    _eps_e2e = float("nan")
    _chi_out = float("nan")
    _w_e2e = float("nan")
    _wq_e2e = float("nan")
    if lam_z is not None and lam_z > 0 and routing is not None and _is_fail.any():
        # loss-network: availability read from the absorbing FAIL sink's flow
        _lam_fail = float(nodes["lambda"].to_numpy(dtype=float)[_is_fail].sum())
        _chi_out = float(lam_z) - _lam_fail
        _eps_e2e = _lam_fail / float(lam_z)
        if _chi_out > 0:
            _w_e2e = _l_net / _chi_out
            _wq_e2e = _lq_net / _chi_out
    elif lam_z is not None and lam_z > 0 and eps_vec is not None:
        # legacy exposure path (not-yet-propagated callers); FAIL-masked
        _eps_arr = np.asarray(eps_vec, dtype=float)[_keep]
        _v_vec = _lams / float(lam_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            _log_survival = np.where(_eps_arr < 1.0, _v_vec * np.log1p(-_eps_arr), -np.inf)
        _survival = float(np.exp(np.sum(_log_survival)))
        _eps_e2e = 1.0 - _survival
        _chi_out = float(lam_z) * _survival
        if _chi_out > 0:
            _w_e2e = _l_net / _chi_out
            _wq_e2e = _lq_net / _chi_out
    _row = {
        "nodes": int(_keep.sum()),
        "total_throughput": _total_lam,
        "avg_mu": float(_mu_arr.mean()) if _mu_arr.size else 0.0,
        "avg_rho": float(_rho_arr.mean()) if _rho_arr.size else 0.0,
        "max_rho": float(_rho_arr.max()) if _rho_arr.size else 0.0,
        "L_net": _l_net,
        "Lq_net": _lq_net,
        "W_net": _w_net,
        "Wq_net": _wq_net,
        "eps_e2e": _eps_e2e,
        "chi_out": _chi_out,
        "W_e2e": _w_e2e,
        "Wq_e2e": _wq_e2e,
    }
    return pd.DataFrame([_row])


def _node_label(nodes: pd.DataFrame, idx: int) -> str:
    """*_node_label()* returns the human-readable node key (e.g. `MAS_{3}`) at row `idx`, falling back to a string index if no `key` / `node` column is present."""
    for _col in ("key", "node", "name"):
        if _col in nodes.columns:
            return str(nodes[_col].iloc[idx])
    return f"row[{idx}]"


def _r1_flow_contributions(nodes: pd.DataFrame,
                           lam_z: float,
                           routing: np.ndarray,
                           top_k: int = 5) -> List[Dict[str, Any]]:
    """*_r1_flow_contributions()* top-`top_k` per-node contributions to the end-to-end failure rate, from flow.

    Each node j routes `lambda_j * routing[j, FAIL]` jobs to the FAIL sink; its contribution is that flow as a share of `lam_z`. Shares are additive and sum to `eps_e2e = lambda_FAIL/lam_z`. Returned sorted descending; zero rows omitted.
    """
    _is_fail = _fail_mask(nodes)
    if lam_z <= 0 or not _is_fail.any():
        return []
    _fail_idx = int(np.argmax(_is_fail))
    _R = np.asarray(routing, dtype=float)
    _lams = nodes["lambda"].to_numpy(dtype=float)
    _to_fail = _R[:, _fail_idx]
    _leak = _lams * _to_fail
    _contrib = _leak / float(lam_z)
    _order = np.argsort(-_contrib)
    _out: List[Dict[str, Any]] = []
    for _i in _order[:top_k]:
        _val = float(_contrib[_i])
        if _val <= 0.0:
            continue
        _out.append({
            "node": _node_label(nodes, int(_i)),
            "epsilon": float(_to_fail[_i]),
            "V": float(_lams[_i] / float(lam_z)),
            "contribution": _val,
        })
    return _out


def _r1_node_contributions(nodes: pd.DataFrame,
                           lam_z: float,
                           eps_vec: np.ndarray,
                           top_k: int = 5) -> List[Dict[str, Any]]:
    """*_r1_node_contributions()* legacy (exposure) top-`top_k` per-node contributions to the end-to-end failure rate.

    Each node j contributes `1 - (1 - eps_j) ** V_j` as its standalone end-to-end failure probability. Returned sorted descending; zero rows omitted. Used by the not-yet-propagated callers.
    """
    _lams = nodes["lambda"].to_numpy(dtype=float)
    _eps = np.asarray(eps_vec, dtype=float)
    if lam_z <= 0:
        return []
    _v_vec = _lams / float(lam_z)
    with np.errstate(divide="ignore", invalid="ignore"):
        _node_fail = np.where(_eps < 1.0, 1.0 - (1.0 - _eps) ** _v_vec, 1.0)
    _order = np.argsort(-_node_fail)
    _out: List[Dict[str, Any]] = []
    for _i in _order[:top_k]:
        _val = float(_node_fail[_i])
        if _val <= 0.0:
            continue
        _out.append({
            "node": _node_label(nodes, int(_i)),
            "epsilon": float(_eps[_i]),
            "V": float(_v_vec[_i]),
            "contribution": _val,
        })
    return _out


def _r2_node_contributions(nodes: pd.DataFrame,
                           lam_z: float,
                           top_k: int = 5) -> List[Dict[str, Any]]:
    """*_r2_node_contributions()* returns the top-`top_k` per-node contributions to the end-to-end response time.

    Each node j contributes `L_j / chi_out` to `W_e2e`; relative shares are `L_j / L_net` (the FAIL sink, with L ~ 0, is excluded). Returned sorted descending by contribution.
    """
    _is_fail = _fail_mask(nodes)
    _keep = ~_is_fail
    _lams = nodes["lambda"].to_numpy(dtype=float)
    _l_vec = nodes["L"].to_numpy(dtype=float)
    _w_vec = nodes["W"].to_numpy(dtype=float)
    if lam_z <= 0:
        return []
    _l_net = float(_l_vec[_keep].sum())
    if _l_net <= 0:
        return []
    _share = np.where(_keep, _l_vec / _l_net, 0.0)
    _order = np.argsort(-_share)
    _out: List[Dict[str, Any]] = []
    for _i in _order[:top_k]:
        _val = float(_share[_i])
        if _val <= 0.0:
            continue
        _out.append({
            "node": _node_label(nodes, int(_i)),
            "L": float(_l_vec[_i]),
            "W": float(_w_vec[_i]),
            "V": float(_lams[_i] / float(lam_z)),
            "share": _val,
        })
    return _out


def check_reqs(nodes: pd.DataFrame,
               lam_z: Optional[float] = None,
               eps_vec: Optional[np.ndarray] = None,
               routing: Optional[np.ndarray] = None,
               failure_rate: Optional[float] = None,
               response_time: Optional[float] = None,
               reference: str = "baseline") -> Dict[str, Dict[str, Any]]:
    """*check_reqs()* evaluates the R1 / R2 targets against the reference thresholds, using end-to-end values derived from the per-node frame.

    End-to-end values come from `aggregate_net(nodes, lam_z, eps_vec, routing)`: the flow path (`routing`, loss-network FAIL sink) for the analytic method, or the legacy exposure path (`eps_vec`) for not-yet-propagated callers. Either `routing` or `eps_vec` (with `lam_z`) is required unless explicit `failure_rate` + `response_time` overrides are supplied.

    Returns:
        Dict[str, Dict[str, Any]]: verdicts keyed by `R1`, `R2`, each with `metric`, `value`, `threshold`, `operator`, `units`, `pass`, `contributions`, `notes`.
    """
    _ref = load_reference(reference)
    _reqs = _ref["requirements"]

    _need_aggregate = failure_rate is None or response_time is None
    _have_inputs = lam_z is not None and (routing is not None or eps_vec is not None)
    if _need_aggregate and not _have_inputs:
        _msg = ("check_reqs needs lam_z plus routing (flow) or eps_vec (legacy) to compute "
                "end-to-end R1/R2; or supply explicit failure_rate + response_time overrides.")
        raise ValueError(_msg)

    _agg = None
    if _need_aggregate:
        _agg = aggregate_net(nodes, lam_z=lam_z, eps_vec=eps_vec, routing=routing)

    _fail_rate = failure_rate
    if _fail_rate is None:
        _fail_rate = float(_agg["eps_e2e"].iloc[0])

    _resp = response_time
    if _resp is None:
        _resp = float(_agg["W_e2e"].iloc[0])

    _r1_pass = bool(_fail_rate <= _reqs["R1"]["threshold"])
    _r2_pass = bool(_resp <= _reqs["R2"]["threshold"])

    # per-node contribution breakdowns (verdict evidence)
    _r1_contribs: List[Dict[str, Any]] = []
    _r2_contribs: List[Dict[str, Any]] = []
    if _need_aggregate and lam_z is not None:
        if routing is not None:
            _r1_contribs = _r1_flow_contributions(nodes, lam_z, routing)
        elif eps_vec is not None:
            _r1_contribs = _r1_node_contributions(nodes, lam_z, eps_vec)
        _r2_contribs = _r2_node_contributions(nodes, lam_z)

    _measured = {
        "R1": (_fail_rate, _r1_pass, _r1_contribs),
        "R2": (_resp, _r2_pass, _r2_contribs),
    }
    _verdicts: Dict[str, Dict[str, Any]] = {}
    for _k, (_val, _ok, _contribs) in _measured.items():
        _spec = _reqs[_k]
        _verdicts[_k] = {
            "metric": _spec["metric"],
            "value": _val,
            "threshold": _spec["threshold"],
            "operator": _spec["operator"],
            "units": _spec["units"],
            "pass": _ok,
            "contributions": _contribs,
            "notes": _spec["notes"],
        }
    return _verdicts
