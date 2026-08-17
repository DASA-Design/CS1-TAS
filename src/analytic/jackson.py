# -*- coding: utf-8 -*-
"""
Module jackson.py
=================

Jackson open-network solver for the analytic method of the TAS case study.

Public API:
    - `solve_jackson_lams(P, lam_z)` linear solve of `(I - P^T) lam = lam_z`; returns per-node effective arrival rates.
    - `solve_network(cfg)` build a `Queue` per artifact with the Jackson-solved `lam`, call `calculate_metrics()`, return a per-node pandas DataFrame.
    - rho-indexed helpers (`compute_lams_per_artifact`, `compute_rhos_per_artifact`, `invert_rho_to_lam_z`, `build_rho_grid`) drive the inverse direction used by the experiment orchestrator to build a rho-indexed operating-point grid. Jackson is linear in `lam_z`, so the inversion is one division.

The routing matrix in `cfg.routing` is `row = source, col = dest`, transposed before the linear solve.
"""
# native python modules
from __future__ import annotations

from typing import List, Sequence, Tuple

# scientific stack
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd

# local modules
from src.analytic.queues import Queue
from src.io.config import NetCfg


def solve_jackson_lams(P: ArrayLike,
                       lam_z: ArrayLike) -> np.ndarray:
    """*solve_jackson_lams()* per-node effective arrival rates from the Jackson traffic equations `(I - P^T) lam = lam_z`.

    Args:
        P (ArrayLike): `(n, n)` routing probability matrix; `P[i, j]` is the probability of routing from node `i` to node `j`. Coerced via `np.asarray`.
        lam_z (ArrayLike): `(n,)` vector of external arrivals per node. Coerced via `np.asarray`.

    Returns:
        np.ndarray: `(n,)` per-node effective arrival rates.
    """
    _P = np.asarray(P, dtype=float)
    _lz = np.asarray(lam_z, dtype=float)
    _I = np.eye(_P.shape[0])
    _lams = np.linalg.solve(_I - _P.T, _lz)
    return _lams


def solve_network(cfg: NetCfg) -> pd.DataFrame:
    """*solve_network()* solves the open Jackson network for the given scenario and returns per-node metrics as a pandas DataFrame.

    For each of the n-artifacts in `cfg`, creates a `Queue` with its declared `type` / $\\mu$ / $c$ / $K$, injects the Jackson-solved $\\lambda$, and calls `calculate_metrics()`. Stops with a clear error if any node comes out unstable ($\\rho \\geq 1$).

    Args:
        cfg (NetCfg): resolved network configuration for one (profile, scenario) pair. Provides artifacts, external arrivals, and the routing matrix.

    Raises:
        ValueError: If one or more nodes are unstable under the Jackson-solved arrival rates.

    Returns:
        pd.DataFrame: one row per artifact with columns `node`, `key`, `name`, `type`, `lambda`, `mu`, `c`, `K`, `rho`, `L`, `Lq`, `W`, `Wq`.
    """
    _lams = solve_jackson_lams(cfg.routing, cfg.build_lam_z_vec())
    _rows: List[dict] = []
    _unstable: List[tuple] = []
    for _i, _a in enumerate(cfg.artifacts):
        _q = Queue(
            model=_a.type_,
            lamb=float(_lams[_i]),
            mu=_a.mu,
            c_max=_a.c,
            K_max=_a.K,
        )
        _q.calculate_metrics()
        if _q.rho >= 1.0:
            _unstable.append((_a.key, _q.rho))
        _rows.append({
            "node": _i,
            "key": _a.key,
            "name": _a.name,
            "type": _a.type_,
            "lambda": _lams[_i],
            "mu": _a.mu,
            "c": _a.c,
            "K": _a.K,
            "rho": _q.rho,
            "L": _q.avg_len,
            "Lq": _q.avg_len_q,
            "W": _q.avg_wait,
            "Wq": _q.avg_wait_q,
        })
    if _unstable:
        _details = ", ".join(f"{_k}: rho={_r:.4f}" for _k, _r in _unstable)
        _msg = f"unstable nodes in scenario {cfg.scenario!r}: {_details}"
        raise ValueError(_msg)
    return pd.DataFrame(_rows)


# ---- rho-indexed helpers ----


def compute_lams_per_artifact(cfg: NetCfg,
                              lam_z: float) -> np.ndarray:
    """*compute_lams_per_artifact()* per-artifact effective arrival rate at the given scalar entry rate.

    Scales the profile's external-arrival vector by `lam_z` so callers can sweep the entry rate without editing the profile, then forward-solves via `solve_jackson_lams`.

    Args:
        cfg (NetCfg): resolved profile + scenario.
        lam_z (float): total external arrival rate at the network entry.

    Returns:
        np.ndarray: per-artifact lambda_i in artifact-declaration order.
    """
    _lam_z_vec = np.asarray(cfg.build_lam_z_vec(), dtype=float)
    _total = float(_lam_z_vec.sum())
    if _total <= 0:
        # no external arrivals declared; first artifact is the entry
        _entry_vec = np.zeros_like(_lam_z_vec)
        _entry_vec[0] = float(lam_z)
    else:
        _entry_vec = _lam_z_vec * (float(lam_z) / _total)
    _ans = solve_jackson_lams(np.asarray(cfg.routing, dtype=float),
                              _entry_vec)
    return _ans


def compute_rhos_per_artifact(cfg: NetCfg,
                              lam_z: float) -> np.ndarray:
    """*compute_rhos_per_artifact()* per-artifact utilisation `rho_i = lam_i / (c_i * mu_i)`.

    Returns `+inf` for any artifact with zero capacity (not expected in well-formed profiles).

    Args:
        cfg (NetCfg): resolved profile + scenario.
        lam_z (float): total external arrival rate at the network entry.

    Returns:
        np.ndarray: per-artifact rho in artifact-declaration order.
    """
    _lams = compute_lams_per_artifact(cfg, lam_z)
    _cs = np.array([float(_a.c) for _a in cfg.artifacts])
    _mus = np.array([float(_a.mu) for _a in cfg.artifacts])
    _cap = _cs * _mus
    _has_cap = _cap > 0
    _rhos = np.full_like(_cap, np.inf)
    _rhos[_has_cap] = _lams[_has_cap] / _cap[_has_cap]
    return _rhos


def invert_rho_to_lam_z(cfg: NetCfg,
                        rho_target: float,
                        *,
                        probe_lam_z: float = 1.0
                        ) -> Tuple[float, int, float]:
    """*invert_rho_to_lam_z()* entry `lam_z` that makes the bottleneck artifact hit `rho_target`.

    Jackson is linear in `lam_z`; one probe at any positive rate identifies the bottleneck (artifact with the highest `rho` per unit `lam_z`) and gives the scaling factor.

    Args:
        cfg (NetCfg): resolved profile + scenario.
        rho_target (float): desired bottleneck utilisation in `(0, 1)`.
        probe_lam_z (float): positive probe rate; only used to identify the per-unit bottleneck.

    Raises:
        ValueError: when `rho_target` is outside `(0, 1)`, or no artifact has positive capacity, or the bottleneck rho-per-unit is non-positive.

    Returns:
        Tuple[float, int, float]: `(lam_z, bottleneck_index, rho_per_unit_lam_z)`.
    """
    if not 0.0 < rho_target < 1.0:
        _msg = f"rho_target must be in (0, 1), got {rho_target}"
        raise ValueError(_msg)
    _rhos = compute_rhos_per_artifact(cfg, probe_lam_z)
    if not np.all(np.isfinite(_rhos)):
        _msg = "at least one artifact has non-finite rho; check mu and c"
        raise ValueError(_msg)
    _bottleneck = int(np.argmax(_rhos))
    _per_unit = float(_rhos[_bottleneck]) / float(probe_lam_z)
    if _per_unit <= 0:
        _msg = "bottleneck rho per lam_z is non-positive; check routing"
        raise ValueError(_msg)
    _lam_z = float(rho_target) / _per_unit
    return _lam_z, _bottleneck, _per_unit


def build_rho_grid(cfg: NetCfg,
                   rho_grid: Sequence[float]
                   ) -> List[Tuple[float, float, int]]:
    """*build_rho_grid()* map a rho-indexed operating-point grid to its `lam_z` values.

    Args:
        cfg (NetCfg): resolved profile + scenario.
        rho_grid (Sequence[float]): target bottleneck utilisations.

    Returns:
        List[Tuple[float, float, int]]: per-target `(rho, lam_z, bottleneck_index)`.
    """
    _out: List[Tuple[float, float, int]] = []
    for _rho in rho_grid:
        _lam_z, _bottle, _ = invert_rho_to_lam_z(cfg, float(_rho))
        _out.append((float(_rho), float(_lam_z), int(_bottle)))
    return _out
