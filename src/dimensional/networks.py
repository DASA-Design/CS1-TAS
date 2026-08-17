# -*- coding: utf-8 -*-
"""
Module networks.py
==================

Configuration-sweep helpers for the dimensional method's yoly notebook.

Sweeps answering different questions:

    - `sweep_artifacts(cfg, sweep_grid)` is an INDEPENDENT per-artifact sweep. Each node's lambda is the seeded analytic value; no routing propagation. Answers *"what is the design space of THIS node in isolation?"*. Useful for per-node introspection but not architecture-level analysis.
    - `sweep_arch(cfg, sweep_grid, tag="TAS")` is a JACKSON-PROPAGATED whole-network sweep. For each `(mu_factor, c, K)` combo, overrides every node's design knobs uniformly, binary-searches for the max external arrival factor that keeps the whole network stable, then linspaces from `lambda_factor_min * f_max` to `f_max` in `lambda_steps` increments. At every step `solve_jackson_lams(P, f * lambda_z)` redistributes external arrivals to per-node rates before the M/M/c/K solve; the first node to saturate stops that combo's ramp, since the whole network's instability is dominated by its busiest node.
    - `sweep_artifact(key, vars_block, sweep_grid, ...)` is the underlying single-artifact helper used by `sweep_artifacts`.

The three sweep functions return dicts shape-compatible with the `src.view.plot_yoly_arts_charts` family; `_read_setpoint` is a one-line accessor returning a float. Coefficients are derived via closed-form directly from the M/M/c/K solver (no PyDASA round-trip):

    - theta = L / K (Occupancy)
    - sigma = lambda * W / K (Stall, queueing share of capacity)
    - eta = chi * K / (mu * c) (Effective-yield)
    - phi = (L * delta) / (K * delta) (Memory-use; explicit byte arithmetic, reduces to L/K under constant delta)

The sweep is deterministic (not Monte Carlo); for stochastic sampling go through `src.methods.dimensional` plus the dimensional notebook.
"""
# native python modules
from __future__ import annotations

# data types
from typing import Any, Dict, Iterable, List, Optional

# scientific stack
import numpy as np

# local modules
from src.analytic.jackson import solve_jackson_lams
from src.analytic.queues import Queue
from src.dimensional.noise import add_white_noise
from src.io import NetCfg


# default utilisation cap; points with rho at or above this are dropped
_UTIL_THRESH_DEFAULT = 0.95

# synthetic absorbing failure-sink node; stays in the routing (Jackson lambda propagation carries
# the give-up leaks to it) but is excluded from the per-artifact sweep output, since it has no
# sweepable service and its near-zero queue length would trip the stability guard
_FAIL_KEY = "FAIL_{1}"


def read_setpoint(vars_block: Dict[str, Dict[str, Any]],
                  prefix: str,
                  artifact_key: str) -> float:
    """*read_setpoint()* read the `_setpoint` of a variable identified by `<prefix>_{<artifact_key>}`.

    Args:
        vars_block (Dict[str, Dict[str, Any]]): per-artifact `vars` dict.
        prefix (str): LaTeX symbol prefix (e.g. `"\\lambda"`, `"\\mu"`, `"K"`).
        artifact_key (str): artifact identifier in LaTeX subscript form.

    Raises:
        KeyError: when no matching variable is present.

    Returns:
        float: setpoint value, cast to float.
    """
    _sym = f"{prefix}_{{{artifact_key}}}"
    if _sym not in vars_block:
        raise KeyError(f"missing {_sym} on artifact vars block")
    return float(vars_block[_sym]["_setpoint"])


def sweep_artifact(artifact_key: str,
                   vars_block: Dict[str, Dict[str, Any]],
                   sweep_grid: Dict[str, Any],
                   *,
                   util_threshold: float = _UTIL_THRESH_DEFAULT,
                   model: str = "M/M/c/K",
                   noise: Any = None,
                   seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """*sweep_artifact()* sweeps the `(mu, c, K)` design grid AND ramps `lambda` for each combo, returning trace-shaped coefficient curves.

    For each cartesian combination of `(mu_factor, c, K)`:

    1. `mu = mu_factor * mu_setpoint`.
    2. Pick a maximum `lambda` that keeps the queue stable: `lambda_max = util_threshold * mu * c` (the textbook cap for M/M/c/K).
    3. Linearly ramp `lambda` from `lambda_factor_min * lambda_max` to `lambda_max` in `lambda_steps` increments.
    4. Solve M/M/c/K at every step; derive `theta`, `sigma`, `eta`, `phi` directly.
    5. Drop steps where the solver's reported `rho` exceeds the cap (numerical safety).

    The result is one TRACE per `(mu, c, K)` combo (not a single dot), so the yoly cloud is a set of curves through coefficient space.

    Args:
        artifact_key (str): artifact identifier in LaTeX subscript form (e.g. `"TAS_{1}"`).
        vars_block (Dict[str, Dict[str, Any]]): the artifact's `vars` dict from the profile JSON. Must carry `\\mu` and `\\epsilon` setpoints under the artifact subscript; `d` is optional (defaults to 1 kB so the phi byte arithmetic still cancels).
        sweep_grid (Dict[str, Any]): grid declared in `data/config/method/dimensional.json::sweep_grid`. Keys: `mu_factor`, `c`, `K`, `lambda_steps`, `lambda_factor_min`.
        util_threshold (float): drop points with `rho >= util_threshold`. Defaults to `0.95`.
        model (str): queue model passed to the `Queue` factory. Defaults to `"M/M/c/K"`.
        noise (Any): when set (a float like 0.05, or a 3-array `[lambda, mu, epsilon]`), disturb the independent inputs with seeded white noise at each trace point so the coefficient cloud carries that uncertainty; the recorded design knobs (mu / c / K / lambda) stay at their nominal values so trajectory grouping is preserved. Defaults to None.
        seed (Optional[int]): seed for the noise generator; ignored when `noise` is None.

    Raises:
        KeyError: If any required variable is missing on the artifact block.

    Returns:
        Dict[str, np.ndarray]: arrays keyed by full LaTeX symbol. Same length across keys; one entry per stable trace point.
    """
    _mu_base = read_setpoint(vars_block,
                             "\\mu",
                             artifact_key)
    _eps = read_setpoint(vars_block,
                         "\\epsilon",
                         artifact_key)
    _rng = np.random.default_rng(seed)
    # per-request payload (kB); 1 kB fallback so M_act/M_buf reduces to L/K cleanly
    _d_sym = f"d_{{{artifact_key}}}"
    if _d_sym in vars_block:
        _delta_kB = float(vars_block[_d_sym].get("_setpoint", 1.0))
    else:
        _delta_kB = 1.0
    _n_steps = int(sweep_grid.get("lambda_steps", 30))
    _lam_min_frac = float(sweep_grid.get("lambda_factor_min", 0.05))
    _theta_lt: List[float] = []
    _sigma_lt: List[float] = []
    _eta_lt: List[float] = []
    _phi_lt: List[float] = []
    _c_lt: List[float] = []
    _mu_lt: List[float] = []
    _K_lt: List[float] = []
    _lam_lt: List[float] = []

    # walk the cartesian grid; the order is (mu, c, K, lambda_step) for repeatability
    for _mf in sweep_grid.get("mu_factor", [1.0]):
        _mu = float(_mu_base) * float(_mf)
        for _c in sweep_grid.get("c", [1]):
            _c_int = int(_c)
            for _K in sweep_grid.get("K", [10]):
                _K_int = int(_K)
                # K must be >= c for M/M/c/K to be defined
                if _K_int < _c_int:
                    continue

                # textbook stability cap: lambda_max = util_threshold * mu * c
                _lam_max = util_threshold * _mu * _c_int
                _lam_min = _lam_max * _lam_min_frac
                _lambdas = np.linspace(_lam_min, _lam_max, _n_steps)
                for _lam in _lambdas:
                    # disturb the independent inputs (lambda, mu, epsilon); the recorded knobs stay nominal
                    if noise is not None:
                        _lam_use, _mu_use, _eps_use = (
                            float(_v) for _v in add_white_noise(
                                [float(_lam), _mu, _eps], noise, rng=_rng))
                    else:
                        _lam_use, _mu_use, _eps_use = float(_lam), _mu, _eps
                    try:
                        _q = Queue(model=model,
                                   lamb=_lam_use,
                                   mu=_mu_use,
                                   c_max=_c_int,
                                   K_max=_K_int)
                        _q.calculate_metrics()
                    except (OverflowError, ValueError, ZeroDivisionError):
                        # M/M/c/K closed-form overflow at large K + c (factorial blow-up); drop the point so genuine bugs still surface as uncaught
                        continue

                    if _q.rho >= util_threshold:
                        continue

                    _L = float(_q.avg_len)
                    _W = float(_q.avg_wait)
                    if _L <= 0:
                        continue

                    # chi = (1 + eps) * lambda: station throughput carries reprocessed error load, so eta reflects error-handling work (availability stays chi_out = (1 - eps_e2e) * lambda_z, computed in analytic.metrics)
                    _chi = _lam_use * (1.0 + _eps_use)
                    _theta = _L / _K_int
                    _sigma = _lam_use * _W / _K_int
                    _eta = _chi * _K_int / (_mu_use * _c_int)
                    # phi = M_act/M_buf with explicit delta; reduces to L/K under constant payload (sanity-check)
                    _m_act = _L * _delta_kB
                    _m_buf = _K_int * _delta_kB
                    if _m_buf > 0:
                        _phi = _m_act / _m_buf
                    else:
                        _phi = float("nan")

                    _theta_lt.append(_theta)
                    _sigma_lt.append(_sigma)
                    _eta_lt.append(_eta)
                    _phi_lt.append(_phi)
                    _c_lt.append(float(_c_int))
                    _mu_lt.append(_mu)
                    _K_lt.append(float(_K_int))
                    _lam_lt.append(float(_lam))
    return {
        f"\\theta_{{{artifact_key}}}": np.asarray(_theta_lt, dtype=float),
        f"\\sigma_{{{artifact_key}}}": np.asarray(_sigma_lt, dtype=float),
        f"\\eta_{{{artifact_key}}}": np.asarray(_eta_lt, dtype=float),
        f"\\phi_{{{artifact_key}}}": np.asarray(_phi_lt, dtype=float),
        f"c_{{{artifact_key}}}": np.asarray(_c_lt, dtype=float),
        f"\\mu_{{{artifact_key}}}": np.asarray(_mu_lt, dtype=float),
        f"K_{{{artifact_key}}}": np.asarray(_K_lt, dtype=float),
        f"\\lambda_{{{artifact_key}}}": np.asarray(_lam_lt, dtype=float),
    }


def sweep_artifacts(cfg: NetCfg,
                    sweep_grid: Dict[str, List[float]],
                    *,
                    util_threshold: float = _UTIL_THRESH_DEFAULT,
                    model: str = "M/M/c/K",
                    artifact_filter: Iterable[str] = ()
                    ) -> Dict[str, Dict[str, np.ndarray]]:
    """*sweep_artifacts()* runs `sweep_artifact` INDEPENDENTLY for every node of a resolved `NetCfg` and returns the nested cloud dict consumed by `src.view.plot_yoly_arts_charts` and friends.

    Each artifact is swept as if it were an isolated M/M/c/K queue; routing is NOT propagated. For the architecture-level view (where changing one node's design affects everyone else through routing) use `sweep_arch`.

    Args:
        cfg (NetCfg): resolved network configuration for one (profile, scenario) pair.
        sweep_grid (Dict[str, List[float]]): grid declared in `data/config/method/dimensional.json::sweep_grid`.
        util_threshold (float): per-artifact stability cap. Defaults to `0.95`.
        model (str): queue model passed to the `Queue` factory. Defaults to `"M/M/c/K"`.
        artifact_filter (Iterable[str]): optional subset of artifact keys to include. Empty `()` means all.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: nested `{artifact_key: per_artifact_sweep}`. Per-artifact dict matches `sweep_artifact`'s return shape.
    """
    # pre-compute the filter set so the per-loop check is O(1)
    _filter = set(artifact_filter)

    _out: Dict[str, Dict[str, np.ndarray]] = {}
    for _a in cfg.artifacts:
        if _filter and _a.key not in _filter:
            continue

        _out[_a.key] = sweep_artifact(_a.key,
                                      _a.vars,
                                      sweep_grid,
                                      util_threshold=util_threshold,
                                      model=model)
    return _out


def _find_max_stable_lam_factor(cfg: NetCfg,
                                mu_vec: np.ndarray,
                                c_int: int,
                                util_threshold: float,
                                iters: int = 40) -> float:
    """*_find_max_stable_lam_factor()* binary-searches the max scalar on `cfg.build_lam_z_vec()` that keeps every Jackson-propagated rho below `util_threshold`.

    Args:
        cfg (NetCfg): resolved network configuration.
        mu_vec (np.ndarray): per-node service rates AFTER applying `mu_factor`.
        c_int (int): uniform override for the server count across every node.
        util_threshold (float): stability cap.
        iters (int): binary-search iterations. Defaults to `40` (convergence to ~1e-12 relative).

    Returns:
        float: the greatest factor `f` such that at `f * cfg.build_lam_z_vec()` every node has `rho < util_threshold`. Returns `0.0` if the stable region is empty at this `(mu_vec, c_int)`.
    """
    _P = cfg.routing
    _lz = cfg.build_lam_z_vec()

    # initial bracket; hi is large enough that saturation is hit at any reasonable mu/c
    _lo = 0.0
    _hi = 100.0

    for _ in range(iters):
        _mid = (_lo + _hi) / 2.0
        _lams = solve_jackson_lams(_P, _mid * _lz)
        _rhos = _lams / (mu_vec * float(c_int))
        if np.any(_rhos >= util_threshold):
            _hi = _mid
        else:
            _lo = _mid

    return float(_lo)


def sweep_arch(cfg: NetCfg,
               sweep_grid: Dict[str, Any],
               *,
               util_threshold: Optional[float] = None,
               model: str = "M/M/c/K",
               noise: Any = None,
               seed: Optional[int] = None
               ) -> Dict[str, Dict[str, np.ndarray]]:
    """*sweep_arch()* Jackson-propagated whole-network sweep; every node's design knobs are overridden uniformly per the grid, routing propagates arrivals, and the external lambda ramps from near zero up to the first-node saturation point.

    For each `(mu_factor, c, K)` combo in `sweep_grid`:

    1. Scale every node's mu by `mu_factor`.
    2. Binary-search for the maximum external-arrival scale `f_max` that keeps every node below `util_threshold` after Jackson propagation.
    3. Linspace `lambda_steps` factors from `lambda_factor_min * f_max` to `0.999 * f_max`.
    4. At each factor: `solve_jackson_lams(P, f * lambda_z)` redistributes arrivals; solve every node's M/M/c/K with `(lambda_i, mu_factor * mu_i, c, K)`; derive theta/sigma/eta/phi per node.
    5. Return per-artifact arrays (one entry per stable sweep point). Arrays are aligned across artifacts because every point is a whole-network solve.

    Args:
        cfg (NetCfg): resolved network configuration.
        sweep_grid (Dict[str, Any]): grid from `data/config/method/dimensional.json::sweep_grid`. Required keys: `mu_factor`, `c`, `K`, `lambda_steps`, `lambda_factor_min`. Optional key: `util_threshold` (used as the stability cap when the `util_threshold` kwarg is `None`).
        util_threshold (Optional[float]): stability cap; overrides the grid value when supplied.
        model (str): queue model. Defaults to `"M/M/c/K"`.
        noise (Any): when set (a float like 0.05, or a 3-array `[lambda, mu, epsilon]`), disturb the independent inputs with seeded white noise before each node solve so the coefficient cloud carries that uncertainty; the recorded design knobs (mu / c / K / lambda) stay at their nominal values so trajectory grouping is preserved. Defaults to None.
        seed (Optional[int]): seed for the noise generator; ignored when `noise` is None.

    Returns:
        Dict[str, Dict[str, np.ndarray]]: nested `{artifact_key: per_artifact_sweep}` with the same symbol keys as `sweep_artifact`. Arrays across artifacts are aligned (row i is the same whole-network sweep point) so `aggregate_sweep_to_arch` can collapse them into architecture-level arrays.
    """
    _rng = np.random.default_rng(seed)
    if util_threshold is not None:
        _util = util_threshold
    else:
        _util = float(sweep_grid.get("util_threshold", _UTIL_THRESH_DEFAULT))
    _mu_factors = sweep_grid.get("mu_factor", [1.0])
    _c_vals = sweep_grid.get("c", [1])
    _K_vals = sweep_grid.get("K", [10])
    _n_steps = int(sweep_grid.get("lambda_steps", 10))
    _lam_min_frac = float(sweep_grid.get("lambda_factor_min", 0.05))
    # pre-read per-node mu setpoints + entry lambda_z; reused across every (mu_factor, c, K) combo
    _arts = cfg.artifacts
    _mu_base = np.array([float(_a.mu) for _a in _arts], dtype=float)
    _lz_base = cfg.build_lam_z_vec()
    _per_art: Dict[str, Dict[str, List[float]]] = {}
    for _a in _arts:
        if _a.key == _FAIL_KEY:
            continue
        _k = _a.key
        _per_art[_k] = {
            f"\\theta_{{{_k}}}": [],
            f"\\sigma_{{{_k}}}": [],
            f"\\eta_{{{_k}}}": [],
            f"\\phi_{{{_k}}}": [],
            f"c_{{{_k}}}": [],
            f"\\mu_{{{_k}}}": [],
            f"K_{{{_k}}}": [],
            f"\\lambda_{{{_k}}}": [],
        }

    for _mf in _mu_factors:
        _mu_vec = _mu_base * float(_mf)
        for _c in _c_vals:
            _c_int = int(_c)
            for _K in _K_vals:
                _K_int = int(_K)
                if _K_int < _c_int:
                    continue

                # binary-search the max stable external-arrival factor
                _f_max = _find_max_stable_lam_factor(cfg,
                                                     _mu_vec,
                                                     _c_int,
                                                     _util)
                if _f_max <= 0.0:
                    continue
                _factors = np.linspace(_lam_min_frac * _f_max,
                                       _f_max * 0.999,
                                       _n_steps)
                for _f in _factors:
                    _lams = solve_jackson_lams(cfg.routing,
                                               float(_f) * _lz_base)

                    # solve every node; first saturation stops the ramp for this combo
                    _solved: List[Any] = []
                    _unstable = False
                    for _i, _a in enumerate(_arts):
                        # the FAIL sink is not a sweepable service; its ~zero queue length would
                        # otherwise trip the avg_len<=0 stability guard and abort the whole combo
                        if _a.key == _FAIL_KEY:
                            continue
                        _lam_i = float(_lams[_i])
                        _mu_i = float(_mu_vec[_i])
                        _eps_i = float(_a.vars[f"\\epsilon_{{{_a.key}}}"]["_setpoint"])
                        # disturb the independent inputs; metadata keeps the clean nominal design point
                        if noise is not None:
                            _lam_use, _mu_use, _eps_use = (
                                float(_v) for _v in add_white_noise(
                                    [_lam_i, _mu_i, _eps_i], noise, rng=_rng))
                        else:
                            _lam_use, _mu_use, _eps_use = _lam_i, _mu_i, _eps_i
                        try:
                            _q = Queue(model=model,
                                       lamb=_lam_use,
                                       mu=_mu_use,
                                       c_max=_c_int,
                                       K_max=_K_int)
                            _q.calculate_metrics()
                        except (OverflowError, ValueError, ZeroDivisionError):
                            # solver overflow at large K + c flags this combo as unstable for the rest of the lambda ramp
                            _unstable = True
                            break
                        if _q.rho >= _util or _q.avg_len <= 0:
                            _unstable = True
                            break
                        _solved.append((_a, _lam_i, _mu_i, _lam_use, _mu_use, _eps_use, _q))

                    if _unstable:
                        break
                    for _a, _lam_i, _mu_i, _lam_use, _mu_use, _eps_use, _q in _solved:
                        _k = _a.key
                        _L = float(_q.avg_len)
                        _W = float(_q.avg_wait)
                        # per-artifact per-request payload (kB); 1 kB fallback when missing
                        _d_sym = f"d_{{{_k}}}"
                        if _d_sym in _a.vars:
                            _delta_kB = float(_a.vars[_d_sym]["_setpoint"])
                        else:
                            _delta_kB = 1.0

                        # chi = (1 + eps) * lambda: station throughput carries reprocessed error load, so eta reflects error-handling work (availability stays chi_out = (1 - eps_e2e) * lambda_z, computed in analytic.metrics)
                        _chi = _lam_use * (1.0 + _eps_use)
                        _theta = _L / _K_int
                        _sigma = _lam_use * _W / float(_K_int)
                        _eta = _chi * _K_int / (_mu_use * _c_int)
                        # phi = M_act/M_buf with explicit delta; reduces to L/K under constant payload (sanity-check)
                        _m_act = _L * _delta_kB
                        _m_buf = _K_int * _delta_kB
                        if _m_buf > 0:
                            _phi = _m_act / _m_buf
                        else:
                            _phi = float("nan")

                        _per_art[_k][f"\\theta_{{{_k}}}"].append(_theta)
                        _per_art[_k][f"\\sigma_{{{_k}}}"].append(_sigma)
                        _per_art[_k][f"\\eta_{{{_k}}}"].append(_eta)
                        _per_art[_k][f"\\phi_{{{_k}}}"].append(_phi)
                        _per_art[_k][f"c_{{{_k}}}"].append(float(_c_int))
                        _per_art[_k][f"\\mu_{{{_k}}}"].append(_mu_i)
                        _per_art[_k][f"K_{{{_k}}}"].append(float(_K_int))
                        _per_art[_k][f"\\lambda_{{{_k}}}"].append(_lam_i)
    _out: Dict[str, Dict[str, np.ndarray]] = {}
    for _k, _block in _per_art.items():
        _out[_k] = {_s: np.asarray(_v, dtype=float) for _s, _v in _block.items()}
    return _out
