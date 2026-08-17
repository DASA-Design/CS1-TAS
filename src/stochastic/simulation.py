# -*- coding: utf-8 -*-
"""
Module simulation.py
====================

SimPy DES engine for the open Jackson queueing network used by the CS-01 TAS case study. Two collaborating layers:

    - **QueueNode** per-node SimPy `Resource` with finite capacity K and c parallel servers, plus per-job timing collection (service time, queue wait, total system time) and event-driven L / Lq tracking for time-weighted averages.
    - **simulate_net()** top-level driver. Spawns a `QueueNode` per slot, fires Poisson arrivals at the externally-driven nodes, runs each replication for `horizon` seconds with a `warmup` cut-off, repeats `reps` times, and returns one summary DataFrame with mean and std per node across replications.

`horizon` and `warmup` are SimPy SECONDS, not invocation counts; the method-config JSON declares the latter and the orchestrator in `src.methods.stochastic` converts via `seconds = invocations / sum(lambda_z)` before calling here.

Public API:
    - `QueueNode` node class.
    - `job(env, node_id, nodes, P, results)` single-job SimPy generator.
    - `job_generator(env, node_id, rate, nodes, P, results)` Poisson source.
    - `simulate_net(mu, lam_z, c, K, P, ...)` multi-rep driver.
    - `solve_net(cfg, method_cfg)` NetCfg adapter mirroring `src.analytic.jackson.solve_net`.

# TODO: replace the per-job blocked-after-warmup approximation with a state-conditioned counter (event-driven) once a regression test exists for the M/M/1/K Erlang-B formula.
"""
# native python modules
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

# scientific stack
import numpy as np
import pandas as pd

# discrete-event simulation
import simpy

# local modules
from src.io.config import NetCfg


# ---------------------------------------------------------------------------
# QueueNode
# ---------------------------------------------------------------------------


class QueueNode:
    """**QueueNode** one node in the open queueing network.

    Wraps a SimPy `Resource` (c servers) with a finite capacity K and
    the per-job plus time-weighted bookkeeping needed to recover
    lambda / rho / L / Lq / W / Wq after a run.

    Attributes:
        env (simpy.Environment): shared simulation environment.
        node_id (int): positional slot in the network's node list.
        mu (float): per-server service rate in jobs per time unit.
        c (int): number of parallel servers.
        K (Optional[int]): system capacity (queue + service); `None` for an unbounded queue.
        server (simpy.Resource): resource pool of capacity c.
        blocked_jobs (int): count of jobs that arrived while the system was at capacity K and were dropped.
    """

    def __init__(self,
                 env: simpy.Environment,
                 node_id: int,
                 mu: float,
                 c: int,
                 K: Optional[int],
                 P: np.ndarray,
                 results: List[List[float]],
                 horizon: float) -> None:
        """*__init__()* bind env plus capacity knobs and initialise the bookkeeping buffers.

        Args:
            env (simpy.Environment): simulation environment.
            node_id (int): index of this node in the network.
            mu (float): per-server service rate.
            c (int): number of parallel servers.
            K (Optional[int]): system capacity; None for unbounded.
            P (np.ndarray): full routing matrix (kept for reference; actual routing decisions live in `job()`).
            results (List[List[float]]): shared list of per-node system-time samples (one inner list per node).
            horizon (float): planned simulation duration in seconds.
        """
        self.env = env
        self.node_id = node_id
        self.mu = mu
        self.c = c
        self.K = K
        self.P = P
        self.results = results
        self.horizon = horizon

        # one SimPy Resource per node; capacity = number of servers
        self.server = simpy.Resource(env, capacity=c)

        # blocking counter for the M/M/c/K boundary
        self.blocked_jobs = 0

        # per-job timing samples WITH collection timestamps so the warm-up cut-off can be applied after the run
        self.coll_service_times: List[Tuple[float, float]] = []
        self.coll_queue_times: List[Tuple[float, float]] = []
        self.coll_system_times: List[Tuple[float, float]] = []

        # raw per-job samples kept for backward compatibility
        self.service_times: List[float] = []
        self.queue_times: List[float] = []
        self.system_times: List[float] = []

        # event-driven (length, time-delta) pairs for time-weighted L / Lq computation; updated on every arrival / departure
        self.queue_len_data: List[Tuple[int, float]] = []
        self.system_len_data: List[Tuple[int, float]] = []
        self.last_event_time = 0.0
        self.in_queue = 0
        self.in_service = 0
        self.current_queue_length = 0
        self.current_system_length = 0

        # debug breadcrumb (optional)
        self.job_log: List[dict] = []

    def is_full(self) -> bool:
        """*is_full()* return True when the system is at capacity K (queue + service combined)."""
        return self.K is not None and (self.in_queue + self.in_service) >= self.K

    def record_state_change(self, env: simpy.Environment) -> None:
        """*record_state_change()* close the previous (length, duration) interval and open a new one.

        Called on every arrival, service start, and departure so the
        time-weighted averages stay accurate.

        Args:
            env (simpy.Environment): active simulation environment, used for the current `env.now` timestamp.
        """
        _now = env.now
        _delta = _now - self.last_event_time
        if _delta > 0:
            self.queue_len_data.append((self.current_queue_length, _delta))
            self.system_len_data.append((self.current_system_length, _delta))

        # snapshot the new state
        self.current_queue_length = self.in_queue
        self.current_system_length = self.in_queue + self.in_service
        self.last_event_time = _now

    def service(self, job_id: str):
        """*service()* SimPy generator that consumes one exponential service interval and records its sample.

        Args:
            job_id (str): unique id of the job being served (logged for traceability).

        Yields:
            simpy.events.Event: the service-duration timeout.
        """
        _service_time = random.expovariate(self.mu)
        _coll_time = self.env.now

        self.coll_service_times.append((_service_time, _coll_time))
        self.service_times.append(_service_time)
        self.job_log.append({
            "job_id": job_id,
            "node": self.node_id,
            "start_time": _coll_time,
            "service_time": _service_time,
            "end_time": _coll_time + _service_time,
        })

        yield self.env.timeout(_service_time)


# ---------------------------------------------------------------------------
# Process functions (SimPy generators)
# ---------------------------------------------------------------------------


def job_generator(env: simpy.Environment,
                  node_id: int,
                  rate: float,
                  nodes: List[QueueNode],
                  P: np.ndarray,
                  results: List[List[float]]):
    """*job_generator()* drive Poisson arrivals at one externally-driven node.

    Spawns a `job()` process per arrival, indefinitely.

    Args:
        env (simpy.Environment): simulation environment.
        node_id (int): node receiving the external arrivals.
        rate (float): external arrival rate at `node_id` (jobs / time).
        nodes (List[QueueNode]): full node list (the new job needs the whole network for routing decisions).
        P (np.ndarray): routing matrix for the spawned job.
        results (List[List[float]]): shared system-time accumulator.

    Yields:
        simpy.events.Event: the inter-arrival exponential timeout.
    """
    while True:
        _interarrival = random.expovariate(rate)
        yield env.timeout(_interarrival)
        env.process(job(env, node_id, nodes, P, results))


def job(env: simpy.Environment,
        node_id: int,
        nodes: List[QueueNode],
        P: np.ndarray,
        results: List[List[float]]):
    """*job()* walk one job through the network.

    Arrive at `node_id`, queue, get served, then make a routing decision based on `P[current]` (or exit when no row weight pulls the job onward).

    Args:
        env (simpy.Environment): simulation environment.
        node_id (int): entry node for this job.
        nodes (List[QueueNode]): full node list.
        P (np.ndarray): `(n, n)` routing matrix; `P[i, j]` is the probability of jumping from i to j after service.
        results (List[List[float]]): shared per-node total-time accumulator (results[i] grows with each completion at i).

    Yields:
        simpy.events.Event: queue-request, service-process, etc.
    """
    _job_id = f"job_{env.now:.4f}_{node_id}"
    _current = node_id

    while True:
        _node = nodes[_current]
        _arrival = env.now

        # blocking: drop the job if the node is at capacity K
        if _node.is_full():
            _node.blocked_jobs += 1
            break

        # arrival event: bump the queue counter and snapshot state
        _node.in_queue += 1
        _node.record_state_change(env)

        with _node.server.request() as _req:
            _q_start = env.now
            yield _req

            # service start: queue -> service transition
            _q_time = env.now - _q_start
            _coll = env.now
            _node.coll_queue_times.append((_q_time, _coll))
            _node.queue_times.append(_q_time)
            _node.in_queue -= 1
            _node.in_service += 1
            _node.record_state_change(env)

            # actual service interval (defined inside QueueNode.service)
            yield env.process(_node.service(_job_id))

            # departure: record total system time, free the server
            _total = env.now - _arrival
            _coll = env.now
            _node.coll_system_times.append((_total, _coll))
            _node.system_times.append(_total)
            results[_current].append(_total)
            _node.in_service -= 1
            _node.record_state_change(env)

        # routing decision: exit with prob (1 - sum(P[current])), otherwise pick a successor weighted by the row probabilities
        _exit_prob = 1.0 - float(np.sum(P[_current]))
        if random.random() < _exit_prob:
            break
        _probs = P[_current] / float(np.sum(P[_current]))
        _next = int(np.random.choice(range(len(P)), p=_probs))
        _job_id = f"{_job_id}_to_{_next}"
        _current = _next


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def simulate_net(mu: List[float],
                 lam_z: List[float],
                 c: List[int],
                 K: List[Optional[int]],
                 P: np.ndarray,
                 *,
                 horizon: float = 5000.0,
                 warmup: float = 1000.0,
                 reps: int = 10,
                 seed: Optional[int] = None,
                 verbose: bool = False) -> pd.DataFrame:
    """*simulate_net()* run the full network simulation `reps` times and return a per-node summary frame (mean / std across replications).

    Args:
        mu (List[float]): per-node service rate.
        lam_z (List[float]): per-node external arrival rate. Nodes with `lam_z[i] > 0` get a Poisson generator.
        c (List[int]): per-node server count.
        K (List[Optional[int]]): per-node capacity; `None` for unbounded.
        P (np.ndarray): `(n, n)` routing matrix.
        horizon (float): SimPy time to run each replication, in seconds.
        warmup (float): warm-up cut-off in SimPy seconds; samples collected before this are dropped.
        reps (int): number of independent replications.
        seed (Optional[int]): if given, seeds `random` AND `numpy.random` once at the start so the whole multi-rep run is reproducible.
        verbose (bool): if True, print one line per replication.

    Returns:
        pd.DataFrame: per-node summary with `_mean` / `_std` columns for `lambda`, `mu`, `rho`, `L`, `Lq`, `W`, `Wq`, `Jobs_Served`, `Jobs_Blocked`, `Blocking_Prob`.
    """
    # seed both PRNGs once so the multi-rep run is reproducible end-to-end
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    _n = len(mu)
    _all: List[dict] = []
    for _r in range(reps):
        if verbose:
            print(f"--- Running Replication {_r + 1}/{reps} ---")
        _env = simpy.Environment()
        _results: List[List[float]] = [[] for _ in range(_n)]
        _nodes: List[QueueNode] = []
        for _i in range(_n):
            _nodes.append(QueueNode(_env,
                                    _i,
                                    mu[_i],
                                    c[_i],
                                    K[_i],
                                    P,
                                    _results,
                                    horizon))
        # open the first time-weighted interval at t=0; without this seed event the L / Lq integrals miss the pre-arrival idle window
        for _node in _nodes:
            _node.record_state_change(_env)
        # arm the Poisson sources at every externally-driven node (lam_z[i] > 0)
        for _i, _rate in enumerate(lam_z):
            if _rate > 0:
                _env.process(job_generator(_env,
                                           _i,
                                           _rate,
                                           _nodes,
                                           P,
                                           _results))
        _env.run(until=horizon)
        # close the trailing length interval at t=horizon so time-weighted means cover the full run
        for _node in _nodes:
            _delta = horizon - _node.last_event_time
            if _delta > 0:
                _node.queue_len_data.append(
                    (_node.current_queue_length, _delta))
                _node.system_len_data.append(
                    (_node.current_system_length, _delta))
        _all.extend(_summarise_replication(_nodes,
                                           _r + 1,
                                           mu,
                                           horizon,
                                           warmup))

    _df_all = pd.DataFrame(_all)
    # aggregate mean / std across replications, one row per node
    _df_summary = _df_all.groupby("node").agg({
        "lambda": ["mean", "std"],
        "mu": ["mean", "std"],
        "rho": ["mean", "std"],
        "L": ["mean", "std"],
        "Lq": ["mean", "std"],
        "W": ["mean", "std"],
        "Wq": ["mean", "std"],
        "Jobs_Served": ["mean", "std"],
        "Jobs_Blocked": ["mean", "std"],
        "Blocking_Prob": ["mean", "std"],
    }).reset_index()
    _df_summary.columns = [
        "_".join(_c).strip("_") for _c in _df_summary.columns.values
    ]

    if verbose:
        print(f"\n=== Summary across {reps} replications: shape = " + f"{_df_summary.shape} ===")

    return _df_summary


def _summarise_replication(nodes: List[QueueNode],
                           rep_idx: int,
                           mu: List[float],
                           horizon: float,
                           warmup: float) -> List[dict]:
    """*_summarise_replication()* compute per-node lambda / rho / L / Lq / W / Wq for ONE replication, applying the warm-up cut-off.

    Args:
        nodes (List[QueueNode]): the `QueueNode` instances after the replication has finished running.
        rep_idx (int): 1-based replication number (recorded on the output rows).
        mu (List[float]): nominal service rates (used as fallback when no service samples were collected).
        horizon (float): full simulation duration, seconds.
        warmup (float): warm-up cut-off, seconds.

    Returns:
        List[dict]: one row per node with raw metrics.
    """
    _coll_duration = horizon - warmup
    if horizon > 0:
        _block_ratio = _coll_duration / horizon
    else:
        _block_ratio = 0.0

    _rows: List[dict] = []
    for _i, _node in enumerate(nodes):
        _q_data = _filter_length_data(_node.queue_len_data, warmup)
        _s_data = _filter_length_data(_node.system_len_data, warmup)
        _service_times: List[float] = []
        for _t, _ts in _node.coll_service_times:
            if _ts >= warmup:
                _service_times.append(_t)
        _queue_times: List[float] = []
        for _t, _ts in _node.coll_queue_times:
            if _ts >= warmup:
                _queue_times.append(_t)
        _system_times: List[float] = []
        for _t, _ts in _node.coll_system_times:
            if _ts >= warmup:
                _system_times.append(_t)
        _jobs_served = len(_system_times)
        # block-count rescaling: arrivals are roughly stationary so the post-warmup share equals the time share
        _blocked = int(_node.blocked_jobs * _block_ratio)
        _arrivals = _jobs_served + _blocked
        _L = compute_time_weighted_mean(_s_data, _coll_duration)
        _Lq = compute_time_weighted_mean(_q_data, _coll_duration)
        if _system_times:
            _W = float(np.mean(_system_times))
        else:
            _W = 0.0
        if _queue_times:
            _Wq = float(np.mean(_queue_times))
        else:
            _Wq = 0.0
        if _coll_duration > 0:
            _lambda = _jobs_served / _coll_duration
        else:
            _lambda = 0.0
        # measured per-server service rate; falls back to the nominal mu when no samples landed in the warmup window
        if _service_times:
            _avg_service = float(np.mean(_service_times))
        else:
            _avg_service = 1.0 / mu[_i]
        if _avg_service > 0:
            _mu_eff = 1.0 / _avg_service
        else:
            _mu_eff = mu[_i]
        if _mu_eff > 0:
            _rho = min(1.0, _lambda / (_node.c * _mu_eff))
        else:
            _rho = 0.0
        _model = format_model_string(_node.c, _node.K)
        if _arrivals > 0:
            _blocking_prob = _blocked / _arrivals
        else:
            _blocking_prob = 0.0

        _rows.append({
            "replication": rep_idx,
            "node": _i,
            "type": _model,
            "lambda": _lambda,
            "mu": _mu_eff,
            "rho": _rho,
            "L": _L,
            "Lq": _Lq,
            "W": _W,
            "Wq": _Wq,
            "L_littles": _lambda * _W,
            "Lq_littles": _lambda * _Wq,
            "Jobs_Served": _jobs_served,
            "Jobs_Blocked": _blocked,
            "Blocking_Prob": _blocking_prob,
        })

    return _rows


def _filter_length_data(data: List[Tuple[int, float]],
                        warmup: float) -> List[Tuple[int, float]]:
    """*_filter_length_data()* drop the (length, duration) intervals that fall before `warmup`. Intervals straddling the boundary keep only the post-warmup portion.
    """
    _out: List[Tuple[int, float]] = []
    _cum = 0.0
    for _length, _delta in data:
        _next = _cum + _delta
        if _cum >= warmup:
            _out.append((_length, _delta))
        elif _next > warmup:
            _out.append((_length, _next - warmup))
        _cum = _next
    return _out


def compute_time_weighted_mean(data: List[Tuple[int, float]],
                               fallback_duration: float) -> float:
    """*compute_time_weighted_mean()* sum(length * duration) / sum(duration) over the (length, duration) intervals; falls back to `fallback_duration` for the denominator if every recorded duration is zero.
    """
    if not data:
        return 0.0
    _total_time = sum(_d for _, _d in data) or fallback_duration
    return sum(_length * _d for _length, _d in data) / _total_time


def format_model_string(c: int, K: Optional[int]) -> str:
    """*format_model_string()* build the queue model label, e.g. `M/M/1`, `M/M/2/10`, matching the analytic notation."""
    if c == 1:
        if K is None:
            return "M/M/1"
        return f"M/M/1/{K}"
    if K is None:
        return f"M/M/{c}"
    return f"M/M/{c}/{K}"


# ---------------------------------------------------------------------------
# NetCfg adapter (mirrors `src.analytic.jackson.solve_net`)
# ---------------------------------------------------------------------------


# stochastic metric columns that get a `_mean` + `_std` pair out of `simulate_net`'s groupby-agg summary
_STAT_COLS = ("lambda", "mu", "rho", "L", "Lq", "W", "Wq")


def solve_net(cfg: NetCfg,
              method_cfg: Dict[str, Any]) -> pd.DataFrame:
    """*solve_net()* run the SimPy DES engine for one resolved `(profile, scenario)` pair and return per-node metrics with the same schema the analytic method produces, plus `<metric>_std` columns for the stochastic CI machinery.

    The method config declares horizon / warmup in *invocations* but `simulate_net` runs in SimPy seconds; the `seconds = invocations / sum(lambda_z)` conversion happens here so callers stay in the natural "invocation count" unit.

    Args:
        cfg (NetCfg): resolved network configuration. Provides `mu`, `c`, `K`, `lambda_z`, and `routing` per artifact.
        method_cfg (Dict[str, Any]): contents of `data/config/method/stochastic.json`. Expected keys:
            - `seed` (int): PRNG seed.
            - `horizon_invocations` (int): total jobs per rep before the run is cut off.
            - `warmup_invocations` (int): jobs to discard for Welch's method.
            - `replications` (int): number of independent reps.

    Returns:
        pd.DataFrame: one row per artifact with columns `node`, `key`, `name`, `type`, `lambda`, `mu`, `c`, `K`, `rho`, `L`, `Lq`, `W`, `Wq`, plus `<metric>_std` for every stochastic metric.
    """
    _mu: List[float] = []
    _c: List[int] = []
    _K: List[Optional[int]] = []
    for _a in cfg.artifacts:
        _mu.append(float(_a.mu))
        _c.append(int(_a.c))
        if _a.K is not None:
            _K.append(int(_a.K))
        else:
            _K.append(None)
    _lambda_z = cfg.build_lam_z_vec().tolist()
    _P = cfg.routing
    # invocation count -> SimPy seconds: with sum(lambda_z) jobs/s, reaching N invocations takes ~ N / sum(lambda_z) s in expectation; fall back to 1.0 on an all-zero lambda_z vector
    _total_rate = float(sum(_lambda_z))
    if _total_rate <= 0:
        _total_rate = 1.0
    _horizon = float(method_cfg["horizon_invocations"]) / _total_rate
    _warmup = float(method_cfg["warmup_invocations"]) / _total_rate
    _summary = simulate_net(
        mu=_mu,
        lam_z=_lambda_z,
        c=_c,
        K=_K,
        P=_P,
        horizon=_horizon,
        warmup=_warmup,
        reps=int(method_cfg["replications"]),
        seed=int(method_cfg["seed"]),
        verbose=False,
    )

    return _reshape_summary(_summary, cfg)


def _reshape_summary(summary: pd.DataFrame,
                     cfg: NetCfg) -> pd.DataFrame:
    """*_reshape_summary()* turn `simulate_net`'s groupby-agg frame into a per-node frame with analytic-compatible column names (`lambda` instead of `lambda_mean`, etc.) plus `_std` companions for every stochastic metric.

    Args:
        summary (pd.DataFrame): `simulate_net()`'s output (columns like `lambda_mean`, `lambda_std`, ...).
        cfg (NetCfg): resolved config, used to attach `key`, `name`, `type`, `c`, `K` columns per slot.

    Returns:
        pd.DataFrame: flat per-node frame ordered by `node` index.
    """
    _rows = []
    for _i, _a in enumerate(cfg.artifacts):
        _s = summary.loc[summary["node"] == _i].iloc[0]
        if _a.K is not None:
            _k_val = int(_a.K)
        else:
            _k_val = None
        _row: Dict[str, Any] = {
            "node": _i,
            "key": _a.key,
            "name": _a.name,
            "type": _a.type_,
            "c": int(_a.c),
            "K": _k_val,
        }
        # `<metric>` (mean across reps) + `<metric>_std` for CIs
        for _m in _STAT_COLS:
            _row[_m] = float(_s[f"{_m}_mean"])
            _row[f"{_m}_std"] = float(_s[f"{_m}_std"])
        _rows.append(_row)

    # canonical column order so downstream plotters see the same shape as the analytic method's output
    _cols = [
        "node", "key", "name", "type",
        "lambda", "mu", "c", "K",
        "rho", "L", "Lq", "W", "Wq",
    ]
    _cols += [f"{_m}_std" for _m in _STAT_COLS]
    return pd.DataFrame(_rows)[_cols]
