# -*- coding: utf-8 -*-
"""
Module search.py
================

DASA-guided routing search for the CS-1 case study.

Stage 1 picks the three lowest-`eps` services per type from the Table 7.3 catalogue (deterministic; `pick_pool`). Stage 2 sweeps the routing dispatch weights over a coarse simplex grid; each candidate is scored by ONE `dimensional.run_from_netcfg` call. Coefficients are primary; the architecture metrics ($W_{e2e}$, $\\varepsilon_{e2e}$, $\\sigma_{\\text{arch}}$, $\\eta_{\\text{arch}}$) are deduced from the same solve. The winner minimises $\\varepsilon_{e2e}$ among DASA + R1 + R2 survivors.

Public API:
    - `pick_pool(catalogue, svc_type, n)` deterministic Stage-1 selector.
    - `compute_bounds(lam_z, r1_max, r2_max_s, *, k_min, c_min, mu_min)` -> the binding bounds at the locked envelope.
    - `load_skeleton()` -> (TAS_{1..6}, third-party template) from opti.json/aggregate.
    - `build_cfg(catalogue, pools, weights, *, lam_z, tas_artifacts, template)` -> in-memory `NetCfg`.
    - `run_search(catalogue, pools, lam_z, tas_artifacts, template, *, method_cfg, bounds, step, progress_callback)` -> results DataFrame.
    - `pick_winner(df, bounds)` -> winning row + `winner_status`.

This module is NOT re-exported from `src.dimensional.__init__` because `run_from_netcfg` lives in `src.methods.dimensional` which itself imports from `src.dimensional`. Importing search via the package init would force a circular dependency. Callers import directly: `from src.dimensional.search import ...`.
"""
# native python modules
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# scientific stack
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# local modules
from src.analytic import aggregate_net, check_reqs, solve_network
from src.dimensional.routing import build_routing
from src.io import ArtifactSpec, NetCfg, load_profile


# Maximum per-node utilisation tolerated on MAS / AS services. The published CS-1
# canonical bound is 0.65 (E-QS regime); above it, the service spends most of its
# time queueing and the dispatch weight is operationally invalid even if the
# Jackson solver still converges.
_DEFAULT_RHO_MAX = 0.65


# Locked envelope: the published CS-1 chapter pins these and the search does not
# touch them. K_min and c_min match the QN solver's chosen ceilings; mu_min is the
# catalogue's slowest published service (MAS_{3}) and sets the worst-case eta_R1
# the search compares every candidate against.
_LOCKED_K = 16
_LOCKED_C = 1
_CATALOGUE_MU_MIN = 150.0

# absorbing failure-sink node (loss-network model); candidates leak give-ups here and
# availability is read from flow as eps_e2e = lambda_FAIL / lambda_z
_FAIL_KEY = "FAIL_{1}"


def pick_pool(catalogue: Dict[str, Any],
              svc_type: str,
              n: int = 3) -> List[str]:
    """*pick_pool()* return the `n` lowest-`eps` service keys for `svc_type`.

    Args:
        catalogue (Dict[str, Any]): the loaded catalogue dict from `src.io.load_catalogue`.
        svc_type (str): one of `MAS`, `AS`, `DS`.
        n (int): number of services to pick. Defaults to 3.

    Returns:
        List[str]: service keys ordered by ascending `eps`.
    """
    _block = catalogue[svc_type]
    _sorted = sorted(_block.items(), key=lambda _kv: _kv[1]["eps"])
    # Drop the eps values; keep only the keys, in the same ascending order.
    _keys: List[str] = []
    for _kv in _sorted[:n]:
        _keys.append(_kv[0])
    return _keys


def compute_bounds(lam_z: float,
                   r1_max: float,
                   r2_max_s: float,
                   *,
                   k_min: int = _LOCKED_K,
                   c_min: int = _LOCKED_C,
                   mu_min: float = _CATALOGUE_MU_MIN) -> Dict[str, float]:
    """*compute_bounds()* compute the binding DASA bounds at the catalogue worst case.

    Formulas per dasa-modelling.md Block 4: `sigma_R2 = W_R2 * lam_z / K_min`, `eta_R1 = lam_z * (1 + eps_R1) * K_min / (mu_min * c_min)`. The defaults reproduce the dissertation reference values `sigma_R2 = 0.525` and `eta_R1 = 34.798`.

    Args:
        lam_z (float): external arrival rate in 1/s (323 at the locked envelope).
        r1_max (float): R1 ceiling as a fraction (e.g. 0.01 = 1%).
        r2_max_s (float): R2 ceiling in seconds (e.g. 0.026 = 26 ms).
        k_min (int): minimum queue capacity (default 16, locked envelope).
        c_min (int): minimum server count (default 1, locked envelope).
        mu_min (float): minimum service rate across the catalogue (default 150, MAS_{3}).

    Returns:
        Dict[str, float]: keys `sigma_R2`, `eta_R1`, plus `lam_z`, `K_min`, `c_min`, `mu_min`, `r1_max`, `r2_max_s` for downstream consumers that re-derive `sigma_arch` / `eta_arch`.
    """
    _sigma_R2 = r2_max_s * lam_z / k_min
    _eta_R1 = lam_z * (1.0 + r1_max) * k_min / (mu_min * c_min)
    _bounds: Dict[str, float] = {
        "sigma_R2": float(_sigma_R2),
        "eta_R1": float(_eta_R1),
        "lam_z": float(lam_z),
        "K_min": int(k_min),
        "c_min": int(c_min),
        "mu_min": float(mu_min),
        "r1_max": float(r1_max),
        "r2_max_s": float(r2_max_s),
    }
    return _bounds


def load_skeleton() -> Tuple[List[ArtifactSpec], ArtifactSpec]:
    """*load_skeleton()* extract the six TAS aggregate-stage artifacts and one third-party template from `opti.json/aggregate`.

    The TAS skeleton is reused verbatim across every search candidate; the third-party template (`MAS_{4}`) is cloned for each pool service.

    Returns:
        Tuple[List[ArtifactSpec], ArtifactSpec]: `(tas_artifacts, template)`.
    """
    _cfg = load_profile(adaptation="aggregate")
    # Walk the artifact list once and split into the two outputs we need: the six
    # TAS stages (kept verbatim) and one third-party template (cloned per pool).
    _tas: List[ArtifactSpec] = []
    _template: Optional[ArtifactSpec] = None
    for _a in _cfg.artifacts:
        if _a.key.startswith("TAS_"):
            _tas.append(_a)
        if _a.key == "MAS_{4}":
            _template = _a
    # MAS_{4} is the canonical clone source. If opti.json/aggregate ever drops it,
    # the search loses its template and we want a loud failure here rather than a
    # silent None propagating into the artifact builder.
    if _template is None:
        _msg = "MAS_{4} not found in opti.json/aggregate; cannot supply a clone template"
        raise KeyError(_msg)
    return _tas, _template


def _iter_ints(k: int, n: int) -> Iterator[Tuple[int, ...]]:
    """*_iter_ints()* yield every composition of `n` into `k` non-negative integer parts.

    Recursion picks the first part (`_i` in `0..n`) and delegates the remainder; the recursive call halts when there is only one part left to assign.

    Args:
        k (int): number of integer parts to split `n` into.
        n (int): the integer budget to be distributed across the `k` parts.

    Yields:
        Tuple[int, ...]: one composition (a `k`-tuple of non-negative integers summing to `n`).
    """
    if k == 1:
        yield (n,)
    else:
        for _i in range(n + 1):
            for _rest in _iter_ints(k - 1, n - _i):
                yield (_i,) + _rest


def _iter_simplex(k: int, n: int,
                  *,
                  interior_only: bool = True) -> Iterator[Tuple[float, ...]]:
    """*_iter_simplex()* scale every integer composition of `n` into `k` parts by `1/n` to obtain a simplex weight tuple.

    With `interior_only=True` (default), drops every tuple that has a zero entry. The dispatched pool services must each receive traffic ($\\chi_j > 0$) or PyDASA's coefficient solver would see a zero-valued setpoint, and zero weights have no operational meaning in the search anyway.

    With `interior_only=False`, includes the simplex boundary too (one or more weights at zero).

    Args:
        k (int): number of weights per tuple.
        n (int): grid resolution; the smallest non-zero weight is `1/n`.
        interior_only (bool): when True, only strict-positive tuples.

    Yields:
        Tuple[float, ...]: one weight tuple on the simplex.
    """
    for _comp in _iter_ints(k, n):
        # Skip simplex-boundary tuples (any zero) when the caller wants only the interior.
        if interior_only:
            _has_zero = False
            for _v in _comp:
                if _v == 0:
                    _has_zero = True
                    break
            if _has_zero:
                continue
        _scaled: List[float] = []
        for _v in _comp:
            _scaled.append(_v / n)
        yield tuple(_scaled)


def _swap_deps(deps: List[Any], swap: Callable[[str], str]) -> List[Any]:
    """*_swap_deps()* apply the subscript swap to every string entry of the `_depends` list, leaving non-string entries unchanged.

    Args:
        deps (List[Any]): the `_depends` list from a PyDASA Variable dict.
        swap (Callable[[str], str]): the per-string rename function provided by `_clone_spec`.

    Returns:
        List[Any]: a new list with string entries renamed.
    """
    _swapped: List[Any] = []
    for _d in deps:
        if isinstance(_d, str):
            _swapped.append(swap(_d))
        else:
            _swapped.append(_d)
    return _swapped


def _clone_spec(key: str,
                name: str,
                mu: float,
                eps: float,
                lam_z: float,
                template: ArtifactSpec) -> ArtifactSpec:
    """*_clone_spec()* clone a template artifact, substitute its subscript label everywhere, and override the `mu` and `eps` setpoints to the catalogue values.

    Renames every occurrence of the template's subscript (e.g. `MAS_{4}`) to `key` (e.g. `MAS_{2}`) in: the `vars` dict keys; the `_sym`, `_alias`, `_name`, `description`, `_depends` fields of every variable. Then writes `mu` and `eps` into the `_setpoint` / `_std_setpoint` / `_mean` / `_std_mean` / `_data` slots of the corresponding variables.
    """
    _tpl_sub = template.key
    _tpl_flat = _tpl_sub.replace("{", "").replace("}", "")
    _key_flat = key.replace("{", "").replace("}", "")

    # `_swap` rewrites both the LaTeX subscript form (`MAS_{4}` -> `MAS_{2}`)
    # and the underscore-flat form used in `_alias` fields (`MAS_4` -> `MAS_2`).
    def _swap(_s: str) -> str:
        return _s.replace(_tpl_sub, key).replace(_tpl_flat, _key_flat)

    # Rebuild the vars dict with renamed keys and renamed string fields. Numeric
    # fields are kept as-is; the mu / eps overrides happen in the next block.
    _new_vars: Dict[str, Dict[str, Any]] = {}
    for _sym, _v in template.vars.items():
        _new_v = copy.deepcopy(_v)
        for _f in ("_sym", "_alias", "_name", "description"):
            if isinstance(_new_v.get(_f), str):
                _new_v[_f] = _swap(_new_v[_f])
        _deps = _new_v.get("_depends", [])
        if isinstance(_deps, list):
            _new_v["_depends"] = _swap_deps(_deps, _swap)
        _new_vars[_swap(_sym)] = _new_v

    # Service rate and failure rate come from the catalogue, not the template;
    # write them into every mirror field PyDASA reads (setpoint / std / mean / data).
    _mu_sym = f"\\mu_{{{key}}}"
    _eps_sym = f"\\epsilon_{{{key}}}"
    for _sym, _val in ((_mu_sym, mu), (_eps_sym, eps)):
        if _sym in _new_vars:
            for _f in ("_setpoint", "_std_setpoint", "_mean", "_std_mean"):
                _new_vars[_sym][_f] = float(_val)
            _new_vars[_sym]["_data"] = [float(_val)]

    _spec = ArtifactSpec(
        key=key,
        name=name,
        type_=template.type_,
        lambda_z=float(lam_z),
        L_z=template.L_z,
        vars=_new_vars,
    )
    return _spec


def _build_routing(node_keys: List[str],
                   eps_by_key: Dict[str, float],
                   weights: Dict[str, Tuple[float, ...]],
                   mas_keys: List[str],
                   as_keys: List[str],
                   ds_keys: List[str],
                   *,
                   retry: Optional[Dict[str, int]] = None) -> np.ndarray:
    """*_build_routing()* thin wrapper around `src.dimensional.routing.build_routing`.

    The routing-matrix construction lives in `routing.py` so it is testable in isolation and reusable. This wrapper preserves the search module's internal call surface and the `retry` keyword (per-handler retry-depth map; `None` is selection-only). See `build_routing` for the full edge specification.
    """
    return build_routing(node_keys, eps_by_key, weights,
                         mas_keys, as_keys, ds_keys, retry=retry)


def _set_lam_z(artifact: ArtifactSpec, lam_z: float) -> ArtifactSpec:
    """*_set_lam_z()* return a fresh `ArtifactSpec` with the requested external arrival rate at the entry node.

    `ArtifactSpec` is frozen, so we cannot mutate `lambda_z` in place. The replacement copies every other field verbatim.

    Args:
        artifact (ArtifactSpec): the entry-node artifact (typically TAS_{1}).
        lam_z (float): the external arrival rate to write.

    Returns:
        ArtifactSpec: a new artifact identical to `artifact` except for `lambda_z`.
    """
    _new = ArtifactSpec(
        key=artifact.key,
        name=artifact.name,
        type_=artifact.type_,
        lambda_z=float(lam_z),
        L_z=artifact.L_z,
        vars=artifact.vars,
    )
    return _new


def build_cfg(catalogue: Dict[str, Any],
              pools: Dict[str, List[str]],
              weights: Dict[str, Tuple[float, ...]],
              *,
              lam_z: float,
              tas_artifacts: List[ArtifactSpec],
              template: ArtifactSpec,
              retry: Optional[Dict[str, int]] = None) -> NetCfg:
    """*build_cfg()* assemble one candidate `NetCfg` from chosen pools, dispatch weights, and the TAS skeleton.

    Node order: `[TAS_{1}, TAS_{2}, TAS_{3}] + mas_pool + as_pool + [TAS_{4}] + ds_pool + [TAS_{5}, TAS_{6}]`. The pool order inside each block matches the order in `pools[svc_type]` (i.e., the order returned by `pick_pool`).

    Args:
        catalogue (Dict[str, Any]): loaded catalogue dict.
        pools (Dict[str, List[str]]): selected service keys per type; e.g. `{"MAS": ["MAS_{2}", "MAS_{4}", "MAS_{1}"], "AS": [...], "DS": [...]}`.
        weights (Dict[str, Tuple[float, ...]]): dispatch weight tuples per pool, e.g. `{"M": (0.4, 0.3, 0.3), "A": (...), "D": (...)}`. Each tuple sums to 1 and aligns with the pool's key order.
        lam_z (float): external arrival rate (passed through to TAS_{1}).
        tas_artifacts (List[ArtifactSpec]): the six locked TAS stage artifacts.
        template (ArtifactSpec): one third-party artifact used as the clone template.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map (`TAS_{4}` / `TAS_{5}` / `TAS_{6}`) forwarded to the routing builder; `None` (default) is selection-only.

    Returns:
        NetCfg: in-memory configuration ready for `dimensional.run_from_netcfg`.
    """
    _mas_keys = list(pools["MAS"])
    _as_keys = list(pools["AS"])
    _ds_keys = list(pools["DS"])

    # Clone the third-party template once per chosen service and record the eps so
    # the routing builder can read it without re-walking the catalogue.
    _eps_by_key: Dict[str, float] = {}
    _specs_by_key: Dict[str, ArtifactSpec] = {}
    for _svc, _keys in (("MAS", _mas_keys), ("AS", _as_keys), ("DS", _ds_keys)):
        for _k in _keys:
            _cat = catalogue[_svc][_k]
            _eps_by_key[_k] = float(_cat["eps"])
            _specs_by_key[_k] = _clone_spec(
                key=_k,
                name=f"{_svc} variant {_k}",
                mu=float(_cat["mu"]),
                eps=float(_cat["eps"]),
                lam_z=0.0,
                template=template,
            )

    # Order the artifacts to match the routing-matrix index convention.
    _tas_by_key: Dict[str, ArtifactSpec] = {}
    for _a in tas_artifacts:
        _tas_by_key[_a.key] = _a
    _ordered: List[ArtifactSpec] = []
    _node_keys: List[str] = []
    for _k in ("TAS_{1}", "TAS_{2}", "TAS_{3}"):
        _ordered.append(_tas_by_key[_k])
        _node_keys.append(_k)
    for _k in _mas_keys:
        _ordered.append(_specs_by_key[_k])
        _node_keys.append(_k)
    for _k in _as_keys:
        _ordered.append(_specs_by_key[_k])
        _node_keys.append(_k)
    _ordered.append(_tas_by_key["TAS_{4}"])
    _node_keys.append("TAS_{4}")
    for _k in _ds_keys:
        _ordered.append(_specs_by_key[_k])
        _node_keys.append(_k)
    for _k in ("TAS_{5}", "TAS_{6}"):
        _ordered.append(_tas_by_key[_k])
        _node_keys.append(_k)
    # absorbing failure sink (loss-network); a dummy high-mu service so it carries ~zero load
    _ordered.append(_clone_spec(key=_FAIL_KEY, name="Failure Sink",
                                mu=1e9, eps=0.0, lam_z=0.0, template=template))
    _node_keys.append(_FAIL_KEY)

    # The TAS template is loaded from opti.json/aggregate where the entry rate may
    # be zero. The search drives the requested `lam_z`, so swap the entry artifact.
    if _ordered[0].lambda_z != lam_z:
        _ordered[0] = _set_lam_z(_ordered[0], lam_z)

    _routing = _build_routing(_node_keys, _eps_by_key, weights,
                              _mas_keys, _as_keys, _ds_keys, retry=retry)

    _cfg = NetCfg(
        profile="search",
        scenario="winner",
        label="DASA-guided search candidate",
        artifacts=_ordered,
        routing=_routing,
        enforce_limits=True,
    )
    return _cfg


def _max_pool_rho(nodes: pd.DataFrame) -> float:
    """*_max_pool_rho()* return the largest `rho` among MAS and AS service rows in the solved-network frame.

    DS services are excluded from the operational rho ceiling per the user direction (the rho<=0.65 limit applies to MAS and AS only). When no MAS/AS row exists (defensive), returns `0.0`.

    Args:
        nodes (pd.DataFrame): per-node frame from `src.analytic.solve_network` (carries `key` and `rho` columns).

    Returns:
        float: max rho across the active MAS + AS rows.
    """
    _peak = 0.0
    for _i in range(len(nodes)):
        _row = nodes.iloc[_i]
        _key = str(_row["key"])
        if _key.startswith("MAS_") or _key.startswith("AS_"):
            _r = float(_row["rho"])
            if _r > _peak:
                _peak = _r
    return _peak


def _score(cfg: NetCfg,
           weights: Dict[str, Tuple[float, ...]],
           *,
           bounds: Dict[str, float],
           rho_max: float = _DEFAULT_RHO_MAX) -> Dict[str, Any]:
    """*_score()* score one in-memory candidate against the operational rho ceiling and the DASA / R1 / R2 bounds.

    Two-stage evaluation; each stage either flags the candidate failed and short-circuits, or proceeds:

        1. **Analytic solve** (`src.analytic.solve_network`). Cheap (~ms per candidate). Returns the per-node frame with `rho`, `lambda`, `W`, `L`. Unstable Jackson configurations (`rho >= 1` somewhere) raise `ValueError`; the candidate is marked `stable=False` and the row is returned.
        2. **Operational filter** (rho ceiling on MAS / AS). Compares the largest rho across active MAS / AS rows to `rho_max` (default 0.65, the E-QS canonical bound). When any service exceeds it, the candidate is marked `rho_pass=False`; aggregate metrics are NOT computed and the row keeps its `inf` placeholders.
        3. **Aggregate metrics** (`aggregate_net` + `check_reqs`). Computes `W_e2e`, `eps_e2e`, the R1 / R2 verdict, and the architecture coefficients (`sigma_arch`, `eta_arch`) from the analytic solve already in hand; no PyDASA per-artifact work, so this stage stays in the millisecond range.

    The dimensional / PyDASA path is only needed to render the winner on the yoly chart; the notebook does that once at the end. Keeping `_score` analytic-only is what makes the full step=0.1 sweep tractable.

    The row schema is constant across all candidates so the search frame is rectangular. Fields not yet computed at the point of short-circuit remain at their default (`inf` metrics, `False` PASS flags).
    """
    # Default row reflects the marked-failed shape; it stays as-is on exception
    # and is updated field-by-field as each stage succeeds.
    _row: Dict[str, Any] = {
        "w_M": weights["M"],
        "w_A": weights["A"],
        "w_D": weights["D"],
        "W_e2e": float("inf"),
        "eps_e2e": float("inf"),
        "sigma_arch": float("inf"),
        "eta_arch": float("inf"),
        "rho_max_active": float("inf"),
        "dasa_pass": False,
        "rho_pass": False,
        "r1_pass": False,
        "r2_pass": False,
        "stable": False,
    }

    # Stage 1: cheap analytic solve. Catches unstable Jackson configurations.
    try:
        _nodes = solve_network(cfg)
    except ValueError:
        return _row
    _row["stable"] = True

    # Stage 2: operational rho ceiling on MAS / AS services.
    _peak_rho = _max_pool_rho(_nodes)
    _row["rho_max_active"] = float(_peak_rho)
    _row["rho_pass"] = bool(_peak_rho <= rho_max)
    if not _row["rho_pass"]:
        return _row

    # Stage 3: aggregate the per-node frame into W_e2e + eps_e2e + the R1 / R2
    # verdict from the loss-network flow (routing carries the give-up leaks to FAIL).
    _lam_z_entry = float(cfg.build_lam_z_vec()[0])
    _net = aggregate_net(_nodes, lam_z=_lam_z_entry, routing=cfg.routing).iloc[0]
    _req = check_reqs(_nodes, lam_z=_lam_z_entry, routing=cfg.routing)
    _W_e2e = float(_net["W_e2e"])
    _eps_e2e = float(_net["eps_e2e"])
    # Architecture coefficients are derived from the network aggregate at the
    # locked envelope; with K, c, mu fixed the inequalities reduce algebraically
    # to R2 and R1 respectively. Keeping both views explicit makes the chapter's
    # narrative readable in DASA coordinates.
    _sigma_arch = _W_e2e * bounds["lam_z"] / bounds["K_min"]
    _eta_arch = bounds["lam_z"] * (1.0 + _eps_e2e) * bounds["K_min"] / (bounds["mu_min"] * bounds["c_min"])
    _dasa_pass = (_sigma_arch < bounds["sigma_R2"]) and (_eta_arch < bounds["eta_R1"])

    _row["W_e2e"] = _W_e2e
    _row["eps_e2e"] = _eps_e2e
    _row["sigma_arch"] = float(_sigma_arch)
    _row["eta_arch"] = float(_eta_arch)
    _row["dasa_pass"] = bool(_dasa_pass)
    _row["r1_pass"] = bool(_req["R1"]["pass"])
    _row["r2_pass"] = bool(_req["R2"]["pass"])
    return _row


def run_search(catalogue: Dict[str, Any],
               pools: Dict[str, List[str]],
               lam_z: float,
               tas_artifacts: List[ArtifactSpec],
               template: ArtifactSpec,
               *,
               bounds: Dict[str, float],
               step: float = 0.1,
               interior_only: bool = True,
               rho_max: float = _DEFAULT_RHO_MAX,
               retry: Optional[Dict[str, int]] = None,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """*run_search()* sweep every dispatch-weight tuple on the simplex grid and score each candidate.

    Each candidate goes through `_score`'s analytic-only pipeline (solve_network -> rho filter -> aggregate_net + check_reqs). The rho ceiling short-circuits most of the 46656 candidates a step=0.1 sweep produces; the survivors get the cheap aggregate computation.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection from Stage 1.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        step (float): simplex grid step. Default 0.1 -> 36 interior weight profiles per pool, 46656 candidates total; the cheap rho filter keeps the run tractable by short-circuiting infeasible ones.
        interior_only (bool): when True (default), restricts the sweep to strict-positive weight tuples so every pool service has `chi > 0`.
        rho_max (float): per-node utilisation ceiling on MAS / AS services (default 0.65, the E-QS canonical bound). Candidates whose max-pool rho exceeds this skip the aggregate metrics computation.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map applied to every candidate (`None` = selection-only). Held fixed across the grid; used by the confusion-matrix cell to validate DASA's prediction over a retry-on region.
        progress_callback (Optional[Callable[[int, int], None]]): called after every candidate with `(i_completed, total)`. Wire `tqdm.update` here from the notebook if a progress bar is wanted; absence is silent.

    Returns:
        pd.DataFrame: one row per candidate with columns `w_M`, `w_A`, `w_D`, `W_e2e`, `eps_e2e`, `sigma_arch`, `eta_arch`, `rho_max_active`, `dasa_pass`, `rho_pass`, `r1_pass`, `r2_pass`, `stable`.
    """
    _n = int(round(1.0 / step))
    # Each pool can carry a different number of services; the simplex is sized
    # per-pool so a singleton (e.g., DS_{1} in CS-1) yields the trivial (1.0,)
    # tuple and only 3-service pools (MAS, AS) sweep the full 36-tuple interior.
    _per_pool: Dict[str, List[Tuple[float, ...]]] = {}
    for _svc, _short in (("MAS", "M"), ("AS", "A"), ("DS", "D")):
        _k = len(pools[_svc])
        _tuples: List[Tuple[float, ...]] = []
        for _comp in _iter_simplex(_k, _n, interior_only=interior_only):
            _tuples.append(_comp)
        _per_pool[_short] = _tuples
    _total = len(_per_pool["M"]) * len(_per_pool["A"]) * len(_per_pool["D"])

    _rows: List[Dict[str, Any]] = []
    _i = 0
    for _wm in _per_pool["M"]:
        for _wa in _per_pool["A"]:
            for _wd in _per_pool["D"]:
                _weights = {"M": _wm, "A": _wa, "D": _wd}
                _cfg = build_cfg(catalogue, pools, _weights,
                                 lam_z=lam_z,
                                 tas_artifacts=tas_artifacts,
                                 template=template,
                                 retry=retry)
                _row = _score(_cfg, _weights,
                              bounds=bounds,
                              rho_max=rho_max)
                _rows.append(_row)
                _i = _i + 1
                if progress_callback is not None:
                    progress_callback(_i, _total)
    _df = pd.DataFrame(_rows)
    return _df


def _to_dict(row: pd.Series, status: str) -> Dict[str, Any]:
    """*_to_dict()* convert a DataFrame row to a flat `Dict[str, Any]` and attach the `winner_status` field.

    `pd.Series.to_dict()` keys as `Hashable`; type-checkers narrow them to `str` after this re-walk so callers get a clean `Dict[str, Any]`.

    Args:
        row (pd.Series): one row from the results DataFrame.
        status (str): `"feasible"`, `"closest_miss"`, or `"no_dasa_survivors"`.

    Returns:
        Dict[str, Any]: flat dict carrying every column plus `winner_status`.
    """
    _result: Dict[str, Any] = {}
    _raw = row.to_dict()
    for _k, _v in _raw.items():
        _result[str(_k)] = _v
    _result["winner_status"] = status
    return _result


def _dc_margin(sigma_arch: float, eta_arch: float, bounds: Dict[str, float]) -> float:
    """*_dc_margin()* return the minimum of the sigma-axis and eta-axis margins for one row.

    The DC margin says "how far inside the DASA viable region does this candidate sit"; using the minimum across the two binding axes means a large margin requires BOTH coefficients to be well inside their bounds, not just one of them.

    Args:
        sigma_arch (float): the candidate's architecture sigma.
        eta_arch (float): the candidate's architecture eta.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        float: `min((sigma_R2 - sigma_arch) / sigma_R2, (eta_R1 - eta_arch) / eta_R1)`. Positive when the candidate is inside both bounds; negative when at least one axis overshoots.
    """
    _sigma_margin = (bounds["sigma_R2"] - sigma_arch) / bounds["sigma_R2"]
    _eta_margin = (bounds["eta_R1"] - eta_arch) / bounds["eta_R1"]
    if _sigma_margin < _eta_margin:
        return _sigma_margin
    return _eta_margin


def _dasa_distance(sigma_arch: float, eta_arch: float, bounds: Dict[str, float]) -> float:
    """*_dasa_distance()* return the additive overshoot over both DASA bounds.

    Each axis contributes `max(0, x / bound - 1)`: zero when inside the bound, positive when over. The sum measures how far OUTSIDE the viable region the candidate is, with both axes weighing equally on the same normalised scale. A configuration with `dasa_distance == 0` clears both bounds; the closest-miss pick is the row with smallest `dasa_distance`, pushing both axes toward feasibility at once.

    Args:
        sigma_arch (float): the candidate's architecture sigma.
        eta_arch (float): the candidate's architecture eta.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        float: sigma overshoot + eta overshoot, normalised by the respective bounds.
    """
    _sigma_over = sigma_arch / bounds["sigma_R2"] - 1.0
    if _sigma_over < 0.0:
        _sigma_over = 0.0
    _eta_over = eta_arch / bounds["eta_R1"] - 1.0
    if _eta_over < 0.0:
        _eta_over = 0.0
    return _sigma_over + _eta_over


def _pick_best(survivors: pd.DataFrame, bounds: Dict[str, float]) -> Dict[str, Any]:
    """*_pick_best()* select the largest-DC-margin row among feasible survivors.

    Both sigma_arch and eta_arch are pushed down: the candidate furthest inside the viable region on its tighter axis wins. Tie-break (within 1e-9 on the DC margin) is smallest `eps_e2e`, so when multiple candidates achieve the same DASA margin the more reliable one is preferred.

    Args:
        survivors (pd.DataFrame): rows that pass DASA + R1 + R2 (non-empty).
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        Dict[str, Any]: the winning row as a flat dict with `winner_status = "feasible"`.
    """
    _margins: List[float] = []
    for _i in range(len(survivors)):
        _row = survivors.iloc[_i]
        _margins.append(_dc_margin(float(_row["sigma_arch"]),
                                   float(_row["eta_arch"]),
                                   bounds))
    _scored = survivors.copy()
    _scored["dc_margin"] = _margins
    _sorted = _scored.sort_values("dc_margin", ascending=False).reset_index(drop=True)
    _best_margin = float(_sorted.iloc[0]["dc_margin"])
    _margin_col = _sorted["dc_margin"].astype(float)
    _ties = _sorted[(_margin_col - _best_margin).abs() < 1e-9].copy()
    _winner_row = _ties.nsmallest(1, "eps_e2e").iloc[0]
    _result = _to_dict(_winner_row, "feasible")
    return _result


def _pick_closest_miss(df: pd.DataFrame, bounds: Dict[str, float]) -> Dict[str, Any]:
    """*_pick_closest_miss()* select the smallest-`dasa_distance` row from the full results frame.

    Used when no candidate clears all three filters (DASA + R1 + R2). The selected row sits as close to the viable region as the sweep allows; the chapter's narrative then states which axis is the binding constraint.

    Args:
        df (pd.DataFrame): full results frame from `run_search`.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        Dict[str, Any]: the closest-miss row as a flat dict with `winner_status = "closest_miss"`.
    """
    _distances: List[float] = []
    for _i in range(len(df)):
        _row = df.iloc[_i]
        _distances.append(_dasa_distance(float(_row["sigma_arch"]),
                                         float(_row["eta_arch"]),
                                         bounds))
    _scored = df.copy()
    _scored["dasa_distance"] = _distances
    _winner_row = _scored.nsmallest(1, "dasa_distance").iloc[0]
    _result = _to_dict(_winner_row, "closest_miss")
    return _result


def pick_winner(df: pd.DataFrame, bounds: Dict[str, float]) -> Dict[str, Any]:
    """*pick_winner()* return the row that pushes both DASA coefficients deepest into the viable region under the operational rho ceiling.

    Three-tier fallback:

        - **Feasible** (at least one row clears rho + DASA + R1 + R2): pick the largest `_dc_margin`, which forces BOTH sigma_arch and eta_arch to sit inside their bounds. Tie-break by smallest `eps_e2e`. `winner_status = "feasible"`.
        - **Closest miss inside the rho ceiling** (rows passed rho but failed DASA / R1 / R2): pick the smallest `_dasa_distance` among rho-passing rows. `winner_status = "closest_miss"`.
        - **No rho-survivors** (every candidate exceeded the rho ceiling): pick the smallest `rho_max_active`; this is the configuration closest to operational feasibility. `winner_status = "no_rho_survivors"`.

    Args:
        df (pd.DataFrame): results frame from `run_search`.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        Dict[str, Any]: winning row as a flat dict, plus `winner_status`.
    """
    _full_pass = df[df["rho_pass"] & df["dasa_pass"] & df["r1_pass"] & df["r2_pass"]]
    if len(_full_pass) > 0:
        return _pick_best(_full_pass, bounds)

    _rho_survivors = df[df["rho_pass"]]
    if len(_rho_survivors) > 0:
        return _pick_closest_miss(_rho_survivors, bounds)

    # No candidate cleared the rho ceiling; report the configuration closest to it.
    _row = df.nsmallest(1, "rho_max_active").iloc[0]
    return _to_dict(_row, "no_rho_survivors")


# ---- yoly selection surface (the candidate cloud the winner is picked from) ----


def cloud_to_selection_surface(df: pd.DataFrame,
                               *,
                               tag: str = "req") -> Dict[str, np.ndarray]:
    """*cloud_to_selection_surface()* reshape a scored candidate frame into the requirement-frame `(sigma_arch, eta_arch)` cloud the DASA selection chart consumes.

    Maps the search frame's `sigma_arch` column to `\\sigma_{<tag>}` and `eta_arch` to `\\eta_{<tag>}`, the two coordinates in which the viable box (`sigma_R2` / `eta_R1` from `compute_bounds`) is defined. Only sigma and eta exist in this requirement frame, so no theta / phi keys are emitted (those belong to the sum-frame yoly chart). Rows whose sigma or eta is non-finite (the `inf` placeholders left by short-circuited candidates in `_score`) are dropped so the scatter axis ranges stay finite.

    The default `tag="req"` keeps these keys distinct from the sum-frame `"TAS"` clouds so the two frames never collide in a shared dict.

    Args:
        df (pd.DataFrame): scored candidate frame from `run_search` / `run_descent` (carries `sigma_arch` and `eta_arch` columns).
        tag (str): architecture subscript baked into the output keys. Defaults to `"req"`.

    Returns:
        Dict[str, np.ndarray]: `{f"\\sigma_{{{tag}}}": ndarray, f"\\eta_{{{tag}}}": ndarray}`, finite rows only.
    """
    _sigma = df["sigma_arch"].to_numpy(dtype=float)
    _eta = df["eta_arch"].to_numpy(dtype=float)
    _finite = np.isfinite(_sigma) & np.isfinite(_eta)
    return {
        f"\\sigma_{{{tag}}}": _sigma[_finite],
        f"\\eta_{{{tag}}}": _eta[_finite],
    }


def select_from_cloud(cloud_df: pd.DataFrame,
                      bounds: Dict[str, float]) -> Dict[str, Any]:
    """*select_from_cloud()* pick the candidate deepest inside the viable box from a coarse-grid cloud.

    Thin alias over `pick_winner` that names the intent: the coarse `run_search` cloud is the selection surface, and the winner is the cloud point deepest inside the DASA viable region (largest DC margin). The selection math is unchanged; this exists so the search consuming the cloud reads as one verb.

    Args:
        cloud_df (pd.DataFrame): pooled coarse-grid candidate cloud from `run_search`.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        Dict[str, Any]: the winning cloud row as a flat dict, plus `winner_status`.
    """
    return pick_winner(cloud_df, bounds)


# ---- DC-guided descent (derivative-free pattern search on the simplex) ----

# local-optimum tolerance: a neighbour must beat the current objective by more than this to be a step
_DESCENT_TOL = 1e-12
# meaningful-improvement threshold: steps that improve the best objective by less than this count
# toward the stall (early-stop) counter
_DESCENT_MIN_IMPROVE = 1e-9
# interior guard: a dispatch weight may not drop to or below this value during a step
_WEIGHT_FLOOR = 1e-12


def _centroid(k: int) -> Tuple[float, ...]:
    """*_centroid()* return the uniform `k`-weight tuple `(1/k, ..., 1/k)`.

    Args:
        k (int): number of pool services.

    Returns:
        Tuple[float, ...]: the simplex centroid for a `k`-service pool.
    """
    _w: List[float] = []
    for _ in range(k):
        _w.append(1.0 / k)
    return tuple(_w)


def _corner(k: int, idx: int, off: float = 0.05) -> Tuple[float, ...]:
    """*_corner()* return a near-corner `k`-weight tuple favouring service `idx`.

    Service `idx` gets `1 - (k-1)*off`; every other service gets the small interior offset `off` so the tuple stays strictly positive (a true corner would zero the others and break the `chi > 0` requirement).

    Args:
        k (int): number of pool services.
        idx (int): index of the favoured service.
        off (float): interior offset assigned to each non-favoured service.

    Returns:
        Tuple[float, ...]: a strictly-interior near-corner weight tuple.
    """
    _w: List[float] = []
    for _i in range(k):
        if _i == idx:
            _w.append(1.0 - (k - 1) * off)
        else:
            _w.append(off)
    return tuple(_w)


def _seed_weights(pools: Dict[str, List[str]]) -> List[Dict[str, Tuple[float, ...]]]:
    """*_seed_weights()* build the multi-start seeds for the descent.

    The centroid start plus the cross-product of each pool's near-corner starts (MAS corners x AS corners), every seed keeping DS at its singleton `(1.0,)`. Multi-start guards against the greedy descent settling in a single basin.

    Args:
        pools (Dict[str, List[str]]): per-type pool selection from Stage 1.

    Returns:
        List[Dict[str, Tuple[float, ...]]]: weight dicts keyed `M` / `A` / `D`.
    """
    _km = len(pools["MAS"])
    _ka = len(pools["AS"])
    _kd = len(pools["DS"])
    _wd = _centroid(_kd)

    _m_starts: List[Tuple[float, ...]] = [_centroid(_km)]
    for _i in range(_km):
        _m_starts.append(_corner(_km, _i))
    _a_starts: List[Tuple[float, ...]] = [_centroid(_ka)]
    for _i in range(_ka):
        _a_starts.append(_corner(_ka, _i))

    _seeds: List[Dict[str, Tuple[float, ...]]] = []
    for _wm in _m_starts:
        for _wa in _a_starts:
            _seeds.append({"M": _wm, "A": _wa, "D": _wd})
    return _seeds


def _simplex_neighbors(weights: Dict[str, Tuple[float, ...]],
                       step: float,
                       pools: Dict[str, List[str]]) -> List[Dict[str, Tuple[float, ...]]]:
    """*_simplex_neighbors()* enumerate the lattice neighbours of one weight point.

    For every multi-service pool and every ordered (donor, receiver) pair within it, move `step` mass from the donor to the receiver. Mass is conserved (the tuple stays on the simplex) and any move that would push the donor to `<= _WEIGHT_FLOOR` is dropped (the point stays strictly interior). Singleton pools (DS) contribute no neighbours.

    Args:
        weights (Dict[str, Tuple[float, ...]]): the current weight point keyed `M` / `A` / `D`.
        step (float): mass moved per neighbour.
        pools (Dict[str, List[str]]): per-type pool selection (sizes the per-pool simplex).

    Returns:
        List[Dict[str, Tuple[float, ...]]]: neighbouring weight dicts.
    """
    _neighbors: List[Dict[str, Tuple[float, ...]]] = []
    for _svc, _short in (("MAS", "M"), ("AS", "A"), ("DS", "D")):
        _k = len(pools[_svc])
        if _k < 2:
            continue
        _w = weights[_short]
        for _donor in range(_k):
            for _recv in range(_k):
                if _donor == _recv:
                    continue
                if _w[_donor] - step <= _WEIGHT_FLOOR:
                    continue
                _new = list(_w)
                _new[_donor] = _w[_donor] - step
                _new[_recv] = _w[_recv] + step
                _moved = dict(weights)
                _moved[_short] = tuple(_new)
                _neighbors.append(_moved)
    return _neighbors


def _descent_objective(row: Dict[str, Any], bounds: Dict[str, float]) -> float:
    """*_descent_objective()* collapse a scored candidate into one scalar to minimise.

    A monotone three-tier ladder so any feasible point beats any infeasible point, and any rho-passing point beats any rho-violating one:

        - rho-violating / unstable: `10 + rho_max_active` (drive back inside the operational region first).
        - rho-passing but DASA / R1 / R2 short: `1 + _dasa_distance` (push both coefficients toward their bounds).
        - fully feasible: `-_dc_margin + 1e-6 * eps_e2e` (push deepest inside the viable region, prefer reliability on ties).

    Args:
        row (Dict[str, Any]): one scored candidate from `_score`.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        float: the objective value (lower is better).
    """
    _sigma = float(row["sigma_arch"])
    _eta = float(row["eta_arch"])
    if not bool(row["rho_pass"]):
        _rma = float(row["rho_max_active"])
        if not np.isfinite(_rma):
            _rma = 1e3
        return 10.0 + _rma
    _feasible = (bool(row["dasa_pass"]) and bool(row["r1_pass"])
                 and bool(row["r2_pass"]))
    if not _feasible:
        return 1.0 + _dasa_distance(_sigma, _eta, bounds)
    return -_dc_margin(_sigma, _eta, bounds) + 1e-6 * float(row["eps_e2e"])


def _score_weights(catalogue: Dict[str, Any],
                   pools: Dict[str, List[str]],
                   weights: Dict[str, Tuple[float, ...]],
                   *,
                   lam_z: float,
                   tas_artifacts: List[ArtifactSpec],
                   template: ArtifactSpec,
                   bounds: Dict[str, float],
                   retry: Optional[Dict[str, int]],
                   rho_max: float) -> Dict[str, Any]:
    """*_score_weights()* build and score one weight point, tagging the row with its retry config.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection.
        weights (Dict[str, Tuple[float, ...]]): the weight point to score.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map for this candidate.
        rho_max (float): MAS / AS utilisation ceiling.

    Returns:
        Dict[str, Any]: the `_score` row plus a `retry` field.
    """
    _cfg = build_cfg(catalogue, pools, weights,
                     lam_z=lam_z,
                     tas_artifacts=tas_artifacts,
                     template=template,
                     retry=retry)
    _row = _score(_cfg, weights, bounds=bounds, rho_max=rho_max)
    if retry is None:
        _row["retry"] = {}
    else:
        _row["retry"] = dict(retry)
    return _row


def _descend_level(catalogue: Dict[str, Any],
                   pools: Dict[str, List[str]],
                   seed: Dict[str, Tuple[float, ...]],
                   *,
                   lam_z: float,
                   tas_artifacts: List[ArtifactSpec],
                   template: ArtifactSpec,
                   bounds: Dict[str, float],
                   retry: Optional[Dict[str, int]],
                   step: float,
                   rho_max: float,
                   patience: int,
                   max_iters: int,
                   time_budget_s: Optional[float]) -> Dict[str, Any]:
    """*_descend_level()* run one greedy coordinate-exchange descent from `seed` at a single step size.

    At each iteration every lattice neighbour (`_simplex_neighbors`) is scored and the best-improving one becomes the new point. The descent halts at the first of: a local optimum (no neighbour beats the current objective by more than `_DESCENT_TOL`), `patience` consecutive steps that fail to improve the best objective by `_DESCENT_MIN_IMPROVE` (early stop on stall), the `time_budget_s` wall-clock cap, or the `max_iters` safety net.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection.
        seed (Dict[str, Tuple[float, ...]]): starting weight point.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map held fixed for this descent.
        step (float): mass moved per neighbour.
        rho_max (float): MAS / AS utilisation ceiling.
        patience (int): consecutive non-improving steps tolerated before early stop.
        max_iters (int): hard iteration cap.
        time_budget_s (Optional[float]): optional per-descent wall-clock cap in seconds.

    Returns:
        Dict[str, Any]: the best scored row visited, tagged with its `retry` config.
    """
    _cur_w = seed
    _cur_row = _score_weights(catalogue, pools, _cur_w,
                              lam_z=lam_z, tas_artifacts=tas_artifacts,
                              template=template, bounds=bounds,
                              retry=retry, rho_max=rho_max)
    _cur_obj = _descent_objective(_cur_row, bounds)
    _best_row = _cur_row
    _best_obj = _cur_obj
    _stall = 0
    _t0 = time.perf_counter()

    for _ in range(max_iters):
        _neighbors = _simplex_neighbors(_cur_w, step, pools)
        if len(_neighbors) == 0:
            break
        _best_nb_w: Optional[Dict[str, Tuple[float, ...]]] = None
        _best_nb_row: Optional[Dict[str, Any]] = None
        _best_nb_obj = float("inf")
        for _nb in _neighbors:
            _nb_row = _score_weights(catalogue, pools, _nb,
                                     lam_z=lam_z, tas_artifacts=tas_artifacts,
                                     template=template, bounds=bounds,
                                     retry=retry, rho_max=rho_max)
            _nb_obj = _descent_objective(_nb_row, bounds)
            if _nb_obj < _best_nb_obj:
                _best_nb_obj = _nb_obj
                _best_nb_w = _nb
                _best_nb_row = _nb_row
        # local optimum: no neighbour strictly improves the current objective
        if _best_nb_w is None or _best_nb_row is None:
            break
        if _best_nb_obj > _cur_obj - _DESCENT_TOL:
            break
        _prev_best = _best_obj
        _cur_w = _best_nb_w
        _cur_row = _best_nb_row
        _cur_obj = _best_nb_obj
        if _cur_obj < _best_obj - _DESCENT_TOL:
            _best_obj = _cur_obj
            _best_row = _cur_row
        # early stop on stall: count steps that barely move the best objective
        if _best_obj < _prev_best - _DESCENT_MIN_IMPROVE:
            _stall = 0
        else:
            _stall = _stall + 1
        if _stall >= patience:
            break
        if time_budget_s is not None and (time.perf_counter() - _t0) > time_budget_s:
            break

    return _best_row


def _descend(catalogue: Dict[str, Any],
             pools: Dict[str, List[str]],
             seed: Dict[str, Tuple[float, ...]],
             *,
             lam_z: float,
             tas_artifacts: List[ArtifactSpec],
             template: ArtifactSpec,
             bounds: Dict[str, float],
             retry: Optional[Dict[str, int]],
             steps: List[float],
             rho_max: float,
             patience: int,
             max_iters: int,
             time_budget_s: Optional[float]) -> Dict[str, Any]:
    """*_descend()* run a coarse-to-fine descent from `seed` over a step schedule.

    Runs one `_descend_level` per entry in `steps`, carrying the optimum forward as the seed for the next (finer) level. A coarse first level (e.g. 0.01) reaches the basin in few iterations; a fine final level (e.g. 0.001) gives weight resolution near the optimum without traversing the whole simplex finely.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection.
        seed (Dict[str, Tuple[float, ...]]): starting weight point.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map held fixed for this descent.
        steps (List[float]): step sizes from coarse to fine.
        rho_max (float): MAS / AS utilisation ceiling.
        patience (int): consecutive non-improving steps tolerated before early stop.
        max_iters (int): hard iteration cap per level.
        time_budget_s (Optional[float]): optional per-level wall-clock cap in seconds.

    Returns:
        Dict[str, Any]: the best scored row from the final (finest) level.
    """
    _cur_seed = seed
    _best_row = _score_weights(catalogue, pools, _cur_seed,
                               lam_z=lam_z, tas_artifacts=tas_artifacts,
                               template=template, bounds=bounds,
                               retry=retry, rho_max=rho_max)
    for _step in steps:
        _best_row = _descend_level(catalogue, pools, _cur_seed,
                                   lam_z=lam_z, tas_artifacts=tas_artifacts,
                                   template=template, bounds=bounds,
                                   retry=retry, step=_step, rho_max=rho_max,
                                   patience=patience, max_iters=max_iters,
                                   time_budget_s=time_budget_s)
        _cur_seed = {
            "M": tuple(_best_row["w_M"]),
            "A": tuple(_best_row["w_A"]),
            "D": tuple(_best_row["w_D"]),
        }
    return _best_row


def run_descent(catalogue: Dict[str, Any],
                pools: Dict[str, List[str]],
                lam_z: float,
                tas_artifacts: List[ArtifactSpec],
                template: ArtifactSpec,
                *,
                bounds: Dict[str, float],
                retry_grid: Optional[List[Optional[Dict[str, int]]]] = None,
                step: float = 0.001,
                step_schedule: Optional[List[float]] = None,
                rho_max: float = _DEFAULT_RHO_MAX,
                seeds: Optional[List[Dict[str, Tuple[float, ...]]]] = None,
                patience: int = 20,
                max_iters: int = 10000,
                time_budget_s: Optional[float] = None,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """*run_descent()* DC-guided descent over weights for every retry config and seed.

    The outer loop walks the discrete `retry_grid`; the inner loop runs a multi-start coordinate-exchange descent (`_descend`) over the continuous dispatch weights for each retry config. The result frame holds one row per (retry config, seed) local optimum, with the same schema as `run_search` plus a `retry` column, so `pick_winner` consumes it unchanged.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection from Stage 1.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry_grid (Optional[List[Optional[Dict[str, int]]]]): retry configs to try; each is a per-handler depth map (or `None` / `{}` for selection-only). Defaults to selection-only only.
        step (float): single-level step size used when `step_schedule` is None (default 0.001).
        step_schedule (Optional[List[float]]): coarse-to-fine step sizes (e.g. `[0.01, 0.001]`); when given, overrides `step`. Each descent runs once per level, carrying the optimum forward.
        rho_max (float): MAS / AS utilisation ceiling.
        seeds (Optional[List[Dict[str, Tuple[float, ...]]]]): multi-start seeds; defaults to `_seed_weights(pools)`.
        patience (int): consecutive non-improving steps tolerated before early stop.
        max_iters (int): hard iteration cap per descent.
        time_budget_s (Optional[float]): optional per-descent wall-clock cap in seconds.
        progress_callback (Optional[Callable[[int, int], None]]): called after each descent with `(i_completed, total)`.

    Returns:
        pd.DataFrame: one row per (retry config, seed) local optimum.
    """
    if retry_grid is None:
        _retry_grid: List[Optional[Dict[str, int]]] = [None]
    else:
        _retry_grid = retry_grid
    if seeds is None:
        _seeds = _seed_weights(pools)
    else:
        _seeds = seeds
    if step_schedule is None:
        _steps = [step]
    else:
        _steps = step_schedule
    _total = len(_retry_grid) * len(_seeds)

    _rows: List[Dict[str, Any]] = []
    _i = 0
    for _retry in _retry_grid:
        for _seed in _seeds:
            _best = _descend(catalogue, pools, _seed,
                             lam_z=lam_z, tas_artifacts=tas_artifacts,
                             template=template, bounds=bounds,
                             retry=_retry, steps=_steps, rho_max=rho_max,
                             patience=patience, max_iters=max_iters,
                             time_budget_s=time_budget_s)
            _rows.append(_best)
            _i = _i + 1
            if progress_callback is not None:
                progress_callback(_i, _total)
    _df = pd.DataFrame(_rows)
    return _df


# ---- scipy cross-validation (independent optimiser over the same objective) ----

# penalty bands mirroring _descent_objective: unstable worst, then rho-violating, so the
# scipy minimiser is driven back into the operational region before optimising the coefficients
_SCIPY_UNSTABLE_PENALTY = 1e6
_SCIPY_RHO_PENALTY = 1e3


def _x_to_weights(x: np.ndarray,
                  pools: Dict[str, List[str]]) -> Dict[str, Tuple[float, ...]]:
    """*_x_to_weights()* map a free-variable vector to per-pool simplex weights.

    Each multi-service pool contributes `k - 1` free coordinates; the last weight is `1 - sum(free)` so the tuple stays on the simplex. Singleton pools (DS) consume no coordinates and are fixed at `(1.0,)`. Negative or out-of-range free values are clamped so the scored point is always a valid dispatch.

    Args:
        x (np.ndarray): flat free-variable vector (length `sum_k (k - 1)` over multi-service pools).
        pools (Dict[str, List[str]]): per-type pool selection (sizes each pool's simplex).

    Returns:
        Dict[str, Tuple[float, ...]]: weights keyed `M` / `A` / `D`.
    """
    _weights: Dict[str, Tuple[float, ...]] = {}
    _idx = 0
    for _svc, _short in (("MAS", "M"), ("AS", "A"), ("DS", "D")):
        _k = len(pools[_svc])
        if _k < 2:
            _weights[_short] = (1.0,)
            continue
        _free: List[float] = []
        for _j in range(_k - 1):
            _v = float(x[_idx + _j])
            if _v < 0.0:
                _v = 0.0
            _free.append(_v)
        _idx = _idx + (_k - 1)
        _last = 1.0 - sum(_free)
        if _last < 0.0:
            _last = 0.0
        _full = list(_free)
        _full.append(_last)
        _total = sum(_full)
        if _total <= 0.0:
            _norm = _centroid(_k)
        else:
            _norm = tuple(_w / _total for _w in _full)
        _weights[_short] = _norm
    return _weights


def _chebyshev_ratio(row: Dict[str, Any], bounds: Dict[str, float]) -> float:
    """*_chebyshev_ratio()* the larger of the two normalised DASA coordinates.

    `max(sigma_arch / sigma_R2, eta_arch / eta_R1)`: below 1 means the candidate sits inside both bounds (feasible on the yoly axes); minimising it pushes the worse of the R1 / R2 coordinates toward the origin. This is the continuous bi-objective scalarisation the scipy minimiser optimises, matching the custom descent's `_dc_margin` (which is `1 - chebyshev_ratio` on the binding axis).

    Args:
        row (Dict[str, Any]): one scored candidate from `_score`.
        bounds (Dict[str, float]): output of `compute_bounds`.

    Returns:
        float: the worst normalised coordinate.
    """
    _s = float(row["sigma_arch"]) / bounds["sigma_R2"]
    _e = float(row["eta_arch"]) / bounds["eta_R1"]
    if _s > _e:
        return _s
    return _e


def _scipy_cost(x: np.ndarray,
                catalogue: Dict[str, Any],
                pools: Dict[str, List[str]],
                *,
                lam_z: float,
                tas_artifacts: List[ArtifactSpec],
                template: ArtifactSpec,
                bounds: Dict[str, float],
                retry: Optional[Dict[str, int]],
                rho_max: float) -> float:
    """*_scipy_cost()* scalar cost a scipy minimiser drives down for one free-variable vector.

    Builds the candidate, scores it analytically, and returns the Chebyshev ratio with the same penalty ladder as `_descent_objective`: unstable configs get `_SCIPY_UNSTABLE_PENALTY`, rho-violating configs get `_SCIPY_RHO_PENALTY + rho_max_active`, and operational configs return `_chebyshev_ratio`.

    Args:
        x (np.ndarray): free-variable vector decoded by `_x_to_weights`.
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map for this candidate.
        rho_max (float): MAS / AS utilisation ceiling.

    Returns:
        float: the scalar cost (lower is better).
    """
    _weights = _x_to_weights(x, pools)
    _row = _score_weights(catalogue, pools, _weights,
                          lam_z=lam_z, tas_artifacts=tas_artifacts,
                          template=template, bounds=bounds,
                          retry=retry, rho_max=rho_max)
    if not bool(_row["stable"]):
        return _SCIPY_UNSTABLE_PENALTY
    if not bool(_row["rho_pass"]):
        return _SCIPY_RHO_PENALTY + float(_row["rho_max_active"])
    return _chebyshev_ratio(_row, bounds)


def _optimize_scipy(catalogue: Dict[str, Any],
                    pools: Dict[str, List[str]],
                    *,
                    lam_z: float,
                    tas_artifacts: List[ArtifactSpec],
                    template: ArtifactSpec,
                    bounds: Dict[str, float],
                    retry: Optional[Dict[str, int]],
                    rho_max: float,
                    starts: List[Dict[str, Tuple[float, ...]]],
                    method: str,
                    maxiter: int) -> Dict[str, Any]:
    """*_optimize_scipy()* minimise the bi-objective cost with scipy from several starts for one retry config.

    Runs `scipy.optimize.minimize` from each start in `starts`, keeps the lowest-cost result, and returns the corresponding scored row tagged with its `retry`. The simplex is handled by the `_x_to_weights` reparametrisation; the per-pool `0 <= free` and `sum(free) <= 1` interior is supplied as inequality constraints for constraint-aware methods (e.g. COBYLA).

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry (Optional[Dict[str, int]]): per-handler retry-depth map held fixed for this optimisation.
        rho_max (float): MAS / AS utilisation ceiling.
        starts (List[Dict[str, Tuple[float, ...]]]): weight points to seed the minimiser from.
        method (str): scipy method name (e.g. `"COBYLA"`).
        maxiter (int): per-start iteration cap passed to scipy.

    Returns:
        Dict[str, Any]: the best scored row found, tagged with its `retry`.
    """
    _cons = _simplex_constraints(pools)

    def _cost(_x: np.ndarray) -> float:
        return _scipy_cost(_x, catalogue, pools,
                           lam_z=lam_z, tas_artifacts=tas_artifacts,
                           template=template, bounds=bounds,
                           retry=retry, rho_max=rho_max)

    _best_row: Optional[Dict[str, Any]] = None
    _best_cost = float("inf")
    for _start in starts:
        _x0 = _weights_to_x(_start, pools)
        _res = minimize(_cost, _x0, method=method,
                        constraints=_cons, options={"maxiter": maxiter})
        _cand_w = _x_to_weights(np.asarray(_res.x), pools)
        _cand_row = _score_weights(catalogue, pools, _cand_w,
                                   lam_z=lam_z, tas_artifacts=tas_artifacts,
                                   template=template, bounds=bounds,
                                   retry=retry, rho_max=rho_max)
        _cand_cost = _cost(np.asarray(_res.x))
        if _cand_cost < _best_cost:
            _best_cost = _cand_cost
            _best_row = _cand_row
    if _best_row is None:
        _fallback = _score_weights(catalogue, pools, starts[0],
                                   lam_z=lam_z, tas_artifacts=tas_artifacts,
                                   template=template, bounds=bounds,
                                   retry=retry, rho_max=rho_max)
        return _fallback
    return _best_row


def _weights_to_x(weights: Dict[str, Tuple[float, ...]],
                  pools: Dict[str, List[str]]) -> np.ndarray:
    """*_weights_to_x()* pack per-pool weights into the free-variable vector `_x_to_weights` decodes.

    Args:
        weights (Dict[str, Tuple[float, ...]]): weights keyed `M` / `A` / `D`.
        pools (Dict[str, List[str]]): per-type pool selection.

    Returns:
        np.ndarray: the free-variable vector (drops the last weight of each multi-service pool).
    """
    _x: List[float] = []
    for _svc, _short in (("MAS", "M"), ("AS", "A"), ("DS", "D")):
        _k = len(pools[_svc])
        if _k < 2:
            continue
        for _j in range(_k - 1):
            _x.append(weights[_short][_j])
    return np.asarray(_x, dtype=float)


def _simplex_constraints(pools: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """*_simplex_constraints()* inequality constraints keeping each pool's free vars on its simplex.

    For every multi-service pool: each free coordinate `>= 0` and `1 - sum(free) >= 0` (the implied last weight stays non-negative). Returned in the scipy `constraints` dict form for constraint-aware methods.

    Args:
        pools (Dict[str, List[str]]): per-type pool selection.

    Returns:
        List[Dict[str, Any]]: scipy inequality-constraint dicts.
    """
    _cons: List[Dict[str, Any]] = []
    _offset = 0
    for _svc in ("MAS", "AS", "DS"):
        _k = len(pools[_svc])
        if _k < 2:
            continue
        _lo = _offset
        _hi = _offset + (_k - 1)
        for _j in range(_lo, _hi):
            _cons.append({"type": "ineq", "fun": _make_lower_bound(_j)})
        _cons.append({"type": "ineq", "fun": _make_sum_bound(_lo, _hi)})
        _offset = _hi
    return _cons


def _make_lower_bound(j: int) -> Callable[[np.ndarray], float]:
    """*_make_lower_bound()* build a `x[j] >= 0` scipy inequality function (binds `j` at definition)."""
    def _fun(_x: np.ndarray) -> float:
        return float(_x[j])
    return _fun


def _make_sum_bound(lo: int, hi: int) -> Callable[[np.ndarray], float]:
    """*_make_sum_bound()* build a `1 - sum(x[lo:hi]) >= 0` scipy inequality function (binds the slice at definition)."""
    def _fun(_x: np.ndarray) -> float:
        return float(1.0 - sum(_x[lo:hi]))
    return _fun


def run_scipy_search(catalogue: Dict[str, Any],
                     pools: Dict[str, List[str]],
                     lam_z: float,
                     tas_artifacts: List[ArtifactSpec],
                     template: ArtifactSpec,
                     *,
                     bounds: Dict[str, float],
                     retry_grid: Optional[List[Optional[Dict[str, int]]]] = None,
                     rho_max: float = _DEFAULT_RHO_MAX,
                     seeds: Optional[List[Dict[str, Tuple[float, ...]]]] = None,
                     method: str = "COBYLA",
                     maxiter: int = 500,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> pd.DataFrame:
    """*run_scipy_search()* independent scipy optimisation mirroring `run_descent`, for external validation.

    Runs `scipy.optimize.minimize` (default COBYLA) over the dispatch weights for every retry config, minimising the same bi-objective the custom descent does (the Chebyshev ratio over the yoly coefficients). The frame schema matches `run_descent` (one row per retry config plus a `retry` column), so `pick_winner` consumes it identically and the notebook can compare the scipy winner against the custom-descent winner. Agreement between the two independent optimisers cross-validates the custom search and demonstrates that the DASA method integrates with standard optimisation tooling.

    Args:
        catalogue (Dict[str, Any]): loaded catalogue.
        pools (Dict[str, List[str]]): per-type pool selection from Stage 1.
        lam_z (float): external arrival rate.
        tas_artifacts (List[ArtifactSpec]): the six TAS stage artifacts.
        template (ArtifactSpec): third-party clone template.
        bounds (Dict[str, float]): output of `compute_bounds`.
        retry_grid (Optional[List[Optional[Dict[str, int]]]]): retry configs to try; defaults to selection-only.
        rho_max (float): MAS / AS utilisation ceiling.
        seeds (Optional[List[Dict[str, Tuple[float, ...]]]]): minimiser starts; defaults to the centroid plus one near-corner per pool.
        method (str): scipy method name (default `"COBYLA"`, derivative-free + constraint-aware).
        maxiter (int): per-start iteration cap.
        progress_callback (Optional[Callable[[int, int], None]]): called after each retry config with `(i_completed, total)`.

    Returns:
        pd.DataFrame: one row per retry config (the best scipy optimum), with the `run_descent` schema plus `retry`.
    """
    if retry_grid is None:
        _retry_grid: List[Optional[Dict[str, int]]] = [None]
    else:
        _retry_grid = retry_grid
    if seeds is None:
        _seeds = _scipy_starts(pools)
    else:
        _seeds = seeds
    _total = len(_retry_grid)

    _rows: List[Dict[str, Any]] = []
    _i = 0
    for _retry in _retry_grid:
        _best = _optimize_scipy(catalogue, pools,
                                lam_z=lam_z, tas_artifacts=tas_artifacts,
                                template=template, bounds=bounds,
                                retry=_retry, rho_max=rho_max,
                                starts=_seeds, method=method, maxiter=maxiter)
        _rows.append(_best)
        _i = _i + 1
        if progress_callback is not None:
            progress_callback(_i, _total)
    _df = pd.DataFrame(_rows)
    return _df


def _scipy_starts(pools: Dict[str, List[str]]) -> List[Dict[str, Tuple[float, ...]]]:
    """*_scipy_starts()* a small start set for the scipy minimiser (centroid + the lowest-index corner per pool).

    Fewer starts than the custom descent's multi-start set: COBYLA is a local minimiser that reaches the basin quickly, so the centroid plus a single near-corner per pool is enough to validate the descent without a long run.

    Args:
        pools (Dict[str, List[str]]): per-type pool selection.

    Returns:
        List[Dict[str, Tuple[float, ...]]]: minimiser start points.
    """
    _km = len(pools["MAS"])
    _ka = len(pools["AS"])
    _kd = len(pools["DS"])
    _wd = _centroid(_kd)
    _starts: List[Dict[str, Tuple[float, ...]]] = [
        {"M": _centroid(_km), "A": _centroid(_ka), "D": _wd},
        {"M": _corner(_km, 0), "A": _corner(_ka, 0), "D": _wd},
    ]
    return _starts
