# -*- coding: utf-8 -*-
"""
Module dimensional.py
=====================

Dimensional method orchestrator for the CS-01 TAS case study. Walks the 13 (or 16) artifacts in a resolved `NetCfg`, builds a PyDASA `AnalysisEngine` per artifact, derives Pi-groups via Buckingham's theorem, applies the operationally meaningful coefficient specs (theta, sigma, eta, phi) from `data/config/method/dimensional.json`, runs a symbolic sensitivity pass at the variable means, and emits a PyDASA-style JSON.

The method also returns the operational layer (`nodes`, `network`, `requirements`) inherited from the Jackson solve that feeds the dimensional refresh, mirroring the analytic + stochastic method contracts; the persisted JSON envelope carries the same `network` + `nodes` + companion `requirements.json` so the cross-method comparison notebook can read all three methods from disk uniformly.

Public API:
    - `run(adp, prf, scn, wrt)` resolves the profile + method config, loops artifacts, and returns per-artifact `pi_groups` / `coefficients` / `sensitivity` blocks plus operational `nodes` / `network` / `requirements`.

Private helpers:
    - `_analyse_artifact(artifact, schema, ...)` full DA workflow on one artifact.
    - `_write_results(cfg, method_cfg, results)` serialises the envelope to `data/results/dimensional/<scenario>/<profile>.json`.

CLI::

    python -m src.methods.dimensional --adaptation baseline
    python -m src.methods.dimensional --adaptation s1 --profile opti
    python -m src.methods.dimensional  # uses _setpoint
"""
# native python modules
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

# data types
from typing import Any, Dict, Optional

# pydasa library
from pydasa.dimensional.vaschy import Schema

# scientific stack
import numpy as np
import pandas as pd

# local modules
from src.analytic import aggregate_net, check_reqs, solve_network
from src.dimensional import (add_white_noise,
                             analyse_symbolic,
                             build_engine,
                             build_schema,
                             derive_coefs)
from src.io import ArtifactSpec, NetCfg, load_method_cfg, load_profile


_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "data" / "results" / "dimensional"

# synthetic absorbing failure-sink node; excluded from the per-artifact dimensional analysis
# (it has no service semantics / no OUTPUT variable for PyDASA to build a matrix from)
_FAIL_KEY = "FAIL_{1}"


def _set_var_value(vars_block: Dict[str, dict], prefix: str, val: float) -> None:
    """*_set_var_value()* overwrite every setpoint/mean field of the first variable whose symbol starts with `prefix`.

    Args:
        vars_block (Dict[str, dict]): an artifact's PyDASA Variable-dict block (mutated in place).
        prefix (str): LaTeX-symbol prefix including the artifact subscript (e.g. `\\mu_{TAS_{1}}`).
        val (float): replacement value written to `_setpoint`, `_std_setpoint`, `_mean`, `_std_mean`, and `_data`.
    """
    for _sym, _var in vars_block.items():
        if _sym.startswith(prefix):
            _var["_setpoint"] = val
            _var["_std_setpoint"] = val
            _var["_mean"] = val
            _var["_std_mean"] = val
            _var["_data"] = [val]
            return


def _perturb_indep(cfg: NetCfg,
                   *,
                   noise: Any,
                   rng: np.random.Generator) -> NetCfg:
    """*_perturb_indep()* return a new `NetCfg` with the M/M/c/K independent variables disturbed by seeded white noise.

    The independent inputs are lambda (arrival), mu (service rate), and epsilon (failure rate). For each non-FAIL artifact the triple `[lambda_z, mu, epsilon]` is perturbed via `src.dimensional.add_white_noise`, then a fresh `ArtifactSpec` is built with the noisy `lambda_z` and a deep-copied `vars` block carrying the noisy mu / epsilon. The dependent metrics (L, Lq, W, Wq, chi, rho) are left for `solve_network` to recompute from the perturbed inputs, so coupling is preserved (theta == phi, eta tied to the same noisy mu / chi). The input `cfg` is not mutated.

    Args:
        cfg (NetCfg): resolved network configuration to perturb.
        noise (Any): relative noise level forwarded to `add_white_noise`; scalar (same level on all three variables) or a 3-array `[lambda, mu, epsilon]` for per-variable levels.
        rng (np.random.Generator): seeded generator supplying the draws.

    Returns:
        NetCfg: a new configuration with disturbed independent variables; artifacts in the same positional order.
    """
    _new_arts = []
    for _a in cfg.artifacts:
        if _a.key == _FAIL_KEY:
            _new_arts.append(_a)
            continue
        _sub = _a.format_sub()
        _triple = np.array([_a.lambda_z, _a.mu, _a.epsilon], dtype=float)
        _noisy = add_white_noise(_triple, noise, rng=rng)
        _vars_copy = deepcopy(_a.vars)
        _set_var_value(_vars_copy, f"\\mu_{{{_sub}}}", float(_noisy[1]))
        _set_var_value(_vars_copy, f"\\epsilon_{{{_sub}}}", float(_noisy[2]))
        _new_arts.append(replace(_a, lambda_z=float(_noisy[0]), vars=_vars_copy))
    return replace(cfg, artifacts=_new_arts)


def _refresh_artifact_vars(cfg: NetCfg, nds) -> None:
    """*_refresh_artifact_vars()* overwrite per-artifact operational variables with values from the analytic solve.

    The dimensional method runs on `cfg.artifacts`, whose `vars` come straight from the profile JSON. For nodes that the profile carries as placeholders (typical when a node is inactive under one adaptation but active under another), the placeholder propagates into PyDASA's `_std_setpoint`/`_mean` and produces wrong coefficient setpoints. This refresh replaces lambda, chi, L, Lq, W, Wq with the values just computed by `solve_network()` so PyDASA reads the actual operating point.

    Args:
        cfg (NetCfg): resolved network configuration (artifacts list is mutated in place).
        nds (pd.DataFrame): per-node metrics frame from `src.analytic.solve_network`.
    """
    for _i, _a in enumerate(cfg.artifacts):
        _row = nds.iloc[_i]
        _sub = _a.format_sub()
        _eps = _a.vars.get(f"\\epsilon_{{{_sub}}}", {}).get("_setpoint", 0.0)
        _d_req = _a.vars.get(f"d_{{{_sub}}}", {}).get("_setpoint", 0.0)
        # chi = (1 + eps) * lambda: the station throughput carries the reprocessed error load (self-loop retries), so eta (effective-yield) reflects the work of handling errors, not just successes. System-level availability uses chi_out = (1 - eps_e2e) * lambda_z, computed separately in analytic.metrics.
        _updates = {
            f"\\lambda_{{{_sub}}}": float(_row["lambda"]),
            f"L_{{{_sub}}}": float(_row["L"]),
            f"L_{{q, {_sub}}}": float(_row["Lq"]),
            f"W_{{{_sub}}}": float(_row["W"]),
            f"W_{{q, {_sub}}}": float(_row["Wq"]),
            f"\\chi_{{{_sub}}}": float(_row["lambda"]) * (1 + _eps),
            f"M_{{act_{{{_sub}}}}}": float(_row["L"]) * _d_req,
        }
        for _sym, _val in _updates.items():
            if _sym in _a.vars:
                _a.vars[_sym]["_setpoint"] = _val
                _a.vars[_sym]["_std_setpoint"] = _val
                _a.vars[_sym]["_mean"] = _val
                _a.vars[_sym]["_std_mean"] = _val
                _a.vars[_sym]["_data"] = [_val]


def _analyse_artifact(artifact: ArtifactSpec,
                      schema: Schema,
                      coef_specs: list[dict[str, Any]],
                      sens_cfg: dict[str, Any]) -> Dict[str, Any]:
    """*_analyse_artifact()* run the full DA workflow on one artifact.

    Builds an engine from the artifact's Variable dict, derives Pi-groups, applies the named coefficient specs, evaluates setpoints, and runs one symbolic sensitivity pass.

    Args:
        artifact (ArtifactSpec): one resolved artifact from the `NetCfg`.
        schema (Schema): framework schema built once for the full run.
        coef_specs (list[dict[str, Any]]): coefficient specs from the method config.
        sens_cfg (dict[str, Any]): sensitivity sub-config keyed by `val_type` and `cat`.

    Returns:
        Dict[str, Any]: per-artifact block with `name`, `type`, `lambda_z`, `L_z`, `pi_groups`, `coefficients`, `sensitivity`.
    """
    _eng = build_engine(artifact.key, artifact.vars, schema)
    _eng.run_analysis()
    # raw Pi-groups need an explicit setpoint pass; PyDASA leaves them lazy after run_analysis
    _pi_keys = [_k for _k in _eng.coefficients.keys() if _k.startswith("\\Pi_")]
    for _k in _pi_keys:
        _eng.coefficients[_k].calculate_setpoint()
    _der = derive_coefs(_eng, coef_specs, artifact_key=artifact.key)
    for _c in _der.values():
        _c.calculate_setpoint()
    _sens = analyse_symbolic(_eng, schema,
                             val_type=sens_cfg.get("val_type", "mean"),
                             cat=sens_cfg.get("cat", "SYM"))
    _pi_out = {_k: {"expr": str(_eng.coefficients[_k].pi_expr),
                    "setpoint": _eng.coefficients[_k].setpoint,
                    "var_dims": dict(_eng.coefficients[_k].var_dims)}
               for _k in _pi_keys}

    _co_out = {_s: {"expr": str(_c.pi_expr),
                    "setpoint": _c.setpoint,
                    "var_dims": dict(_c.var_dims),
                    "name": _c.name,
                    "description": _c.description}
               for _s, _c in _der.items()}

    return {
        "name": artifact.name,
        "type": artifact.type_,
        "lambda_z": artifact.lambda_z,
        "L_z": artifact.L_z,
        "pi_groups": _pi_out,
        "coefficients": _co_out,
        "sensitivity": _sens,
    }


def run_from_netcfg(cfg: NetCfg,
                    *,
                    wrt: bool = False,
                    method_cfg: Optional[Dict[str, Any]] = None,
                    noise: Any = None,
                    noise_scope: str = "all") -> Dict[str, Any]:
    """*run_from_netcfg()* analyse every artifact of an in-memory `NetCfg` dimensionally.

    This is the side-effect-free core that `run()` wraps; it accepts a pre-built `NetCfg` (typically produced by `src.dimensional.search.build_netcfg` for in-memory sweeps) and returns the same result shape as `run()` without re-reading disk. Callers that already have a profile loaded (the search loop is the primary example) avoid the per-iteration `load_profile(...)` cost.

    Args:
        cfg (NetCfg): resolved network configuration.
        wrt (bool): if True, write JSON artifact to `data/results/dimensional/<scenario>/<profile>.json`. Defaults to False (in-memory callers rarely want disk writes).
        method_cfg (Optional[Dict[str, Any]]): inline override for the dimensional method parameters (`fdus`, `coefficients`, `sensitivity`, ...). When `None`, loads `data/config/method/dimensional.json`.
        noise (Any): when set (a float like 0.05, or a 3-array `[lambda, mu, epsilon]`), disturb the M/M/c/K independent variables with seeded white noise before solving, so the derived coefficients carry that uncertainty. The generator is seeded from the method config's `seed`, keeping the run reproducible. Defaults to None (no noise; bit-identical to the clean path).
        noise_scope (str): how far the noise propagates when `noise` is set. `"all"` (default) perturbs the inputs before the operational solve too, so `nodes` / `network` / `requirements` (the R1 / R2 verdict) are noisy as well; this is the winner-robustness readout path used by `05-search.ipynb` section 11. `"coefficients"` keeps the operational layer on the clean `cfg` (so the persisted verdict still matches the analytic + stochastic methods for cross-method congruence) and applies the noise only to the dimensional coefficient pass, giving a robust DC cloud with a clean verdict. Ignored when `noise is None`.

    Returns:
        Dict[str, Any]: result dict with keys `config`, `method_config`, `artifacts`, `nodes`, `network`, `requirements`, `paths`.
    """
    if method_cfg is not None:
        _mcfg = method_cfg
    else:
        _mcfg = load_method_cfg("dimensional")
    if noise_scope not in ("all", "coefficients"):
        _msg = f"noise_scope must be 'all' or 'coefficients', got {noise_scope!r}"
        raise ValueError(_msg)
    # Choose which configuration drives the operational solve (nodes / network / verdict) versus the dimensional coefficient pass. Under "coefficients" scope the operational layer stays on the clean cfg (verdict matches analytic / stochastic) and only the coefficients carry the seeded input uncertainty; under "all" scope both are perturbed (the original behaviour).
    _cfg_solve = cfg
    _cfg_coef = cfg
    if noise is not None:
        _rng = np.random.default_rng(_mcfg.get("seed"))
        _cfg_noisy = _perturb_indep(cfg, noise=noise, rng=_rng)
        _cfg_coef = _cfg_noisy
        if noise_scope == "all":
            _cfg_solve = _cfg_noisy
    # Solve the analytic Jackson network for the operational layer, then refresh the artifact operational variables (lambda, chi, L, Lq, W, Wq) for the coefficient pass. Under coefficients-only noise the coefficient pass runs on a separate noisy solve so the DCs propagate the uncertainty while the operational `_nds` stays clean.
    _nds = solve_network(_cfg_solve)
    if _cfg_coef is _cfg_solve:
        _refresh_artifact_vars(_cfg_coef, _nds)
    else:
        _nds_coef = solve_network(_cfg_coef)
        _refresh_artifact_vars(_cfg_coef, _nds_coef)
    # one schema reused across every artifact: PyDASA validates the framework once
    _sch = build_schema(_mcfg["fdus"])
    _arts: Dict[str, Dict[str, Any]] = {}
    for _a in _cfg_coef.artifacts:
        # the synthetic FAIL sink carries no service semantics (no OUTPUT variable); skip its DA pass
        if _a.key == _FAIL_KEY:
            continue
        _arts[_a.key] = _analyse_artifact(_a, _sch,
                                          _mcfg["coefficients"],
                                          _mcfg["sensitivity"])
    # operational aggregate + R1 / R2 verdict so the dimensional notebook reads `nets`/`reqs` like the analytic + stochastic ones do (avoids re-solving Jackson in the notebook for the sigma_arch / eta_arch derivation per dasa-modelling.md Block 4).
    _lam_z = float(_cfg_solve.build_lam_z_vec()[0])
    # availability from the loss-network flow (routing carries the give-up leaks to the FAIL sink)
    _net = aggregate_net(_nds, lam_z=_lam_z, routing=_cfg_solve.routing)
    _req = check_reqs(_nds, lam_z=_lam_z, routing=_cfg_solve.routing)
    _paths: Dict[str, str] = {}
    if wrt:
        # persist the clean operational layer (nodes / network / verdict / lambda_z / routing) so cross-method readers stay congruent; the artifacts block carries whatever noise the coefficient pass applied
        _paths = _write_results(_cfg_solve, _mcfg, _arts, _nds, _net, _req)

    return {
        "config": _cfg_coef,
        "method_config": _mcfg,
        "artifacts": _arts,
        "nodes": _nds,
        "network": _net,
        "requirements": _req,
        "paths": _paths,
    }


def run(adp: Optional[str] = None,
        prf: Optional[str] = None,
        scn: Optional[str] = None,
        wrt: bool = True,
        method_cfg: Optional[Dict[str, Any]] = None,
        noise: Any = None,
        noise_scope: str = "all") -> Dict[str, Any]:
    """*run()* analyse every artifact of one (profile, scenario) pair dimensionally.

    Loads the profile from disk and delegates to `run_from_netcfg`. Optionally writes the JSON artifact to disk.

    Args:
        adp (Optional[str]): adaptation value; one of `baseline`, `s1`, `s2`, `aggregate`. Resolves to (profile, scenario) via `src.io.load_profile`.
        prf (Optional[str]): profile file stem (`dflt` or `opti`); overrides `adp`'s implied profile when paired with `scn`.
        scn (Optional[str]): explicit scenario name within the profile.
        wrt (bool): if True, write JSON artifact to `data/results/dimensional/<scenario>/<profile>.json`. Defaults to True.
        method_cfg (Optional[Dict[str, Any]]): inline override for the dimensional method parameters. When `None`, loads `data/config/method/dimensional.json`.
        noise (Any): optional seeded white-noise level on the independent variables (lambda, mu, epsilon); see `run_from_netcfg`. Defaults to None (clean run).
        noise_scope (str): `"all"` (default, perturbs the verdict too) or `"coefficients"` (clean verdict, noisy coefficients); see `run_from_netcfg`.

    Returns:
        Dict[str, Any]: result dict with the same shape as `run_from_netcfg`.
    """
    _cfg = load_profile(adaptation=adp,
                        profile=prf,
                        scenario=scn)
    return run_from_netcfg(_cfg, wrt=wrt, method_cfg=method_cfg,
                           noise=noise, noise_scope=noise_scope)


def _write_results(cfg: NetCfg,
                   method_cfg: Dict[str, Any],
                   artifacts: Dict[str, Dict[str, Any]],
                   nds: pd.DataFrame,
                   net: pd.DataFrame,
                   req: Dict[str, Any]) -> Dict[str, str]:
    """*_write_results()* serialise the dimensional-analysis outputs to disk in the standard result envelope.

    The envelope mirrors the analytic + stochastic methods (`network` + `nodes` + a separate `requirements.json`) so the cross-method comparison notebook can read all three from disk uniformly; the dimensional-method-specific payload lives under `artifacts` + `method_config`.

    Args:
        cfg (NetCfg): resolved network configuration.
        method_cfg (Dict[str, Any]): dimensional method params; copied verbatim into the result envelope so the run is self-describing on disk.
        artifacts (Dict[str, Dict[str, Any]]): per-artifact dimensional analysis blocks (pi_groups + coefficients + sensitivity).
        nds (pd.DataFrame): operational per-node frame (from the inherited Jackson solve).
        net (pd.DataFrame): operational network aggregate (one row; includes `W_e2e`, `eps_e2e`, `chi_out`).
        req (Dict[str, Any]): R1 / R2 verdict dict with per-node contributions.

    Returns:
        Dict[str, str]: on-disk paths of the written files, keyed by `profile` and `requirements`, relative to the repo root.
    """
    _out_dir = _RESULTS_DIR / cfg.scenario
    _out_dir.mkdir(parents=True, exist_ok=True)
    # routing + lambda_z embedded so the blob reconstructs cross-artifact without re-reading configs
    _doc = {
        "profile": cfg.profile,
        "scenario": cfg.scenario,
        "label": cfg.label,
        "method": "dimensional",
        "method_config": method_cfg,
        "network": net.iloc[0].to_dict(),
        "nodes": nds.to_dict(orient="records"),
        "artifacts": artifacts,
        "routing": cfg.routing.tolist(),
        "lambda_z": cfg.build_lam_z_vec().tolist(),
    }

    _profile_path = _out_dir / f"{cfg.profile}.json"
    with _profile_path.open("w", encoding="utf-8") as _fh:
        json.dump(_doc, _fh, indent=4, ensure_ascii=False)

    # write the R1 / R2 verdict (profile-agnostic, one per run; same convention as analytic + stochastic)
    _req_path = _out_dir / "requirements.json"
    with _req_path.open("w", encoding="utf-8") as _fh:
        json.dump(req, _fh, indent=4, ensure_ascii=False)

    return {
        "profile": str(_profile_path.relative_to(_ROOT)),
        "requirements": str(_req_path.relative_to(_ROOT)),
    }


def main() -> None:
    """*main()* CLI entry point.

    Parses command-line flags, calls `run()`, and prints a concise one-screen summary plus the path of any written file.
    """
    _parser = argparse.ArgumentParser(
        description="Dimensional-analysis solver for CS-01 TAS.",)
    _parser.add_argument(
        "--adaptation",
        choices=["baseline", "s1", "s2", "aggregate"],
        default=None,
        help=("adaptation state (resolves to profile + scenario); "
              "defaults to the profile's _setpoint"),)
    _parser.add_argument(
        "--profile",
        choices=["dflt", "opti"],
        default=None,
        help="explicit profile file stem (overrides adaptation's profile)",)
    _parser.add_argument(
        "--scenario",
        default=None,
        help="explicit scenario name within the profile",)
    _parser.add_argument(
        "--no-write",
        action="store_true",
        help="skip writing result files (useful for dry runs)",)
    _args = _parser.parse_args()
    _result = run(
        adp=_args.adaptation,
        prf=_args.profile,
        scn=_args.scenario,
        wrt=not _args.no_write,)
    _cfg = _result["config"]
    _arts = _result["artifacts"]
    _mc = _result["method_config"]
    print(f"profile={_cfg.profile}  scenario={_cfg.scenario}")
    print(f"label: {_cfg.label}")
    print(f"framework={_mc['fdus'][0]['_fwk']}  "
          f"fdus={len(_mc['fdus'])}  "
          f"coefficients={len(_mc['coefficients'])}  "
          f"sensitivity={_mc['sensitivity']['cat']}@{_mc['sensitivity']['val_type']}")
    print(f"artifacts ({len(_arts)}):")
    for _k, _a in _arts.items():
        _co = _a["coefficients"]
        _vals = "  ".join(
            f"{_sym.split('_')[0].lstrip(chr(92))}={_co[_sym]['setpoint']:.4g}"
            for _sym in _co.keys()
        )
        print(f"  {_k:<14} {_vals}")
    if _result["paths"]:
        for _k, _p in _result["paths"].items():
            print(f"  wrote {_k}: {_p}")


if __name__ == "__main__":
    main()
