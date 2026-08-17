# -*- coding: utf-8 -*-
"""
Module stochastic.py
====================

Stochastic (SimPy DES) method orchestrator for the CS-01 TAS case study. Mirrors the analytic method's contract call-for-call so the two can be compared directly downstream. Aggregation and threshold checks reuse `src.analytic.metrics` (`aggregate_net` / `check_reqs`) since the math is identical across the two methods.

The engine runs in SimPy SECONDS while the method config declares horizon and warmup in INVOCATIONS; the conversion happens in `src.stochastic.simulation.solve_net`.

Public API:
    - `run(adp, prf, scn, wrt)` loads a resolved `NetCfg` (same `profile/*.json` as analytic) plus the stochastic method config (`data/config/method/stochastic.json`), runs the DES engine (`src.stochastic.solve_net`), and returns per-node metrics + network aggregate + R1 / R2 verdict.

CLI::

    python -m src.methods.stochastic --adaptation baseline
    python -m src.methods.stochastic --adaptation s1 --profile opti
    python -m src.methods.stochastic  # uses _setpoint
"""
# native python modules
from __future__ import annotations

import argparse
import json
from pathlib import Path

# data types
from typing import Any, Dict, Optional

# scientific stack
import pandas as pd

# local modules
from src.analytic import aggregate_net, check_reqs
from src.io import NetCfg, load_method_cfg, load_profile
from src.stochastic import solve_net


_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "data" / "results" / "stochastic"


def run(adp: Optional[str] = None,
        prf: Optional[str] = None,
        scn: Optional[str] = None,
        wrt: bool = True,
        method_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """*run()* solve the stochastic Jackson network for one (profile, scenario) pair.

    Optionally writes the JSON artifacts to disk.

    Args:
        adp (Optional[str]): adaptation value; one of `baseline`, `s1`, `s2`, `aggregate`. Resolves to (profile, scenario) via `src.io.load_profile`.
        prf (Optional[str]): profile file stem (`dflt` or `opti`); overrides `adp`'s implied profile when paired with `scn`.
        scn (Optional[str]): explicit scenario name within the profile.
        wrt (bool): if True, write JSON artifacts under `data/results/stochastic/<scenario>/`. Defaults to True.
        method_cfg (Optional[Dict[str, Any]]): inline override for the stochastic method parameters (`seed`, `horizon_invocations`, `warmup_invocations`, `replications`, ...). When `None`, loads `data/config/method/stochastic.json`. Useful for tests that want a tiny horizon so the run finishes in seconds.

    Returns:
        Dict[str, Any]: result dict with keys:

            - `config` (NetCfg): resolved config.
            - `method_config` (Dict): stochastic method parameters.
            - `nodes` (pd.DataFrame): per-node frame (analytic schema plus `_std` columns).
            - `network` (pd.DataFrame): network aggregate (one row).
            - `requirements` (Dict): R1 / R2 verdict dict.
            - `paths` (Dict[str, str]): written file paths; empty when `wrt=False`.
    """
    _cfg = load_profile(adaptation=adp, profile=prf, scenario=scn)
    if method_cfg is not None:
        _method_cfg = method_cfg
    else:
        _method_cfg = load_method_cfg("stochastic")
    _nds = solve_net(_cfg, _method_cfg)
    _lam_z = float(_cfg.build_lam_z_vec()[0])
    # availability from the loss-network flow: the DES leaks give-ups to the FAIL sink (row-sum<1),
    # so eps_e2e = lambda_FAIL/lambda_z is read from the FAIL node's measured arrival rate
    _net = aggregate_net(_nds, lam_z=_lam_z, routing=_cfg.routing)
    _req = check_reqs(_nds, lam_z=_lam_z, routing=_cfg.routing)
    _paths: Dict[str, str] = {}
    if wrt:
        _paths = _write_results(_cfg, _method_cfg, _nds, _net, _req)

    return {
        "config": _cfg,
        "method_config": _method_cfg,
        "nodes": _nds,
        "network": _net,
        "requirements": _req,
        "paths": _paths,
    }


def _write_results(cfg: NetCfg,
                   method_cfg: Dict[str, Any],
                   nds: pd.DataFrame,
                   net: pd.DataFrame,
                   req: dict) -> Dict[str, str]:
    """*_write_results()* serialise the solver outputs to disk in the PyDASA-style envelope used across methods.

    Args:
        cfg (NetCfg): resolved network configuration.
        method_cfg (Dict[str, Any]): stochastic method params; copied verbatim into the result envelope so the run is fully self-describing on disk.
        nds (pd.DataFrame): per-node metrics frame.
        net (pd.DataFrame): network aggregate (one row).
        req (dict): R1 / R2 verdict dict.

    Returns:
        Dict[str, str]: on-disk paths of the two written files, keyed by `profile` and `requirements`, relative to the repo root.
    """
    _out_dir = _RESULTS_DIR / cfg.scenario
    _out_dir.mkdir(parents=True, exist_ok=True)
    # routing + lambda_z embedded so the blob reconstructs cross-artifact without re-reading configs
    _doc = {
        "profile": cfg.profile,
        "scenario": cfg.scenario,
        "label": cfg.label,
        "method": "stochastic",
        "method_config": method_cfg,
        "network": net.iloc[0].to_dict(),
        "nodes": nds.to_dict(orient="records"),
        "routing": cfg.routing.tolist(),
        "lambda_z": cfg.build_lam_z_vec().tolist(),
    }

    _profile_path = _out_dir / f"{cfg.profile}.json"
    with _profile_path.open("w", encoding="utf-8") as _fh:
        json.dump(_doc, _fh, indent=4, ensure_ascii=False)

    _req_path = _out_dir / "requirements.json"
    with _req_path.open("w", encoding="utf-8") as _fh:
        json.dump(req, _fh, indent=4, ensure_ascii=False)

    return {
        "profile": str(_profile_path.relative_to(_ROOT)),
        "requirements": str(_req_path.relative_to(_ROOT)),
    }


def main() -> None:
    """*main()* CLI entry point.

    Parses command-line flags, calls `run()`, and prints a concise one-screen summary plus the paths of any written files.
    """
    _parser = argparse.ArgumentParser(
        description="Stochastic SimPy DES solver for CS-01 TAS.",)
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
    _net = _result["network"].iloc[0]
    _req = _result["requirements"]
    _mc = _result["method_config"]

    print(f"profile={_cfg.profile}  scenario={_cfg.scenario}")
    print(f"label: {_cfg.label}")
    print(f"seed={_mc['seed']}  reps={_mc['replications']}  "
          f"horizon={_mc['horizon_invocations']} inv.  "
          f"warmup={_mc['warmup_invocations']} inv.")
    print(f"  nodes={int(_net['nodes'])}  "
          f"avg_rho={_net['avg_rho']:.4f}  "
          f"max_rho={_net['max_rho']:.4f}  "
          f"W_net={_net['W_net']*1000:.3f}ms (per-visit)  "
          f"W_e2e={_net['W_e2e']*1000:.3f}ms (end-to-end)  "
          f"eps_e2e={_net['eps_e2e']*100:.3f}%")

    print("requirements:")
    for _k, _v in _req.items():
        _status = "PASS" if _v["pass"] else "FAIL"
        _val = _v["value"]
        _thr = _v["threshold"]
        if _v["units"] == "seconds":
            _val_str = f"{_val*1000:.3f} ms"
            _thr_str = f"{_thr*1000:.1f} ms"
        elif _v["units"] in ("fraction", "probability"):
            _val_str = f"{_val*100:.3f}%"
            _thr_str = f"{_thr*100:.2f}%"
        elif isinstance(_val, (int, float)):
            _val_str = f"{_val:.6g}"
            _thr_str = f"{_thr:.6g}"
        else:
            _val_str = "n/a"
            _thr_str = f"{_thr}"
        print(f"  {_k}: {_status}  ({_v['metric']}={_val_str} vs threshold={_thr_str})")
        _contribs = _v.get("contributions", [])
        if _contribs:
            print("\ttop contributors:")
            for _c in _contribs[:3]:
                if _k == "R1":
                    print(f"      {_c['node']:>10s}  eps={_c['epsilon']:.2f}  V={_c['V']:.4f}  contrib={_c['contribution']*100:.3f}%")
                else:
                    print(f"      {_c['node']:>10s}  L={_c['L']:.4f}  W={_c['W']*1000:.3f}ms  share={_c['share']*100:.2f}%")
    if _result["paths"]:
        for _k, _p in _result["paths"].items():
            print(f"  wrote {_k}: {_p}")


if __name__ == "__main__":
    main()
