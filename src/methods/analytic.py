# -*- coding: utf-8 -*-
"""
Module analytic.py
==================

Analytic method orchestrator for the CS-01 TAS case study.

Loads a resolved `NetCfg`, solves the Jackson network in closed form via M/M/c/K at each node, emits node + network metrics as a single PyDASA-style JSON, and writes an R1 / R2 verdict alongside.

Public API:
    - `run(adp, prf, scn, wrt)` standard orchestrator contract.
    - `main()` CLI entry point.

The written result blob carries the full `routing` matrix and `lambda_z` vector so downstream consumers can reconstruct node paths without re-opening the config files.

CLI::

    python -m src.methods.analytic --adaptation baseline
    python -m src.methods.analytic --adaptation s1 --profile opti
    python -m src.methods.analytic  # uses _setpoint
"""
# native python modules
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

# scientific stack
# import numpy as np
import pandas as pd

# local modules
from src.analytic import aggregate_net, check_reqs, solve_network
from src.io import NetCfg, load_profile


_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "data" / "results" / "analytic"


def run(adp: Optional[str] = None,
        prf: Optional[str] = None,
        scn: Optional[str] = None,
        wrt: bool = True) -> Dict[str, Any]:
    """*run()* solve the analytic Jackson network for one (profile, scenario) pair.

    Optionally writes the JSON artifacts to disk.

    Args:
        adp (Optional[str]): adaptation value; one of `baseline`, `s1`, `s2`, `aggregate`. Resolves to (profile, scenario) via `src.io.load_profile`.
        prf (Optional[str]): profile file stem (`dflt` or `opti`); overrides `adp`'s implied profile when paired with `scn`.
        scn (Optional[str]): explicit scenario name within the profile.
        wrt (bool): if True, write JSON artifacts to `data/results/analytic/<scenario>/`. Defaults to True.

    Returns:
        Dict[str, Any]: result dict with keys:

            - `config` (NetCfg): resolved config (for display).
            - `nodes` (pd.DataFrame): per-node DataFrame.
            - `network` (pd.DataFrame): network aggregate (one row).
            - `requirements` (Dict): R1 / R2 verdict dict.
            - `paths` (Dict[str, str]): written file paths; empty when `wrt=False`.
    """
    # resolve the config then solve the network end-to-end
    _cfg = load_profile(adaptation=adp, profile=prf, scenario=scn)
    _nds = solve_network(_cfg)
    _lam_z = float(_cfg.build_lam_z_vec()[0])
    # availability from the loss-network flow: routing carries the give-up leaks to the FAIL sink
    _net = aggregate_net(_nds, lam_z=_lam_z, routing=_cfg.routing)
    _req = check_reqs(_nds, lam_z=_lam_z, routing=_cfg.routing)
    # write only when the caller asks; keeps `run(wrt=False)` side-effect-free for tests + sweeps
    _paths: Dict[str, str] = {}
    if wrt:
        _paths = _write_results(_cfg, _nds, _net, _req)

    return {
        "config": _cfg,
        "nodes": _nds,
        "network": _net,
        "requirements": _req,
        "paths": _paths,
    }


def _write_results(cfg: NetCfg,
                   nds: pd.DataFrame,
                   net: pd.DataFrame,
                   req: dict) -> Dict[str, str]:
    """*_write_results()* serialises the solver outputs to disk in the standard result envelope.

    Args:
        cfg (NetCfg): resolved network configuration.
        nds (pd.DataFrame): per-node metrics frame.
        net (pd.DataFrame): network aggregate frame (one row).
        req (dict): R1 / R2 verdict dict.

    Returns:
        Dict[str, str]: on-disk paths of the two written files, keyed by `profile` and `requirements`, relative to the repo root.
    """
    # prepare the scenario-scoped output directory
    _out_dir = _RESULTS_DIR / cfg.scenario
    _out_dir.mkdir(parents=True, exist_ok=True)

    # # assemble the result envelope. topology (routing + lambda_z)
    _doc = {
        "profile": cfg.profile,
        "scenario": cfg.scenario,
        "label": cfg.label,
        "method": "analytic",
        "network": net.iloc[0].to_dict(),
        "nodes": nds.to_dict(orient="records"),
        "routing": cfg.routing.tolist(),
        "lambda_z": cfg.build_lam_z_vec().tolist(),
    }

    # write the per-profile result blob
    _profile_path = _out_dir / f"{cfg.profile}.json"
    with _profile_path.open("w", encoding="utf-8") as _fh:
        json.dump(_doc, _fh, indent=4, ensure_ascii=False)

    # write the R1 / R2 verdict (profile-agnostic, one per run)
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
    # build the argument parser with the four CLI flags
    _parser = argparse.ArgumentParser(
        description="Analytic Jackson-network solver for CS-01 TAS.",)

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
    # run the solver end-to-end with the parsed flags
    _result = run(
        adp=_args.adaptation,
        prf=_args.profile,
        scn=_args.scenario,
        wrt=not _args.no_write,)

    # unpack the result blob for the summary print
    _cfg = _result["config"]
    _net = _result["network"].iloc[0]
    _req = _result["requirements"]
    # header: which (profile, scenario) was solved
    print(f"profile={_cfg.profile}  scenario={_cfg.scenario}")
    print(f"label: {_cfg.label}")
    # network-wide summary line: W_net (per-visit), W_e2e (end-to-end, R2-comparable), eps_e2e (R1-comparable)
    print(f"\tnodes={int(_net['nodes'])}  "
          f"avg_rho={_net['avg_rho']:.4f}  "
          f"max_rho={_net['max_rho']:.4f}  "
          f"W_net={_net['W_net']*1000:.3f}ms (per-visit)  "
          f"W_e2e={_net['W_e2e']*1000:.3f}ms (end-to-end)  "
          f"eps_e2e={_net['eps_e2e']*100:.3f}%")

    # per-requirement PASS / FAIL with the numeric value, threshold, and top contributors (evidence)
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
        print(f"\t{_k}: {_status}  ({_v['metric']}={_val_str} vs threshold={_thr_str})")
        # top per-node contributors as evidence
        _contribs = _v.get("contributions", [])
        if _contribs:
            print("\ttop contributors:")
            for _c in _contribs[:3]:
                if _k == "R1":
                    print(f"\t{_c['node']:>10s}  eps={_c['epsilon']:.2f}  V={_c['V']:.4f}  contrib={_c['contribution']*100:.3f}%")
                else:
                    print(f"\t{_c['node']:>10s}  L={_c['L']:.4f}  W={_c['W']*1000:.3f}ms  share={_c['share']*100:.2f}%")

    # written-file paths (only when wrt=True)
    if _result["paths"]:
        for _k, _p in _result["paths"].items():
            print(f"\twrote {_k}: {_p}")


if __name__ == "__main__":
    main()
