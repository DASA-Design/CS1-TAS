# -*- coding: utf-8 -*-
"""
Module results.py
=================

Disk loaders for the per-method result envelopes. The analytic, stochastic, and dimensional methods each persist a matched pair of files under `data/results/<method>/<adp>/`:

    - `<profile>.json` carries the network aggregate + per-node frame + routing + lambda_z (plus method-specific payloads under `artifacts` / `method_config` / `variables` depending on the method).
    - `requirements.json` carries the R1 / R2 verdict with per-node `contributions` lists (top-5 drivers per requirement).

This module reads those two files for any (method, adaptation) pair. The downstream consumer is the cross-method comparison notebook (`06-comparison.ipynb`), which reads the persisted JSONs rather than re-running the three methods.
"""
# native python modules
from __future__ import annotations

import json
from pathlib import Path

# data types
from typing import Any, Dict


_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "data" / "results"

# adaptation -> profile file stem (matches `_ADAPTATION_TO_SOURCE` in src/io/config.py)
_ADAPTATION_TO_PROFILE: Dict[str, str] = {
    "baseline": "dflt",
    "s1": "opti",
    "s2": "opti",
    "aggregate": "opti",
}

_METHODS = ("analytic", "stochastic", "dimensional")
_ADAPTATIONS = tuple(_ADAPTATION_TO_PROFILE.keys())


def _resolve_method(method: str) -> str:
    """*_resolve_method()* validate `method` against the persisted-method allowlist and return it unchanged.

    Raises:
        ValueError: if `method` is not in `{"analytic", "stochastic", "dimensional"}`.
    """
    if method not in _METHODS:
        _msg = f"unknown method {method!r}; allowed: {list(_METHODS)}"
        raise ValueError(_msg)
    return method


def _resolve_profile(adp: str) -> str:
    """*_resolve_profile()* map an adaptation key to its profile file stem.

    Raises:
        ValueError: if `adp` is not in `{"baseline", "s1", "s2", "aggregate"}`.
    """
    if adp not in _ADAPTATION_TO_PROFILE:
        _msg = f"unknown adaptation {adp!r}; allowed: {list(_ADAPTATIONS)}"
        raise ValueError(_msg)
    return _ADAPTATION_TO_PROFILE[adp]


def load_qn_result(method: str, adp: str) -> Dict[str, Any]:
    """*load_qn_result()* load the network + per-node envelope for one (method, adaptation) pair.

    Reads `data/results/<method>/<adp>/<profile>.json`. The profile (`dflt` or `opti`) is resolved from the adaptation per the project's adaptation-axis convention. Top-level keys vary slightly by method: all three carry `profile`, `scenario`, `label`, `method`, `network`, `nodes` (operational per-node records, the same flat-record shape across all three methods), `routing`, `lambda_z`; dimensional adds `artifacts` and `method_config`; stochastic adds `method_config`.

    Args:
        method (str): one of `"analytic"`, `"stochastic"`, `"dimensional"`.
        adp (str): one of `"baseline"`, `"s1"`, `"s2"`, `"aggregate"`.

    Returns:
        Dict[str, Any]: the parsed JSON envelope.

    Raises:
        ValueError: invalid `method` or `adp`.
        FileNotFoundError: the per-run JSON was never written (re-run the method first).
    """
    _method = _resolve_method(method)
    _profile = _resolve_profile(adp)
    _path = _RESULTS_DIR / _method / adp / f"{_profile}.json"
    if not _path.exists():
        _msg = (f"result envelope missing at {_path.relative_to(_ROOT)}; "
                f"run `python -m src.methods.{_method} --adaptation {adp}` first")
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)


def load_requirements(method: str, adp: str) -> Dict[str, Any]:
    """*load_requirements()* load the R1 / R2 verdict for one (method, adaptation) pair.

    Reads `data/results/<method>/<adp>/requirements.json`. Each verdict carries `metric`, `value`, `threshold`, `operator`, `units`, `pass`, `contributions`, `notes`. The `contributions` list holds the top-5 per-node drivers (R1: `(node, epsilon, V, contribution)`; R2: `(node, L, W, V, share)`).

    Args:
        method (str): one of `"analytic"`, `"stochastic"`, `"dimensional"`.
        adp (str): one of `"baseline"`, `"s1"`, `"s2"`, `"aggregate"`.

    Returns:
        Dict[str, Any]: requirement dict keyed by `R1` and `R2`.

    Raises:
        ValueError: invalid `method` or `adp`.
        FileNotFoundError: the per-run `requirements.json` was never written.
    """
    _method = _resolve_method(method)
    _resolve_profile(adp)
    _path = _RESULTS_DIR / _method / adp / "requirements.json"
    if not _path.exists():
        _msg = (f"requirements envelope missing at {_path.relative_to(_ROOT)}; "
                f"run `python -m src.methods.{_method} --adaptation {adp}` first")
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)
