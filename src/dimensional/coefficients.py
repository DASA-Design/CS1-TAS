# -*- coding: utf-8 -*-
"""
Module coefficients.py
======================

Config-driven derivation of operationally meaningful coefficients (theta, sigma, eta, phi) from a post-`run_analysis` engine. Specs come from `data/config/method/dimensional.json` under `coefficients`::

    {
        "symbol": "theta",
        "expr_pattern": "{pi[6]} * {pi[3]}**(-1)",
        "name": "Occupancy",
        "description": "theta = L/K - queue fill ratio"
    }

`{pi[i]}` placeholders resolve to the i-th Pi coefficient key as it sits in `engine.coefficients` after `run_analysis()`. Must be called AFTER analysis, else no Pi-groups exist yet.

Public API:
    - `derive_coefs(engine, specs, artifact_key)` apply every spec in order and return `{full_sym: Coefficient}` for the derived ones only.

Pi-index ordering is stable across adaptations for a given artifact but can shift if the variable set changes; re-verify with a spot test when the profile schema is edited.
"""
# native python modules
from __future__ import annotations

# text processing
import re

# data types
from typing import Any

# pydasa library
from pydasa import AnalysisEngine


# placeholder matcher for the `{pi[i]}` indices in expr_pattern strings
_PI_PAT = re.compile(r"\{pi\[(\d+)\]\}")


def _resolve_expr(expr_pattern: str, pi_keys: list[str]) -> str:
    """*_resolve_expr()* substitute `{pi[i]}` tokens with the i-th key in `pi_keys`.

    Args:
        expr_pattern (str): spec `expr_pattern` with `{pi[i]}` placeholders.
        pi_keys (list[str]): Pi coefficient keys in the order returned by `run_analysis()`.

    Raises:
        IndexError: when any `{pi[i]}` references an index outside `pi_keys`.

    Returns:
        str: expression ready to pass to `engine.derive_coefficient(expr=...)`.
    """
    def _sub(m: re.Match[str]) -> str:
        _i = int(m.group(1))
        if _i >= len(pi_keys):
            _msg = f"expr_pattern references pi[{_i}] but only {len(pi_keys)} "
            _msg += "Pi-groups were derived; check variable/FDU counts."
            raise IndexError(_msg)
        return pi_keys[_i]

    return _PI_PAT.sub(_sub, expr_pattern)


def derive_coefs(engine: AnalysisEngine,
                 specs: list[dict[str, Any]],
                 *,
                 artifact_key: str) -> dict[str, Any]:
    """*derive_coefs()* apply named coefficient specs to a post-analysis engine.

    Each spec's `symbol` (a bare name like `"theta"`, no leading backslash) is prepended with `\\\\` and subscripted with `artifact_key` so the final coefficient symbol becomes e.g. `\\theta_{TAS_{1}}`.

    Args:
        engine (AnalysisEngine): engine with Pi-groups already derived (`run_analysis()` must have been called).
        specs (list[dict[str, Any]]): coefficient specs from `dimensional.json` with keys `symbol` (bare LaTeX name, no leading backslash), `expr_pattern`, `name`, `description`.
        artifact_key (str): artifact identifier in LaTeX subscript form, e.g. `"TAS_{1}"`.

    Returns:
        dict[str, Any]: `{full_sym: Coefficient}` for the derived coefficients only; raw Pi-groups remain in `engine.coefficients`.
    """
    # collect Pi-group keys in order so `expr_pattern` indices line up with engine.coefficients
    _pi_keys = [_k for _k in engine.coefficients.keys() if _k.startswith("\\Pi_")]
    # apply specs in declaration order; iteration order matters for any pi[i] resolution
    _der: dict[str, Any] = {}
    for _sp in specs:
        _sym = _sp["symbol"]
        _full = f"\\{_sym}_{{{artifact_key}}}"
        _exp = _resolve_expr(_sp["expr_pattern"], _pi_keys)
        # idx=-1 appends to pydasa's coefficient list rather than overwriting a slot
        _coeff = engine.derive_coefficient(_exp,
                                           _full,
                                           f"{artifact_key} {_sp['name']} coefficient",
                                           _sp["description"],
                                           -1)
        _der[_full] = _coeff
    return _der
