# -*- coding: utf-8 -*-
"""
Module sensitivity.py
=====================

Symbolic sensitivity analysis wrapper around PyDASA's `SensitivityAnalysis`. Runs one symbolic-differentiation sweep at a single evaluation point (mean, setpoint, min, or max) and reshapes the raw output into a numeric-only nested dict.

PyDASA keys sensitivity entries as `SEN_{<coeff_symbol>}`, so raw Pi entries become `SEN_{\\Pi_{0}}` and derived ones e.g. `SEN_{\\theta_{TAS_{1}}}`; callers that want the raw coefficient symbol back need to strip the prefix.

Public API:
    - `analyse_symbolic(engine, schema, val_type="mean")` returns `{SEN_{coeff}: {var: float}}`.
"""
# native python modules
from __future__ import annotations

# data types
from typing import Dict

# pydasa library
from pydasa import AnalysisEngine, Schema, SensitivityAnalysis


def analyse_symbolic(engine: AnalysisEngine,
                     schema: Schema,
                     *,
                     val_type: str = "mean",
                     cat: str = "SYM",
                     fwk: str = "CUSTOM",
                     idx: int = 0,
                     name: str = "sensitivity") -> Dict[str, Dict[str, float]]:
    """*analyse_symbolic()* run symbolic sensitivity at a single value type.

    Args:
        engine (AnalysisEngine): engine with variables and coefficients already populated.
        schema (Schema): framework schema used to build `engine`.
        val_type (str): evaluation point for the partial derivatives; one of `"mean"`, `"setpoint"`, `"min"`, `"max"`. Defaults to `"mean"`.
        cat (str): sensitivity category; `"SYM"` for symbolic differentiation. Defaults to `"SYM"`.
        fwk (str): framework name. Defaults to `"CUSTOM"`.
        idx (int): workflow index (informational).
        name (str): human-readable workflow name.

    Returns:
        Dict[str, Dict[str, float]]: nested `{SEN_{coeff}: {var: sensitivity_value}}` over every coefficient in `engine.coefficients` (raw Pi and derived). Non-numeric entries are filtered out.

    Raises:
        ValueError: when `val_type` is not one of `{"mean", "setpoint", "min", "max"}` (propagated from `pydasa.SensitivityAnalysis.analyze_symbolic`).
    """
    # pass by keyword: SensitivityAnalysis dataclass field order differs from this function's arg order
    _sen = SensitivityAnalysis(_idx=idx,
                               _fwk=fwk,
                               _schema=schema,
                               _name=name,
                               _cat=cat)
    _sen.variables = engine.variables
    _sen.coefficients = engine.coefficients
    _raw = _sen.analyze_symbolic(val_type=val_type)

    # drop sympy residues; keep only numeric leaves so callers can index without isinstance checks
    _out: Dict[str, Dict[str, float]] = {}
    for _coef, _vmap in _raw.items():
        _numeric: Dict[str, float] = {}
        for _var_sym, _val in _vmap.items():
            if isinstance(_val, (int, float)):
                _numeric[_var_sym] = float(_val)
        _out[_coef] = _numeric
    return _out
