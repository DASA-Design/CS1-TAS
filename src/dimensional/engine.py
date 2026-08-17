# -*- coding: utf-8 -*-
"""
Module engine.py
================

Per-artifact PyDASA `AnalysisEngine` construction for the TAS case study. One engine per artifact (`TAS_{1..6}`, `MAS_{1..4}`, `AS_{1..4}`, `DS_{1,3}`); `engine.run_analysis()` then derives Pi-groups from the attached variables.

Public API:
    - `build_engine(artifact_key, artifact_vars, schema)` instantiate an engine for the named artifact and attach its `Variable` dict.

PyDASA takes ownership of the Variables on assignment; later mutation of the source dict does not propagate back to the engine.
"""
# native python modules
from __future__ import annotations

# data types
from typing import Any

# pydasa library
from pydasa.dimensional.vaschy import Schema
from pydasa.elements.parameter import Variable
from pydasa.workflows.phenomena import AnalysisEngine


def build_engine(artifact_key: str,
                 artifact_vars: dict[str, dict[str, Any]],
                 schema: Schema,
                 *,
                 fwk: str = "CUSTOM",
                 idx: int = 0) -> AnalysisEngine:
    """*build_engine()* assemble a configured `AnalysisEngine` for one TAS artifact.

    Args:
        artifact_key (str): artifact identifier in LaTeX subscript form, e.g. `"TAS_{1}"`. Used in the engine name and description.
        artifact_vars (dict[str, dict[str, Any]]): per-variable param dicts from the PACS envelope (`config["artifacts"][artifact_key]["vars"]`). Each value must match the `Variable(**params)` signature.
        schema (Schema): framework schema built by `build_schema()`.
        fwk (str): framework name; must match the schema. Defaults to `"CUSTOM"`.
        idx (int): engine index (informational).

    Returns:
        AnalysisEngine: engine with `engine.variables` already populated. Call `engine.run_analysis()` next to derive Pi-groups.
    """
    _vars = {_s: Variable(**_p) for _s, _p in artifact_vars.items()}
    # pass-by-keyword: pydasa's field order across SymBasis/IdxBasis/Foundation/WorkflowBase makes positional args fragile
    _eng = AnalysisEngine(_idx=idx,
                          _fwk=fwk,
                          _schema=schema,
                          _name=f"TAS {artifact_key} dimensional analysis",
                          description=f"Dimensional analysis for artifact {artifact_key} (M/M/c/K queueing node).")
    # pydasa takes ownership; later mutation of the source dict does not propagate
    _eng.variables = _vars
    return _eng
