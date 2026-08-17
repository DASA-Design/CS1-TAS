# -*- coding: utf-8 -*-
"""
Module conftest.py
==================

Shared pytest fixtures for the whole test tree. Fixtures are lazy and scoped so files that do not depend on them pay no runtime cost.

Dimensional / PyDASA fixtures (used by `tests/dimensional/`):

    - `method_cfg` / `dflt_profile` / `opti_profile`: session-scoped JSON loads.
    - `schema` / `tas1_vars` / `engine_bare` / `engine_ready`: module-scoped PyDASA build steps for TAS_{1}.
    - `sensitivity_results`: module-scoped, derived from `engine_ready`.
"""
# native python modules
from __future__ import annotations

import json
from pathlib import Path

# data types
from typing import Any, Dict, Tuple

# testing framework
import pytest

# pydasa library
from pydasa import AnalysisEngine
from pydasa.dimensional.vaschy import Schema

# local modules
from src.dimensional import (analyse_symbolic,
                             build_engine,
                             build_schema,
                             derive_coefs)


_ROOT = Path(__file__).resolve().parents[1]
_METHOD_DIM = _ROOT / "data" / "config" / "method" / "dimensional.json"
_DFLT_PATH = _ROOT / "data" / "config" / "profile" / "dflt.json"
_OPTI_PATH = _ROOT / "data" / "config" / "profile" / "opti.json"


# ---- Dimensional / PyDASA fixtures ----

@pytest.fixture(scope="session")
def method_cfg() -> Dict[str, Any]:
    """*method_cfg()* `data/config/method/dimensional.json` parsed once per session."""
    return json.loads(_METHOD_DIM.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def dflt_profile() -> Dict[str, Any]:
    """*dflt_profile()* `data/config/profile/dflt.json` parsed once per session."""
    return json.loads(_DFLT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def opti_profile() -> Dict[str, Any]:
    """*opti_profile()* `data/config/profile/opti.json` parsed once per session."""
    return json.loads(_OPTI_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema(method_cfg: Dict[str, Any]) -> Schema:
    """*schema()* PyDASA `Schema` built from `method_cfg["fdus"]`, module-cached."""
    return build_schema(method_cfg["fdus"])


@pytest.fixture(scope="module")
def tas1_vars(dflt_profile: Dict[str, Any]) -> Dict[str, Any]:
    """*tas1_vars()* `dflt_profile["artifacts"]["TAS_{1}"]["vars"]`, module-cached."""
    return dflt_profile["artifacts"]["TAS_{1}"]["vars"]


@pytest.fixture(scope="module")
def engine_bare(schema: Schema,
                tas1_vars: Dict[str, Any]) -> AnalysisEngine:
    """*engine_bare()* TAS_{1} engine with variables attached but BEFORE `run_analysis()`."""
    return build_engine("TAS_{1}", tas1_vars, schema)


@pytest.fixture(scope="module")
def engine_ready(schema: Schema,
                 tas1_vars: Dict[str, Any],
                 method_cfg: Dict[str, Any]) -> Tuple[AnalysisEngine, Dict[str, Any]]:
    """*engine_ready()* `(engine, derived)` after `run_analysis()` + `derive_coefs()` with every setpoint evaluated; `derived` is the 4-entry dict returned by `derive_coefs`."""
    _eng = build_engine("TAS_{1}", tas1_vars, schema)
    _eng.run_analysis()
    # raw Pi-groups need an explicit setpoint pass; PyDASA leaves them lazy after run_analysis
    for _c in _eng.coefficients.values():
        _c.calculate_setpoint()
    _der = derive_coefs(_eng, method_cfg["coefficients"],
                        artifact_key="TAS_{1}")
    for _c in _der.values():
        _c.calculate_setpoint()
    return _eng, _der


@pytest.fixture(scope="module")
def sensitivity_results(engine_ready: Tuple[AnalysisEngine, Dict[str, Any]],
                        schema: Schema,
                        method_cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """*sensitivity_results()* `analyse_symbolic(engine, schema, val_type, cat)` over `engine_ready[0]`, module-cached."""
    _eng, _ = engine_ready
    _sc = method_cfg["sensitivity"]
    return analyse_symbolic(_eng, schema,
                            val_type=_sc["val_type"],
                            cat=_sc["cat"])
