# -*- coding: utf-8 -*-
"""
Module schema.py
================

Config-driven PyDASA `Schema` construction for the TAS case study. Kept deliberately thin: PyDASA validates the framework and builds the FDU matrix; this layer only shuttles the FDU list from `data/config/method/dimensional.json` into a `Schema(...)` call.

Public API:
    - `build_schema(fdus, fwk="CUSTOM")` returns a `Schema` with `_setup_fdus()` already applied, ready to attach to an `AnalysisEngine`.

Every FDU dict must carry `_fwk == fwk`; mismatches raise `ValueError` with an explicit list of offenders before PyDASA is touched.
"""
# native python modules
from __future__ import annotations

# data types
from typing import Any

# pydasa library
from pydasa.dimensional.vaschy import Schema


def build_schema(fdus: list[dict[str, Any]],
                 *,
                 fwk: str = "CUSTOM") -> Schema:
    """*build_schema()* construct a PyDASA `Schema` from a list of FDU dicts.

    Args:
        fdus (list[dict[str, Any]]): FDU dicts with keys `_idx`, `_sym`, `_fwk`, `_name`, `_unit`, `description`. Shape must match `Schema._fdu_lt`; typically loaded from `data/config/method/dimensional.json`.
        fwk (str): framework name; every FDU's `_fwk` field must match this. Defaults to `"CUSTOM"`.

    Raises:
        ValueError: when any FDU dict has `_fwk` different from `fwk`.

    Returns:
        Schema: initialised `Schema` with `_setup_fdus()` already called.
    """
    # FDU/framework mismatch must surface here, before PyDASA's silent _setup_fdus accept
    _bad = [_f for _f in fdus if _f.get("_fwk") != fwk]
    if _bad:
        _syms = ", ".join(_f.get("_sym", "?") for _f in _bad)
        _msg = f"FDUs with _fwk != {fwk!r}: {_syms}"
        raise ValueError(_msg)
    # PyDASA's Schema declares _fdu_lt: List[Dimension] but accepts dicts and materialises them internally; the ignore is scoped to this call only.
    _sch = Schema(_fwk=fwk,
                  _fdu_lt=list(fdus),  # type: ignore[arg-type]
                  _idx=0)
    return _sch
