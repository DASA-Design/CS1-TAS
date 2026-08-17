# -*- coding: utf-8 -*-
"""
Module test_schema.py
=====================

Config-driven `build_schema()` contract:

    - **TestSchemaConstruction**: FDUs from `data/config/method/dimensional.json` land on the Schema with `_setup_fdus()` already applied.
    - **TestSchemaGuardrails**: mismatched `_fwk` raises a clear error.
"""
# testing framework
import pytest

# pydasa library
from pydasa.dimensional.vaschy import Schema

# module under test
from src.dimensional import build_schema


class TestSchemaConstruction:
    """**TestSchemaConstruction** the Schema carries the TAS T/S/D framework after construction, with `_setup_fdus()` already applied."""

    def test_is_schema_instance(self, schema: Schema) -> None:
        """*test_is_schema_instance()* `isinstance(build_schema(...), Schema)` holds."""
        assert isinstance(schema, Schema)

    def test_three_fdus_registered(self, schema: Schema) -> None:
        """*test_three_fdus_registered()* `len(schema._fdu_lt) == 3` (TAS uses T, S, D)."""
        assert len(schema._fdu_lt) == 3

    def test_fdu_symbols_are_T_S_D(self, schema: Schema) -> None:
        """*test_fdu_symbols_are_T_S_D()* `{fdu._sym for fdu in schema._fdu_lt} == {"T", "S", "D"}` (canonical TAS set)."""
        _syms = {_fdu._sym for _fdu in schema._fdu_lt}
        assert _syms == {"T", "S", "D"}

    def test_fdu_units_are_s_req_kB(self, schema: Schema) -> None:
        """*test_fdu_units_are_s_req_kB()* `{T: "s", S: "req", D: "kB"}` propagated from the config into `_unit`."""
        _units = {_fdu._sym: _fdu._unit for _fdu in schema._fdu_lt}
        assert _units == {"T": "s", "S": "req", "D": "kB"}

    def test_fwk_is_custom(self, schema: Schema) -> None:
        """*test_fwk_is_custom()* `schema.fwk == "CUSTOM"` and every `fdu._fwk == "CUSTOM"` end-to-end."""
        assert schema.fwk == "CUSTOM"
        for _fdu in schema._fdu_lt:
            assert _fdu._fwk == "CUSTOM"


class TestSchemaGuardrails:
    """**TestSchemaGuardrails** misconfigured inputs raise before PyDASA is touched."""

    def test_rejects_mismatched_fwk(self) -> None:
        """*test_rejects_mismatched_fwk()* an FDU with `_fwk != fwk` raises `ValueError` matching `"_fwk"`."""
        _bad_fdus = [
            {"_idx": 0, "_sym": "T", "_fwk": "OTHER", "_name": "Time",
             "_unit": "s", "description": "time"},
        ]
        with pytest.raises(ValueError, match="_fwk"):
            build_schema(_bad_fdus)

    def test_empty_fdu_list_raises(self) -> None:
        """*test_empty_fdu_list_raises()* `build_schema([])` raises `ValueError` matching `"FDUs"` (PyDASA rejects empty CUSTOM frameworks; this wrapper propagates the error)."""
        with pytest.raises(ValueError, match="FDUs"):
            build_schema([])
