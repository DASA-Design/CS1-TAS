# -*- coding: utf-8 -*-
"""
Module test_engine.py
=====================

Per-artifact `build_engine()` + `engine.run_analysis()` contract.

    - **TestEngineConstruction** variables flow through to the engine and the IN/OUT/CTRL counts match the TAS_{1} schema.
    - **TestPiGroupDerivation** Buckingham's theorem holds (`n_relevant - n_fdus` Pi-groups), and Pi symbolic expressions are invariant across TAS adaptation scenarios (same variable set, different setpoints).
"""
# data types
from typing import Any, Dict, List

# module under test
from src.dimensional import build_engine


class TestEngineConstruction:
    """**TestEngineConstruction** `build_engine()` attaches all 13 variables to the engine with the expected IN/OUT/CTRL categorisation and framework."""

    def test_thirteen_vars(self, engine_bare: Any) -> None:
        """*test_thirteen_vars()* `len(engine.variables) == 13`."""
        assert len(engine_bare.variables) == 13

    def test_ten_relevant_vars(self, engine_bare: Any) -> None:
        """*test_ten_relevant_vars()* 10 of 13 variables carry `relevant=True`."""
        _relevant = sum(1 for _v in engine_bare.variables.values() if _v.relevant)
        assert _relevant == 10

    def test_three_in_vars(self, engine_bare: Any) -> None:
        """*test_three_in_vars()* 3 variables have `cat == "IN"` (lambda, c, delta)."""
        _ins = [_v for _v in engine_bare.variables.values() if _v.cat == "IN"]
        assert len(_ins) == 3

    def test_one_out_var(self, engine_bare: Any) -> None:
        """*test_one_out_var()* 1 variable has `cat == "OUT"` (W)."""
        _outs = [_v for _v in engine_bare.variables.values() if _v.cat == "OUT"]
        assert len(_outs) == 1

    def test_nine_ctrl_vars(self, engine_bare: Any) -> None:
        """*test_nine_ctrl_vars()* 9 variables have `cat == "CTRL"` (rounds out the 10-relevant / 3-FDU Buckingham budget)."""
        _ctrls = [_v for _v in engine_bare.variables.values() if _v.cat == "CTRL"]
        assert len(_ctrls) == 9

    def test_fwk_matches_schema(self, engine_bare: Any, schema: Any) -> None:
        """*test_fwk_matches_schema()* `engine.fwk == schema.fwk == "CUSTOM"`."""
        assert engine_bare.fwk == schema.fwk == "CUSTOM"


class TestPiGroupDerivation:
    """**TestPiGroupDerivation** Buckingham's theorem (`n_relevant - n_fdus` Pi-groups) holds, and Pi symbolic expressions are invariant to setpoint-only changes across adaptations."""

    def test_seven_pi_groups(self, engine_ready: Any) -> None:
        """*test_seven_pi_groups()* `len(pi_keys) == 7` (10 relevant variables minus 3 FDUs per Buckingham)."""
        _engine, _ = engine_ready
        _pi_keys = [_k for _k in _engine.coefficients if _k.startswith("\\Pi_")]
        assert len(_pi_keys) == 7

    def test_pi_exprs_stable_across_adps(self,
                                         schema: Any,
                                         dflt_profile: Dict[str, Any],
                                         opti_profile: Dict[str, Any]) -> None:
        """*test_pi_exprs_stable_across_adps()* `[str(c.pi_expr) for c in dflt_pis] == [str(c.pi_expr) for c in opti_pis]` (setpoints change, Pi structure does not)."""
        _exp_per_profile: Dict[str, List[str]] = {}
        for _label, _profile in [("dflt", dflt_profile), ("opti", opti_profile)]:
            _vars = _profile["artifacts"]["TAS_{1}"]["vars"]
            _engine = build_engine("TAS_{1}", _vars, schema)
            _engine.run_analysis()
            _pi_keys = [_k for _k in _engine.coefficients if _k.startswith("\\Pi_")]
            _exp_per_profile[_label] = [str(_engine.coefficients[_k].pi_expr)
                                        for _k in _pi_keys]
        _msg = "Pi-group symbolic expressions should be invariant to setpoint-only changes"
        assert _exp_per_profile["dflt"] == _exp_per_profile["opti"], _msg

    def test_pi_zero_is_lam_W_over_c(self, engine_ready: Any) -> None:
        """*test_pi_zero_is_lam_W_over_c()* `Pi_0.var_dims == {lambda: 1, W: 1, c: -1}` (Pi_0 = lambda * W / c, per observed TAS ordering)."""
        _engine, _ = engine_ready
        _pi0 = _engine.coefficients["\\Pi_{0}"]
        _dims = _pi0.var_dims
        assert _dims.get("\\lambda_{TAS_{1}}") == 1
        assert _dims.get("W_{TAS_{1}}") == 1
        assert _dims.get("c_{TAS_{1}}") == -1
