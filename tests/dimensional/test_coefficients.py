# -*- coding: utf-8 -*-
"""
Module test_coefficients.py
===========================

Coefficient derivation + numerical agreement with by-hand formulas.

Semantic pin: PyDASA's `Coefficient.calculate_setpoint()` evaluates at each variable's `_mean` (not `_setpoint`). For TAS_{1} most variables have `_mean == _setpoint`, but `c` has `_setpoint=1` and `_mean=2`; η therefore uses c=2 and by-hand checks must mirror that.

    - **TestCoefficientDerivation**: the four expected derived coefficients are created per artifact.
    - **TestCoefficientValues**: numerical setpoints match by-hand closed form to 6 decimals for TAS_{1}.
    - **TestExpressionGuardrails**: malformed expr_patterns raise clearly.
"""
# data types
from typing import Any, Dict, Tuple

# testing framework
import pytest

# pydasa library
from pydasa import AnalysisEngine

# module under test
from src.dimensional import build_engine, derive_coefs


def _std_mean(engine: AnalysisEngine, sym: str) -> float:
    """*_std_mean()* returns `std_mean` for a variable on the engine; this is the field PyDASA actually reads during `calculate_setpoint()`. Raises `ValueError` when the variable has no mean set (every variable in the test profiles does, so this is a defensive narrow for the type checker)."""
    _val = engine.variables[sym].std_mean
    if _val is None:
        _msg = f"variable {sym!r} has no std_mean set on the engine"
        raise ValueError(_msg)
    return float(_val)


class TestCoefficientDerivation:
    """**TestCoefficientDerivation** every artifact produces the four named derived coefficients (theta, sigma, eta, phi) after `run_analysis()`."""

    def test_four_coefs_for_tas1(self,
                                 engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_four_coefs_for_tas1()* `derive_coefs` on the post-analysis TAS_{1} engine returns 4 entries."""
        _, _derived = engine_ready
        assert len(_derived) == 4

    def test_derived_symbols(self,
                             engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_derived_symbols()* the 4 entries carry the artifact-subscripted `\\theta / \\sigma / \\eta / \\phi` keys."""
        _, _derived = engine_ready
        _expected = {
            "\\theta_{TAS_{1}}",
            "\\sigma_{TAS_{1}}",
            "\\eta_{TAS_{1}}",
            "\\phi_{TAS_{1}}",
        }
        assert set(_derived.keys()) == _expected

    def test_four_coefs_per_art(self,
                                schema: Any,
                                method_cfg: Dict[str, Any],
                                dflt_profile: Dict[str, Any]) -> None:
        """*test_four_coefs_per_art()* every service artifact in `dflt.json` yields exactly 4 derived coefficients (the synthetic `FAIL_{1}` sink, which has no output variable, is excluded)."""
        _arts = dflt_profile["artifacts"]
        for _a_key, _a in _arts.items():
            if _a_key == "FAIL_{1}":
                continue
            _engine = build_engine(_a_key, _a["vars"], schema)
            _engine.run_analysis()
            _derived = derive_coefs(_engine,
                                    method_cfg["coefficients"],
                                    artifact_key=_a_key)
            _msg = f"{_a_key}: expected 4 derived coefficients, got {len(_derived)}"
            assert len(_derived) == 4, _msg


class TestCoefficientValues:
    """**TestCoefficientValues** numerical setpoints match the by-hand formulas evaluated at PyDASA's `_std_mean` (which is what `calculate_setpoint()` actually reads).

    Values are read from the engine at test time so the tests track whatever is in `data/config/profile/dflt.json`; in particular the seeded values written by `src.utils.seed_dim_from_analytic` after the analytic solver is run.
    """

    def test_theta_equals_L_over_K(self,
                                   engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_theta_equals_L_over_K()* theta = L / K (queue fill ratio) at the seeded setpoint."""
        _eng, _derived = engine_ready
        _theta = _derived["\\theta_{TAS_{1}}"].setpoint
        _expected = _std_mean(_eng, "L_{TAS_{1}}") / _std_mean(_eng, "K_{TAS_{1}}")
        assert _theta == pytest.approx(_expected, rel=1e-6)

    def test_sigma_equals_lam_w_over_K(self,
                                       engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_sigma_equals_lam_w_over_K()* sigma = lambda * W / K (Stall, queueing share of capacity) at the seeded setpoint."""
        _eng, _derived = engine_ready
        _sigma = _derived["\\sigma_{TAS_{1}}"].setpoint
        _lam = _std_mean(_eng, "\\lambda_{TAS_{1}}")
        _w = _std_mean(_eng, "W_{TAS_{1}}")
        _K = _std_mean(_eng, "K_{TAS_{1}}")
        _expected = _lam * _w / _K
        assert _sigma == pytest.approx(_expected, rel=1e-6)

    def test_sigma_close_to_theta_under_little(self,
                                               engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_sigma_close_to_theta_under_little()* under Little's law (lambda * W = L), sigma = lambda*W/K reduces to L/K = theta. The seeded setpoints satisfy lambda*W ~= L to ~1e-4 precision, so the agreement holds at that tolerance (formulas are distinct; this is a sanity check, not an identity)."""
        _, _derived = engine_ready
        _sigma = _derived["\\sigma_{TAS_{1}}"].setpoint
        _theta = _derived["\\theta_{TAS_{1}}"].setpoint
        assert _sigma == pytest.approx(_theta, rel=1e-3)

    def test_eta_equals_chi_K_over_mu_c(self,
                                        engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_eta_equals_chi_K_over_mu_c()* eta = chi * K / (mu * c) (saturation coefficient) at the seeded setpoint."""
        _eng, _derived = engine_ready
        _eta = _derived["\\eta_{TAS_{1}}"].setpoint
        _chi = _std_mean(_eng, "\\chi_{TAS_{1}}")
        _K = _std_mean(_eng, "K_{TAS_{1}}")
        _mu = _std_mean(_eng, "\\mu_{TAS_{1}}")
        _c = _std_mean(_eng, "c_{TAS_{1}}")
        _expected = _chi * _K / (_mu * _c)
        assert _eta == pytest.approx(_expected, rel=1e-6)

    def test_phi_equals_M_act_over_M_buf(self,
                                         engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_phi_equals_M_act_over_M_buf()* phi = M_act / M_buf (memory occupancy) at the seeded setpoint."""
        _eng, _derived = engine_ready
        _phi = _derived["\\phi_{TAS_{1}}"].setpoint
        _m_act = _std_mean(_eng, "M_{act_{TAS_{1}}}")
        _m_buf = _std_mean(_eng, "M_{buf_{TAS_{1}}}")
        _expected = _m_act / _m_buf
        assert _phi == pytest.approx(_expected, rel=1e-6)

    def test_phi_collapses_to_L_over_K(self,
                                       engine_ready: Tuple[AnalysisEngine, Dict[str, Any]]) -> None:
        """*test_phi_collapses_to_L_over_K()* since `M_act = L * delta` and `M_buf = K * delta`, phi must equal theta once the common delta cancels."""
        _, _derived = engine_ready
        _phi = _derived["\\phi_{TAS_{1}}"].setpoint
        _theta = _derived["\\theta_{TAS_{1}}"].setpoint
        assert _phi == pytest.approx(_theta, rel=1e-9)


class TestExpressionGuardrails:
    """**TestExpressionGuardrails** malformed `expr_pattern` strings raise clear errors before touching PyDASA."""

    def test_out_of_range_pi_index_raises(self,
                                          schema: Any,
                                          tas1_vars: Dict[str, Any]) -> None:
        """*test_out_of_range_pi_index_raises()* an expr_pattern referencing `{pi[99]}` against a 7-Pi engine raises `IndexError` with the offending index."""
        _engine = build_engine("TAS_{1}", tas1_vars, schema)
        _engine.run_analysis()
        _bad_spec = [
            {
                "symbol": "bogus",
                "expr_pattern": "{pi[99]}",
                "name": "Bogus",
                "description": "should fail",
            }
        ]
        with pytest.raises(IndexError, match="pi\\[99\\]"):
            derive_coefs(_engine, _bad_spec, artifact_key="TAS_{1}")
