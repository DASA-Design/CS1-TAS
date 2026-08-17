# -*- coding: utf-8 -*-
"""
Module test_sensitivity.py
==========================

Symbolic sensitivity shape + sign checks on TAS_{1} at variable means.

PyDASA keys sensitivity entries as `SEN_{<coeff_symbol>}`; the short aliases
below hide that prefix so the sign tests stay readable.

    - **TestSensitivityShape**: the wrapper returns a nested `{SEN_{coeff}: {var: float}}` dict with only numeric leaves and contains entries for every derived coefficient.
    - **TestSensitivitySigns**: partial derivatives have the expected sign for the four derived coefficients (θ, σ, η, φ). Signs, not magnitudes; magnitudes depend on evaluation point and drift with config tweaks.
"""
# native python modules
from numbers import Real

# data types
from typing import Dict

_THETA = "SEN_{\\theta_{TAS_{1}}}"
_SIGMA = "SEN_{\\sigma_{TAS_{1}}}"
_ETA = "SEN_{\\eta_{TAS_{1}}}"
_PHI = "SEN_{\\phi_{TAS_{1}}}"


class TestSensitivityShape:
    """**TestSensitivityShape** the wrapper returns a nested `{SEN_{coeff}: {var: float}}` dict with only numeric leaves, containing entries for every derived coefficient."""

    def test_returns_dict(self,
                          sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_returns_dict()* the top-level return is a plain dict (not a pandas DataFrame or sympy Matrix)."""
        assert isinstance(sensitivity_results, dict)

    def test_contains_derived_coefficients(self,
                                           sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_contains_derived_coefficients()* the four derived coefficients (theta, sigma, eta, phi) each have a `SEN_{...}` entry."""
        _keys = set(sensitivity_results.keys())
        for _derived in (_THETA, _SIGMA, _ETA, _PHI):
            assert _derived in _keys, f"missing sensitivity for {_derived}"

    def test_all_leaves_are_numeric(self,
                                    sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_all_leaves_are_numeric()* every inner-dict value is a real number; the wrapper filters out sympy residues that would otherwise sneak through."""
        for _coeff_sym, _var_map in sensitivity_results.items():
            assert isinstance(_var_map, dict)
            for _var_sym, _val in _var_map.items():
                assert isinstance(_val, Real), (
                    f"{_coeff_sym} / {_var_sym} is {type(_val).__name__}, expected Real"
                )


class TestSensitivitySigns:
    """**TestSensitivitySigns** partial derivatives carry the expected sign for the four derived coefficients at the mean operating point. Signs, not magnitudes; magnitudes shift with evaluation point and config tweaks."""

    def test_theta_partial_L_positive(self,
                                      sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_theta_partial_L_positive()* theta = L/K -> d_theta/d_L = 1/K > 0."""
        assert sensitivity_results[_THETA]["L_{TAS_{1}}"] > 0

    def test_theta_partial_K_negative(self,
                                      sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_theta_partial_K_negative()* theta = L/K -> d_theta/d_K = -L/K^2 < 0."""
        assert sensitivity_results[_THETA]["K_{TAS_{1}}"] < 0

    def test_eta_partial_mu_negative(self,
                                     sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_eta_partial_mu_negative()* eta = chi*K/(mu*c) -> d_eta/d_mu = -chi*K/(mu^2*c) < 0."""
        assert sensitivity_results[_ETA]["\\mu_{TAS_{1}}"] < 0

    def test_eta_partial_K_positive(self,
                                    sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_eta_partial_K_positive()* eta = chi*K/(mu*c) -> d_eta/d_K = chi/(mu*c) > 0."""
        assert sensitivity_results[_ETA]["K_{TAS_{1}}"] > 0

    def test_sigma_partial_K_negative(self,
                                      sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_sigma_partial_K_negative()* sigma = lambda*W/K -> d_sigma/d_K = -lambda*W/K^2 < 0."""
        assert sensitivity_results[_SIGMA]["K_{TAS_{1}}"] < 0

    def test_sigma_partial_W_positive(self,
                                      sensitivity_results: Dict[str, Dict[str, float]]) -> None:
        """*test_sigma_partial_W_positive()* sigma = lambda*W/K -> d_sigma/d_W = lambda/K > 0."""
        assert sensitivity_results[_SIGMA]["W_{TAS_{1}}"] > 0
