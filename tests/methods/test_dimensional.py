# -*- coding: utf-8 -*-
"""
Module test_dimensional.py
==========================

End-to-end sanity checks for the dimensional-method orchestrator in `src.methods.dimensional`.

Each adaptation is solved ONCE in a module-scope fixture and the cached dict is reused across every assertion. One full 13-artifact solve runs in ~1.5 s, so the file finishes in ~5 s.

    - **TestDimensionalEndToEnd**: each adaptation solves end-to-end via PyDASA and produces 13 (or 16) artifact blocks with the expected pi-group / coefficient / sensitivity shape.
    - **TestResultEnvelope**: the JSON envelope written to `data/results/dimensional/<scenario>/<profile>.json` is well-formed and round-trips on disk.
    - **TestMethodCfgOverride**: the `method_cfg=` kwarg lets callers inject a trimmed spec so tests do not depend on disk state.
    - **TestRunFromNetCfg**: `run_from_netcfg` on an in-memory `NetCfg` returns identical `nodes` / `network` / `requirements` / per-artifact coefficient setpoints as `run` on the same adaptation.
"""
# native python modules
import json
from typing import Any, Dict

# testing framework
import pytest

# modules under test
from src.io import load_profile
from src.methods.dimensional import run as run_dimensional
from src.methods.dimensional import run_from_netcfg


@pytest.fixture(scope="module")
def _result_baseline() -> Dict[str, Any]:
    """*_result_baseline()* `run_dimensional(adp="baseline", wrt=False)` once per module."""
    return run_dimensional(adp="baseline", wrt=False)


@pytest.fixture(scope="module")
def _result_s1() -> Dict[str, Any]:
    """*_result_s1()* `run_dimensional(adp="s1", wrt=False)` once per module."""
    return run_dimensional(adp="s1", wrt=False)


@pytest.fixture(scope="module")
def _result_aggregate() -> Dict[str, Any]:
    """*_result_aggregate()* `run_dimensional(adp="aggregate", wrt=False)`; exercises the 16-artifact opti profile."""
    return run_dimensional(adp="aggregate", wrt=False)


class TestDimensionalEndToEnd:
    """**TestDimensionalEndToEnd** the PyDASA pipeline runs end-to-end across `baseline` / `s1` / `aggregate`, producing one artifact block per queue node with the four derived coefficients plus sensitivity. Covers both profiles (`dflt` / `opti`) and both artifact counts (13 / 16)."""

    @pytest.fixture(params=["baseline", "s1", "aggregate"])
    def _result(self,
                request: pytest.FixtureRequest,
                _result_baseline: Dict[str, Any],
                _result_s1: Dict[str, Any],
                _result_aggregate: Dict[str, Any]) -> Dict[str, Any]:
        """*_result()* dispatch the right per-adaptation result by `request.param`."""
        _map = {
            "baseline": _result_baseline,
            "s1": _result_s1,
            "aggregate": _result_aggregate,
        }
        return _map[request.param]

    def test_art_count_matches_cfg(self, _result: Dict[str, Any]) -> None:
        """*test_art_count_matches_cfg()* the dimensional analysis covers every config artifact except the synthetic `FAIL_{1}` sink (which has no service variables)."""
        _arts = _result["artifacts"]
        _cfg = _result["config"]
        _real = [_a for _a in _cfg.artifacts if _a.key != "FAIL_{1}"]
        assert len(_arts) == len(_real)

    def test_art_keys_match_cfg(self, _result: Dict[str, Any]) -> None:
        """*test_art_keys_match_cfg()* the dimensional artifact keys are the config artifacts in order, minus the `FAIL_{1}` sink."""
        _expected = [_a.key for _a in _result["config"].artifacts if _a.key != "FAIL_{1}"]
        assert list(_result["artifacts"].keys()) == _expected

    def test_seven_pi_groups_per_art(self, _result: Dict[str, Any]) -> None:
        """*test_seven_pi_groups_per_art()* Buckingham yields `10 relevant - 3 FDUs = 7` Pi-groups for every artifact."""
        for _k, _a in _result["artifacts"].items():
            assert len(_a["pi_groups"]) == 7, f"{_k}: {len(_a['pi_groups'])} Pi-groups"

    def test_four_coefs_per_art(self, _result: Dict[str, Any]) -> None:
        """*test_four_coefs_per_art()* `len(_a["coefficients"]) == 4` (theta, sigma, eta, phi) for every artifact."""
        for _k, _a in _result["artifacts"].items():
            assert len(_a["coefficients"]) == 4, f"{_k}: {len(_a['coefficients'])} coefficients"

    def test_coef_setpoints_are_numeric(self, _result: Dict[str, Any]) -> None:
        """*test_coef_setpoints_are_numeric()* `isinstance(co["setpoint"], (int, float))` after `calculate_setpoint()`."""
        for _k, _a in _result["artifacts"].items():
            for _sym, _co in _a["coefficients"].items():
                assert isinstance(_co["setpoint"], (int, float)), (
                    f"{_k}/{_sym}: setpoint is {type(_co['setpoint']).__name__}"
                )

    def test_sens_block_present(self, _result: Dict[str, Any]) -> None:
        """*test_sens_block_present()* `len(_a["sensitivity"]) > 0` and every key starts with `SEN_`."""
        for _k, _a in _result["artifacts"].items():
            _sens = _a["sensitivity"]
            assert len(_sens) > 0
            assert all(_s.startswith("SEN_") for _s in _sens.keys())

    def test_theta_varies_per_art_baseline(self,
                                           _result_baseline: Dict[str, Any]) -> None:
        """*test_theta_varies_per_art_baseline()* `max(thetas) - min(thetas) > 0.05` after `seed_dim_from_analytic` populates per-artifact L/K ratios; uniform theta means the seed failed."""
        _thetas = [_a["coefficients"][f"\\theta_{{{_k}}}"]["setpoint"]
                   for _k, _a in _result_baseline["artifacts"].items()]
        _range = max(_thetas) - min(_thetas)
        assert _range > 0.05, f"theta range {_range} too small; seed may have failed"


class TestResultEnvelope:
    """**TestResultEnvelope** the JSON envelope written to disk is well-formed and round-trips cleanly. `tmp_path` keeps the real `data/results/dimensional/` tree untouched."""

    def test_wrt_true_writes_file(self,
                                  tmp_path,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """*test_wrt_true_writes_file()* `wrt=True` produces `tmp_path/baseline/dflt.json` with `method == "dimensional"` and 13 artifacts."""
        # redirect _ROOT alongside _RESULTS_DIR so relative_to() can express the path as repo-relative
        from src.methods import dimensional as _mod
        monkeypatch.setattr(_mod, "_ROOT", tmp_path)
        monkeypatch.setattr(_mod, "_RESULTS_DIR", tmp_path)
        _result = run_dimensional(adp="baseline", wrt=True)
        assert "profile" in _result["paths"]
        _path = tmp_path / "baseline" / "dflt.json"
        assert _path.exists(), f"expected {_path} to exist"
        _doc = json.loads(_path.read_text(encoding="utf-8"))
        assert _doc["method"] == "dimensional"
        assert _doc["profile"] == "dflt"
        assert _doc["scenario"] == "baseline"
        assert len(_doc["artifacts"]) == 13

    def test_envelope_carries_method_cfg(self,
                                         tmp_path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
        """*test_envelope_carries_method_cfg()* the written blob has `method_config.fdus` and `method_config.coefficients` so the run is self-describing on disk."""
        from src.methods import dimensional as _mod
        monkeypatch.setattr(_mod, "_ROOT", tmp_path)
        monkeypatch.setattr(_mod, "_RESULTS_DIR", tmp_path)
        run_dimensional(adp="baseline", wrt=True)
        _doc = json.loads((tmp_path / "baseline" / "dflt.json").read_text())
        assert "method_config" in _doc
        assert "fdus" in _doc["method_config"]
        assert "coefficients" in _doc["method_config"]


class TestMethodCfgOverride:
    """**TestMethodCfgOverride** the `method_cfg=` kwarg lets tests inject a trimmed spec without touching disk."""

    def test_single_coef_override(self) -> None:
        """*test_single_coef_override()* a one-entry coefficient spec yields `len(_a["coefficients"]) == 1` and `\\theta_{<key>}` per artifact."""
        _trim = {
            "seed": 42,
            "fdus": [
                {
                    "_idx": 0,
                    "_sym": "T",
                    "_fwk": "CUSTOM",
                    "_name": "Time",
                    "_unit": "s",
                    "description": "t"
                },
                {
                    "_idx": 1,
                    "_sym": "S",
                    "_fwk": "CUSTOM",
                    "_name": "Structure",
                    "_unit": "req",
                    "description": "s"
                },
                {
                    "_idx": 2,
                    "_sym": "D",
                    "_fwk": "CUSTOM",
                    "_name": "Data",
                    "_unit": "kB",
                    "description": "d"
                },
            ],
            "coefficients": [
                {
                    "symbol": "theta",
                    "expr_pattern": "{pi[6]} * {pi[3]}**(-1)",
                    "name": "Occupancy",
                    "description": "theta = L/K"
                },
            ],
            "sensitivity": {
                "val_type": "mean",
                "cat": "SYM"
            },
        }
        _result = run_dimensional(adp="baseline", wrt=False, method_cfg=_trim)
        for _k, _a in _result["artifacts"].items():
            assert len(_a["coefficients"]) == 1
            assert f"\\theta_{{{_k}}}" in _a["coefficients"]


class TestRunFromNetCfg:
    """**TestRunFromNetCfg** `run_from_netcfg` is the in-memory core that `run` wraps; both produce identical output when called with the same resolved profile."""

    def test_identity_vs_run(self, _result_baseline: Dict[str, Any]) -> None:
        """*test_identity_vs_run()* `run_from_netcfg(load_profile("baseline"))` matches `run_dimensional(adp="baseline", wrt=False)` on `nodes`, `network`, `requirements`, and every per-artifact coefficient setpoint."""
        _cfg = load_profile(adaptation="baseline")
        _via_netcfg = run_from_netcfg(_cfg, wrt=False)
        # nodes frame: same row count + matching W values per node
        assert len(_via_netcfg["nodes"]) == len(_result_baseline["nodes"])
        for _i, _row in _result_baseline["nodes"].iterrows():
            assert _via_netcfg["nodes"].iloc[_i]["key"] == _row["key"]
            assert _via_netcfg["nodes"].iloc[_i]["W"] == pytest.approx(_row["W"])
            assert _via_netcfg["nodes"].iloc[_i]["lambda"] == pytest.approx(_row["lambda"])
        # network aggregate: identical eps_e2e and W_e2e
        _net_ref = _result_baseline["network"].iloc[0]
        _net_via = _via_netcfg["network"].iloc[0]
        assert _net_via["eps_e2e"] == pytest.approx(_net_ref["eps_e2e"])
        assert _net_via["W_e2e"] == pytest.approx(_net_ref["W_e2e"])
        # requirements verdict: identical pass / value
        for _req in ("R1", "R2"):
            assert _via_netcfg["requirements"][_req]["pass"] == _result_baseline["requirements"][_req]["pass"]
            assert _via_netcfg["requirements"][_req]["value"] == pytest.approx(_result_baseline["requirements"][_req]["value"])
        # per-artifact coefficient setpoints: identical
        for _key, _a_ref in _result_baseline["artifacts"].items():
            _a_via = _via_netcfg["artifacts"][_key]
            for _sym, _c_ref in _a_ref["coefficients"].items():
                assert _a_via["coefficients"][_sym]["setpoint"] == pytest.approx(_c_ref["setpoint"])


def _coef(result: Dict[str, Any], key: str, prefix: str) -> float:
    """*_coef()* setpoint of the first coefficient on `key` whose symbol starts with `prefix` (e.g. `\\theta`)."""
    _coefs = result["artifacts"][key]["coefficients"]
    for _sym, _c in _coefs.items():
        if _sym.startswith(prefix):
            return float(_c["setpoint"])
    _msg = f"no coefficient {prefix!r} on {key}"
    raise KeyError(_msg)


class TestNoise:
    """**TestNoise** the seeded `noise` kwarg disturbs the independent variables (lambda, mu, epsilon) before the solve so the derived coefficients carry uncertainty, while `noise=None` stays bit-identical and `phi == theta` is preserved."""

    def test_no_noise_matches_clean(self, _result_baseline: Dict[str, Any]) -> None:
        """*test_no_noise_matches_clean()* `noise=None` reproduces the plain run exactly."""
        _via = run_dimensional(adp="baseline", wrt=False, noise=None)
        assert _coef(_via, "TAS_{1}", "\\theta") == pytest.approx(_coef(_result_baseline, "TAS_{1}", "\\theta"))

    def test_seeded_reproducible(self) -> None:
        """*test_seeded_reproducible()* two noisy runs at 5 % give identical coefficients (seed from method config)."""
        _a = run_dimensional(adp="baseline", wrt=False, noise=0.05)
        _b = run_dimensional(adp="baseline", wrt=False, noise=0.05)
        assert _coef(_a, "TAS_{1}", "\\eta") == pytest.approx(_coef(_b, "TAS_{1}", "\\eta"))

    def test_noise_moves_coefs(self, _result_baseline: Dict[str, Any]) -> None:
        """*test_noise_moves_coefs()* a 5 % level shifts theta off the clean value."""
        _noisy = run_dimensional(adp="baseline", wrt=False, noise=0.05)
        assert _coef(_noisy, "TAS_{1}", "\\theta") != pytest.approx(_coef(_result_baseline, "TAS_{1}", "\\theta"))

    def test_phi_eq_theta_noisy(self) -> None:
        """*test_phi_eq_theta_noisy()* phi equals theta on every artifact under noise (shared recomputed L)."""
        _noisy = run_dimensional(adp="aggregate", wrt=False, noise=0.05)
        for _key in _noisy["artifacts"]:
            assert _coef(_noisy, _key, "\\phi") == pytest.approx(_coef(_noisy, _key, "\\theta"))

    def test_coef_scope_keeps_verdict_clean(self, _result_baseline: Dict[str, Any]) -> None:
        """*test_coef_scope_keeps_verdict_clean()* noise_scope='coefficients' leaves the network metrics + R1 / R2 verdict bit-identical to the clean run (cross-method congruence) while still moving the coefficients."""
        _k = run_dimensional(adp="baseline", wrt=False, noise=0.05, noise_scope="coefficients")
        _clean_net = _result_baseline["network"].iloc[0]
        _k_net = _k["network"].iloc[0]
        assert _k_net["W_e2e"] == pytest.approx(_clean_net["W_e2e"])
        assert _k_net["eps_e2e"] == pytest.approx(_clean_net["eps_e2e"])
        assert _k["requirements"]["R1"]["pass"] == _result_baseline["requirements"]["R1"]["pass"]
        assert _k["requirements"]["R2"]["pass"] == _result_baseline["requirements"]["R2"]["pass"]
        assert _coef(_k, "TAS_{1}", "\\eta") != pytest.approx(_coef(_result_baseline, "TAS_{1}", "\\eta"))

    def test_invalid_noise_scope(self) -> None:
        """*test_invalid_noise_scope()* an unknown noise_scope raises ValueError."""
        with pytest.raises(ValueError):
            run_dimensional(adp="baseline", wrt=False, noise=0.05, noise_scope="bogus")
