# -*- coding: utf-8 -*-
"""
Module test_config.py
=====================

Sanity checks for the profile + scenario loader in `src.io.config`.

    - **TestResolution**: the four adaptation aliases (`baseline`, `s1`, `s2`, `aggregate`) resolve to the correct (profile, scenario) pair and surface the expected artifacts at the three swap slots (5, 8, 10).
    - **TestSetpointFallback**: the `_setpoint` fallback kicks in when the caller omits `adaptation` and `scenario`.
    - **TestArtifactSpec**: the `ArtifactSpec` properties (`mu`, `lambda_z`) read the Variable dict correctly, including the dflt / opti service swap at slot 5.
    - **TestErrors**: unknown adaptations and unknown scenarios fail loud at load time.
    - **TestSourceSwitch**: `source="artifacts"` (default) vs `source="specs"` flips between the model layer and the deployment layer.
    - **TestEnforceLimits**: the `enforce_limits` umbrella key flows through `NetCfg.enforce_limits`.
    - **TestLoadCatalogue**: `load_catalogue()` reads `data/config/catalogue.json` and returns the Table 7.3 service variants.

# TODO: extend with a regression case for a profile that omits `_labels[scenario]` (optional field) once we add new scenarios.
"""
# data types
from typing import List

# testing framework
import pytest

# module under test
from src.io import ArtifactSpec, load_catalogue, load_profile


class TestResolution:
    """**TestResolution** the four adaptation aliases resolve to the correct (profile, scenario) pair and surface the expected artifacts at the three swap slots (5, 8, 10)."""

    def test_baseline_resolves_dflt(self) -> None:
        """*test_baseline_resolves_dflt()* `adaptation="baseline"` -> `profile="dflt"`, `scenario="baseline"`, `n_nodes == 14` (13 services + the FAIL sink)."""
        _cfg = load_profile(adaptation="baseline")
        assert _cfg.profile == "dflt"
        assert _cfg.scenario == "baseline"
        assert _cfg.n_nodes == 14

    def test_s1_uses_dflt_swaps(self) -> None:
        """*test_s1_uses_dflt_swaps()* `s1` runs on `profile="opti"` with dflt-variant keys at slots 5/8/10 (`MAS_{3}` / `AS_{3}` / `DS_{1}`)."""
        _cfg = load_profile(adaptation="s1")
        assert _cfg.profile == "opti"
        assert _cfg.scenario == "s1"
        _keys = _cfg.list_node_keys()
        assert _keys[5] == "MAS_{3}"
        assert _keys[8] == "AS_{3}"
        assert _keys[10] == "DS_{1}"

    def test_s2_uses_opti_swaps(self) -> None:
        """*test_s2_uses_opti_swaps()* `s2` swaps in the opti-variant keys at slots 5/8/10 (`MAS_{4}` / `AS_{4}` / `DS_{1}`)."""
        _cfg = load_profile(adaptation="s2")
        assert _cfg.profile == "opti"
        _keys = _cfg.list_node_keys()
        assert _keys[5] == "MAS_{4}"
        assert _keys[8] == "AS_{4}"
        assert _keys[10] == "DS_{1}"

    def test_aggregate_uses_opti_swaps(self) -> None:
        """*test_aggregate_uses_opti_swaps()* `aggregate` combines opti routing with opti-variant keys at the swap slots."""
        _cfg = load_profile(adaptation="aggregate")
        assert _cfg.scenario == "aggregate"
        _keys = _cfg.list_node_keys()
        assert _keys[5] == "MAS_{4}"
        assert _keys[10] == "DS_{1}"


class TestSetpointFallback:
    """**TestSetpointFallback** the loader falls back to the profile's declared `_setpoint` when the caller does not pin a scenario."""

    def test_no_args_hits_dflt_baseline(self) -> None:
        """*test_no_args_hits_dflt_baseline()* `load_profile()` -> `profile="dflt"`, `scenario="baseline"` (hard default)."""
        _cfg = load_profile()
        assert _cfg.profile == "dflt"
        assert _cfg.scenario == "baseline"

    def test_profile_only_uses_setpoint(self) -> None:
        """*test_profile_only_uses_setpoint()* `load_profile(profile="opti")` falls through to `opti.json::environments._setpoint == "aggregate"`."""
        _cfg = load_profile(profile="opti")
        assert _cfg.profile == "opti"
        assert _cfg.scenario == "aggregate"


class TestArtifactSpec:
    """**TestArtifactSpec** `ArtifactSpec` properties resolve their PyDASA Variable-dict setpoints across both profiles, including the dflt / opti service swap at slot 5."""

    def test_mu_at_dflt_mas_3(self) -> None:
        """*test_mu_at_dflt_mas_3()* `MAS_{3}.mu == 150.0` in the dflt profile (published baseline value)."""
        _cfg = load_profile(adaptation="baseline")
        _mas_3 = next(_a for _a in _cfg.artifacts if _a.key == "MAS_{3}")
        assert _mas_3.mu == 150.0

    def test_mu_at_opti_mas_4(self) -> None:
        """*test_mu_at_opti_mas_4()* `MAS_{4}.mu == 875.0` in the opti profile (upgrade slot)."""
        _cfg = load_profile(adaptation="aggregate")
        _mas_4 = next(_a for _a in _cfg.artifacts if _a.key == "MAS_{4}")
        assert _mas_4.mu == 875.0

    def test_lambda_z_only_at_entry(self) -> None:
        """*test_lambda_z_only_at_entry()* exactly one artifact has `lambda_z > 0`, and its key is `TAS_{1}`."""
        _cfg = load_profile(adaptation="baseline")
        _entries: List[ArtifactSpec] = []
        for _a in _cfg.artifacts:
            if _a.lambda_z > 0:
                _entries.append(_a)
        assert len(_entries) == 1
        assert _entries[0].key == "TAS_{1}"
        assert _entries[0].lambda_z > 0


class TestErrors:
    """**TestErrors** bad inputs raise `ValueError` at load time, before any solver is touched."""

    def test_unknown_adaptation_raises(self) -> None:
        """*test_unknown_adaptation_raises()* an adaptation absent from `_ADAPTATION_TO_SOURCE` raises `ValueError` matching `"unknown adaptation"`."""
        with pytest.raises(ValueError, match="unknown adaptation"):
            load_profile(adaptation="xyz")

    def test_unknown_scenario_raises(self) -> None:
        """*test_unknown_scenario_raises()* a scenario absent from `_scenarios` raises `ValueError` matching `"not in"`."""
        with pytest.raises(ValueError, match="not in"):
            load_profile(profile="dflt", scenario="bogus")


class TestSourceSwitch:
    """**TestSourceSwitch** the `source` kwarg picks between the modelled (`artifacts`) and practical (`specs`) layers, defaulting to `artifacts`."""

    def test_default_source_is_artifacts(self) -> None:
        """*test_default_source_is_artifacts()* `load_profile(adaptation="baseline")` matches `source="artifacts"` on key / mu / c / K per artifact."""
        _cfg_default = load_profile(adaptation="baseline")
        _cfg_explicit = load_profile(adaptation="baseline", source="artifacts")
        assert _cfg_default.n_nodes == _cfg_explicit.n_nodes
        for _a, _b in zip(_cfg_default.artifacts, _cfg_explicit.artifacts):
            assert _a.key == _b.key
            assert _a.mu == _b.mu
            assert _a.c == _b.c
            assert _a.K == _b.K

    def test_specs_source_loads(self) -> None:
        """*test_specs_source_loads()* `source="specs"` loads with the same node count and key order as `source="artifacts"`."""
        _cfg_arts = load_profile(adaptation="baseline", source="artifacts")
        _cfg_specs = load_profile(adaptation="baseline", source="specs")
        assert _cfg_specs.n_nodes == _cfg_arts.n_nodes
        for _a, _b in zip(_cfg_arts.artifacts, _cfg_specs.artifacts):
            assert _a.key == _b.key

    def test_specs_source_per_adp(self) -> None:
        """*test_specs_source_per_adp()* `source="specs"` resolves to `n_nodes == 14` (13 services + FAIL sink) under every adaptation alias."""
        for _adp in ("baseline", "s1", "s2", "aggregate"):
            _cfg = load_profile(adaptation=_adp, source="specs")
            assert _cfg.n_nodes == 14

    def test_invalid_source_raises(self) -> None:
        """*test_invalid_source_raises()* `source="bogus"` raises `ValueError` matching `"source must be"`."""
        with pytest.raises(ValueError, match="source must be"):
            load_profile(adaptation="baseline", source="bogus")


class TestEnforceLimits:
    """**TestEnforceLimits** the `enforce_limits` umbrella key (top-level under `specs`) flows through `NetCfg.enforce_limits`."""

    def test_specs_layer_enforce_limits_true(self) -> None:
        """*test_specs_layer_enforce_limits_true()* `cfg.enforce_limits is True` for every adaptation under `source="specs"` (both profiles ship the umbrella key as `true`)."""
        for _adp in ("baseline", "s1", "s2", "aggregate"):
            _cfg = load_profile(adaptation=_adp, source="specs")
            assert _cfg.enforce_limits is True

    def test_artifacts_layer_defaults_enforce_limits_true(self) -> None:
        """*test_artifacts_layer_defaults_enforce_limits_true()* the `artifacts` block carries no umbrella key; the loader defaults `cfg.enforce_limits` to True."""
        _cfg = load_profile(adaptation="baseline", source="artifacts")
        assert _cfg.enforce_limits is True


class TestLoadCatalogue:
    """**TestLoadCatalogue** `load_catalogue()` reads `data/config/catalogue.json` and returns the Table 7.3 service variants keyed by service type."""

    def test_keys(self) -> None:
        """*test_keys()* the catalogue carries `MAS`, `AS`, `DS` blocks with five service entries each."""
        _cat = load_catalogue()
        for _svc_type in ("MAS", "AS", "DS"):
            assert _svc_type in _cat
            assert len(_cat[_svc_type]) == 5

    def test_mas_3_value(self) -> None:
        """*test_mas_3_value()* `MAS_{3}` carries the published bottleneck values `mu = 150` and `eps = 0.18`."""
        _cat = load_catalogue()
        _mas3 = _cat["MAS"]["MAS_{3}"]
        assert _mas3["mu"] == 150
        assert _mas3["eps"] == 0.18

    def test_unknown_name_raises(self) -> None:
        """*test_unknown_name_raises()* `FileNotFoundError` is raised when `name` points at a catalogue file that does not exist."""
        with pytest.raises(FileNotFoundError, match="catalogue file not found"):
            load_catalogue(name="this_catalogue_does_not_exist")
