# -*- coding: utf-8 -*-
"""
Module config.py
================

Profile + scenario loader for the CS-01 TAS case study. Resolves a `(profile, scenario)` selection into a flat `NetCfg` by reading the PACS-style JSONs under `data/config/profile/`.

Expected envelope shape::

    environments._setpoint  -> default scenario when --adaptation is absent
    environments._nodes     -> {scenario: [artifact_key, ... 13 entries]}
    environments._routs     -> {scenario: [[13x13 matrix]]}
    environments._labels    -> {scenario: str}
    artifacts[<key>]        -> {name, type, lambda_z, L_z, vars: {sym: {...}}}

For the `opti` profile, `_nodes[scenario]` at the two swap slots picks different artifacts per scenario: slot 5 carries `MAS_{3}` under s1 vs `MAS_{4}` under s2 / aggregate; slot 8 carries `AS_{3}` vs `AS_{4}`. Slot 10 carries `DS_{1}` in every scenario (no swap on the drug service).

Public API:
    - `ArtifactSpec` frozen per-node dataclass with setpoint accessors (`mu`, `c`, `K`, `epsilon`, `d_kb`, `d_bytes`).
    - `NetCfg` normalised view of a resolved (profile, scenario) pair.
    - `load_profile(adaptation, profile, scenario)` main loader.
    - `load_method_cfg(name)` method-config JSON reader.
    - `load_reference(name)` reference-file JSON reader.

# TODO: validate row-stochasticity on the routing matrix at load time once we add more scenarios (keeps typos from reaching the solver).
"""
# native python modules
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# data types
from typing import Any, Dict, List, Optional, Tuple

# scientific stack
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_DIR = _ROOT / "data" / "config" / "profile"
_METHOD_DIR = _ROOT / "data" / "config" / "method"
_CATALOGUE_DIR = _ROOT / "data" / "config" / "catalogue"
_REFERENCE_DIR = _ROOT / "data" / "reference"


# adaptation value -> (profile file stem, scenario name within that profile)
_ADAPTATION_TO_SOURCE: Dict[str, Tuple[str, str]] = {
    "baseline": ("dflt", "baseline"),
    "s1": ("opti", "s1"),
    "s2": ("opti", "s2"),
    "aggregate": ("opti", "aggregate"),
}


@dataclass(frozen=True)
class ArtifactSpec:
    """**ArtifactSpec** one node's full spec, resolved from the profile + scenario load.

    Attributes:
        key (str): artifact key in the PACS envelope (e.g. `TAS_{1}`).
        name (str): human-readable name.
        type_ (str): queue model string (e.g. `M/M/c/K`).
        lambda_z (float): external arrival rate entering this node.
        L_z (float): external queue length initialisation (currently 0).
        vars (Dict[str, dict]): PyDASA `Variable`-dict block, keyed by LaTeX symbol (e.g. `\\mu_{TAS_{1}}`).
    """

    key: str
    name: str
    type_: str
    lambda_z: float
    L_z: float
    vars: Dict[str, dict]

    def read_setpoint(self, prefix: str) -> float:
        """*read_setpoint()* read the `_setpoint` of the first variable whose LaTeX symbol starts with `prefix`.

        Args:
            prefix (str): LaTeX-symbol prefix including the artifact's own subscript (e.g. `\\mu_{TAS_{1}}`).

        Raises:
            KeyError: when no variable on this artifact matches the prefix.

        Returns:
            float: setpoint value of the matching variable.
        """
        for _sym, _var in self.vars.items():
            if _sym.startswith(prefix):
                return float(_var["_setpoint"])
        _msg = f"no variable with prefix {prefix!r} on {self.key}"
        raise KeyError(_msg)

    def format_sub(self) -> str:
        """*format_sub()* return the LaTeX subscript form of the artifact key.

        Since `data/config/profile/*.json` was migrated to store keys already in LaTeX form (e.g. `TAS_{1}`), this is an identity pass-through kept for API compatibility with callers written against the old `TAS_1` flat-key convention.

        Returns:
            str: artifact key verbatim (already a LaTeX subscript).
        """
        return self.key

    @property
    def mu(self) -> float:
        """*mu* service rate setpoint for this artifact."""
        return self.read_setpoint(f"\\mu_{{{self.format_sub()}}}")

    @property
    def c(self) -> int:
        """*c* server count setpoint for this artifact."""
        return int(self.read_setpoint(f"c_{{{self.format_sub()}}}"))

    @property
    def K(self) -> int:
        """*K* system capacity setpoint for this artifact."""
        return int(self.read_setpoint(f"K_{{{self.format_sub()}}}"))

    @property
    def w_proc(self) -> int:
        """*w_proc* OS worker-process count setpoint for this artifact (specs-layer only).

        Number of parallel uvicorn / waitress worker processes the apparatus spawns for this service under `dpl="multiprocess"`. Lives in the specs layer because it is an apparatus knob, not a queueing-theory parameter; analytic / stochastic / dimensional methods never read it. Named `w_proc` (not bare `w` or `\\w`) to disambiguate from calibration's `w` (worker ramp axis) and queueing-theory's `c` (parallel servers in a station).
        """
        return int(self.read_setpoint(f"w_proc_{{{self.format_sub()}}}"))

    @property
    def epsilon(self) -> float:
        """*epsilon* per-node failure rate setpoint for this artifact."""
        return self.read_setpoint(f"\\epsilon_{{{self.format_sub()}}}")

    @property
    def d_kb(self) -> float:
        """*d_kb* request data density in kB per request; reads the profile key `d_{<artifact>}` (setpoint in kB/req)."""
        return self.read_setpoint(f"d_{{{self.format_sub()}}}")

    @property
    def d_bytes(self) -> int:
        """*d_bytes* expected request size in bytes at this artifact.

        Read from the profile's `d_{<artifact>}` variable (kB/req) and converted to bytes (1 kB = 1024 bytes) so the apparatus can size mock payloads and memory buffers against a single declarative source.
        """
        return int(round(self.d_kb * 1024))


@dataclass(frozen=True)
class NetCfg:
    """**NetCfg** normalised view of a resolved (profile, scenario) pair.

    `artifacts` is a list of 13 (or 16 for the opti profile) entries in the positional order given by `environments._nodes[scenario]`. `routing` is the matching NxN matrix aligned with that order.

    Attributes:
        profile (str): profile file stem (`dflt` or `opti`).
        scenario (str): scenario name within the profile.
        label (str): human-readable scenario label (from `_labels`).
        artifacts (List[ArtifactSpec]): resolved artifact specs in positional order.
        routing (np.ndarray): NxN routing-probability matrix; `row = source`, `col = dest`.
        enforce_limits (bool): umbrella switch for runtime resource gates (read from the source layer's `enforce_limits` key; defaults to True). Plumbed into every `SvcSpec` built from this config so the launcher's K-admission gate can be disabled profile-wide without per-artifact overrides.
    """

    profile: str
    scenario: str
    label: str
    artifacts: List[ArtifactSpec]
    routing: np.ndarray
    enforce_limits: bool = True

    @property
    def n_nodes(self) -> int:
        """*n_nodes* number of artifacts in this configuration."""
        return len(self.artifacts)

    def list_node_keys(self) -> List[str]:
        """*list_node_keys()* return the artifact keys in positional order.

        Returns:
            List[str]: artifact keys aligned with `routing`.
        """
        return [_a.key for _a in self.artifacts]

    def build_lam_z_vec(self) -> np.ndarray:
        """*build_lam_z_vec()* return the external arrivals per node.

        Returns:
            np.ndarray: `(n_nodes,)` vector of external arrival rates.
        """
        return np.array([_a.lambda_z for _a in self.artifacts], dtype=float)


def _resolve_source(adaptation: Optional[str],
                    profile: Optional[str],
                    scenario: Optional[str],) -> Tuple[str, str]:
    """*_resolve_source()* map user-facing arguments to the (profile_stem, scenario_name) tuple on disk.

    Precedence:

        1. Explicit (profile, scenario) wins when both are given.
        2. `adaptation` maps via `_ADAPTATION_TO_SOURCE`.
        3. `profile` alone uses that profile's `_setpoint` scenario.
        4. Nothing given defaults to `("dflt", "baseline")`.

    Args:
        adaptation (Optional[str]): adaptation value or None.
        profile (Optional[str]): profile file stem or None.
        scenario (Optional[str]): scenario name or None.

    Raises:
        ValueError: when `adaptation` is given but not in the registry.

    Returns:
        Tuple[str, str]: `(profile_stem, scenario_name)`.
    """
    if profile and scenario:
        return profile, scenario
    if adaptation:
        if adaptation not in _ADAPTATION_TO_SOURCE:
            _allowed = sorted(_ADAPTATION_TO_SOURCE.keys())
            _msg = f"unknown adaptation {adaptation!r}; allowed: {_allowed}"
            raise ValueError(_msg)
        return _ADAPTATION_TO_SOURCE[adaptation]
    if profile:
        _doc = _read_profile(profile)
        return profile, _doc["environments"]["_setpoint"]
    return "dflt", "baseline"


def _read_profile(profile_stem: str) -> Dict[str, Any]:
    """*_read_profile()* load the raw PACS-style JSON for a profile.

    Args:
        profile_stem (str): profile file stem (e.g. `dflt`, `opti`).

    Raises:
        FileNotFoundError: when the profile file does not exist.

    Returns:
        Dict[str, Any]: parsed JSON envelope.
    """
    _path = _PROFILE_DIR / f"{profile_stem}.json"
    if not _path.exists():
        _msg = f"profile not found: {_path}"
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)


def load_profile(adaptation: Optional[str] = None,
                 profile: Optional[str] = None,
                 scenario: Optional[str] = None,
                 source: str = "artifacts") -> NetCfg:
    """*load_profile()* load a resolved `NetCfg` for one `(profile, scenario)` pair.

    Args:
        adaptation (Optional[str]): one of `baseline`, `s1`, `s2`, `aggregate`. Maps to `(profile, scenario)` via `_ADAPTATION_TO_SOURCE`.
        profile (Optional[str]): profile file stem (`dflt` or `opti`); overrides `adaptation`'s implied profile when paired with `scenario`.
        scenario (Optional[str]): explicit scenario name within the profile. Defaults to the profile's `environments._setpoint` when absent.
        source (str): top-level key to read per-node specs from. `"artifacts"` (default) reads the modelled / theoretical layer consumed by analytic / stochastic / dimensional. `"specs"` reads the practical / deployment layer used by the experiment runner; this layer can diverge on `c`, `K`, `port`, `mem_per_buffer` without contaminating the model's predictions.

    Raises:
        ValueError: when the scenario is not declared in the profile, when `source` is not `"artifacts"` or `"specs"`, or when the routing matrix shape does not match the node count.

    Returns:
        NetCfg: resolved artifacts plus the aligned routing matrix for the requested scenario.
    """
    if source not in ("artifacts", "specs"):
        _msg = f"source must be 'artifacts' or 'specs'; got {source!r}"
        raise ValueError(_msg)
    _profile, _scenario = _resolve_source(adaptation, profile, scenario)
    _doc = _read_profile(_profile)
    _env = _doc["environments"]
    if _scenario not in _env["_scenarios"]:
        _msg = (f"scenario {_scenario!r} not in {_profile}.json "
                f"(available: {_env['_scenarios']})")
        raise ValueError(_msg)
    if source not in _doc:
        _msg = (f"profile {_profile}.json has no top-level {source!r} block; "
                "schema migration may be incomplete.")
        raise ValueError(_msg)
    _node_keys = _env["_nodes"][_scenario]
    _routing = np.array(_env["_routs"][_scenario], dtype=float)
    _label = _env["_labels"].get(_scenario, "")
    _block = _doc[source]
    _enforce = bool(_block.get("enforce_limits", True))
    _artifacts: List[ArtifactSpec] = []
    for _key in _node_keys:
        _a = _block[_key]
        _artifacts.append(ArtifactSpec(_key,
                                       _a["name"],
                                       _a["type"],
                                       float(_a["lambda_z"]),
                                       float(_a["L_z"]),
                                       _a["vars"],))
    if _routing.shape != (len(_artifacts), len(_artifacts)):
        _msg = (f"routing matrix shape {_routing.shape} does not match "
                f"node count {len(_artifacts)} for scenario {_scenario!r}")
        raise ValueError(_msg)
    return NetCfg(_profile,
                  _scenario,
                  _label,
                  _artifacts,
                  _routing,
                  _enforce,)


def load_method_cfg(name: str) -> Dict[str, Any]:
    """*load_method_cfg()* load `data/config/method/<name>.json` (e.g. `stochastic`, `experiment`).

    Args:
        name (str): method config file stem.

    Raises:
        FileNotFoundError: when the method config does not exist.

    Returns:
        Dict[str, Any]: parsed JSON contents.
    """
    _path = _METHOD_DIR / f"{name}.json"
    if not _path.exists():
        _msg = f"method config not found: {_path}"
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)


def load_catalogue(name: str = "tas") -> Dict[str, Any]:
    """*load_catalogue()* load `data/config/catalogue/<name>.json`.

    The catalogue carries the published service variants for the third-party services with five variants each of `MAS`, `AS`, `DS`. Each variant is a profile of service parameters (taken from Weyns 2015 and Weyns and Iftikhar 2016).
    Args:
        name (str): catalogue file stem. Defaults to `"tas"`.

    Raises:
        FileNotFoundError: when the catalogue file does not exist.

    Returns:
        Dict[str, Any]: parsed JSON envelope. Top-level keys: `_meta`, `MAS`, `AS`, `DS`. Each service-type block maps a LaTeX-subscript service key (e.g. `MAS_{1}`) to a profile dict carrying `eps`, `W_ms`, `L`, `mu`.
    """
    _path = _CATALOGUE_DIR / f"{name}.json"
    if not _path.exists():
        _msg = f"catalogue file not found: {_path}"
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)


def load_reference(name: str = "baseline") -> Dict[str, Any]:
    """*load_reference()* load `data/reference/<name>.json`.

    These files hold case-study ground truth or validation targets that must not live inside Python (e.g. the Camara 2023 R1 / R2 thresholds in `baseline.json`). Consumed by `src.analytic.metrics` and any other module that needs an external reference value.

    Args:
        name (str): reference file stem. Defaults to `"baseline"`.

    Raises:
        FileNotFoundError: when the reference file does not exist.

    Returns:
        Dict[str, Any]: parsed JSON contents.
    """
    _path = _REFERENCE_DIR / f"{name}.json"
    if not _path.exists():
        _msg = f"reference file not found: {_path}"
        raise FileNotFoundError(_msg)
    with _path.open(encoding="utf-8") as _fh:
        return json.load(_fh)
