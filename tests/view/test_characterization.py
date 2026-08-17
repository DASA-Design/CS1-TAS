"""Tests for `src.view.characterization`.

Smoke checks against synthetic envelopes: each plotter returns a `Figure`, accepts the documented kwargs, and persists `<fname>.png` + `<fname>.svg` when `file_path` is given. The notebook validates real-data rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pytest
from matplotlib.figure import Figure

from src.view.characterization import (
    plot_calibration_summary,
    plot_envelope_overlay,
    plot_handler_scaling,
    plot_jitter,
    plot_loopback,
    plot_rate_sweep,
    plot_timer,
    plot_workers_scaling,
)


# Use the headless Agg backend so tests do not pop a window.
matplotlib.use("Agg")


def _synth_envelope(*,
                    host: str = "h-test",
                    dpl: str = "localhost") -> dict[str, Any]:
    """Build a fully populated synthetic envelope inside the noise-floor band.

    Returns:
        dict[str, Any]: an envelope with all probe blocks + a passing gate verdict, suitable for smoke-testing every plotter.
    """
    _env: dict[str, Any] = {
        "version": "1.0",
        "run_id": "calib_test",
        "host": host,
        "dpl": dpl,
        "framework": "fastapi",
        "wsgi_server": None,
        "started_ts": 0.0,
        "finished_ts": 1.0,
        "timer": {
            "samples_n": 100,
            "median_ns": 100,
            "mean_ns": 102.0,
            "std_ns": 4.5,
            "min_ns": 90,
            "max_ns": 105,
        },
        "jitter": {
            "samples_n": 50,
            "target_us": 1000,
            "min_us": 1005.0,
            "max_us": 1300.0,
            "mean_us": 1015.0,
            "std_us": 25.0,
            "median_us": 1010.0,
            "p95_us": 1100.0,
            "p99_us": 1200.0,
        },
        "loopback": {
            "samples_n": 50,
            "payload_bytes": 64,
            "min_us": 95.0,
            "max_us": 115.0,
            "mean_us": 101.0,
            "std_us": 3.0,
            "median_us": 100.0,
            "p95_us": 102.0,
            "p99_us": 110.0,
        },
        "handler_scaling": {
            "concurs": [1, 2, 4],
            "stats": {
                "1": {
                    "samples_n": 10,
                    "min_us": 240.0,
                    "max_us": 320.0,
                    "mean_us": 260.0,
                    "std_us": 15.0,
                    "median_us": 250.0,
                    "p95_us": 290.0,
                    "p99_us": 310.0,
                },
                "2": {
                    "samples_n": 20,
                    "min_us": 270.0,
                    "max_us": 380.0,
                    "mean_us": 295.0,
                    "std_us": 22.0,
                    "median_us": 285.0,
                    "p95_us": 340.0,
                    "p99_us": 370.0,
                },
                "4": {
                    "samples_n": 40,
                    "min_us": 320.0,
                    "max_us": 540.0,
                    "mean_us": 360.0,
                    "std_us": 38.0,
                    "median_us": 340.0,
                    "p95_us": 450.0,
                    "p99_us": 510.0,
                },
            },
        },
        "rate": {
            "ramp": [50, 100, 150],
            "per_rate": [
                {
                    "rate": 50,
                    "total": 50,
                    "errors": 0,
                    "loss_pct": 0.0,
                    "median_us": 200.0,
                    "p95_us": 250.0,
                    "p99_us": 300.0,
                },
                {
                    "rate": 100,
                    "total": 100,
                    "errors": 0,
                    "loss_pct": 0.0,
                    "median_us": 220.0,
                    "p95_us": 270.0,
                    "p99_us": 320.0,
                },
                {
                    "rate": 150,
                    "total": 150,
                    "errors": 0,
                    "loss_pct": 0.0,
                    "median_us": 240.0,
                    "p95_us": 290.0,
                    "p99_us": 340.0,
                },
            ],
            "target_urls": ["http://127.0.0.1:8001/"],
            "target_loss_pct": 5.0,
            "max_p95_latency_us": 100_000.0,
            "saturated": False,
            "saturation_rate": None,
            "reason": "below all thresholds",
        },
        "gate": {
            "passed": True,
            "noise_floor_pct": 5.0,
            "checks": {
                "timer": {
                    "passed": True,
                    "value_pct": 5.0,
                    "limit_pct": 5.0,
                    "reason": "ok",
                },
                "jitter": {
                    "passed": True,
                    "value_pct": 1.0,
                    "limit_pct": 5.0,
                    "reason": "ok",
                },
                "loopback": {
                    "passed": True,
                    "value_pct": 2.0,
                    "limit_pct": 5.0,
                    "reason": "ok",
                },
                "handler_scaling": {
                    "passed": True,
                    "value_pct": 4.0,
                    "limit_pct": 5.0,
                    "reason": "ok",
                },
            },
        },
    }
    return _env


def _workers_step(*,
                  n_workers: int,
                  total: int,
                  per_worker_rps: float,
                  efficiency_pct: float) -> dict[str, Any]:
    """Build one `per_step` row for a synthetic `workers_scaling` block.

    Args:
        n_workers (int): worker count at this step.
        total (int): completed requests recorded at this step.
        per_worker_rps (float): per-worker rps to embed in the row.
        efficiency_pct (float): per-worker efficiency relative to n=1.

    Returns:
        dict[str, Any]: a per-step row with the standard driver + derived fields.
    """
    _ans: dict[str, Any] = {
        "n_workers": n_workers,
        "rate_target": n_workers * 200,
        "total": total,
        "errors": 0,
        "loss_pct": 0.0,
        "min_us": 100.0,
        "max_us": 200.0,
        "mean_us": 120.0,
        "std_us": 10.0,
        "median_us": 110.0,
        "p95_us": 180.0,
        "p99_us": 195.0,
        "actual_rps": float(total),
        "per_worker_rps": per_worker_rps,
        "efficiency_pct": efficiency_pct,
    }
    return _ans


def _add_workers_block(env: dict[str, Any]) -> dict[str, Any]:
    """Stamp a populated `workers_scaling` block onto a synthetic envelope.

    Args:
        env (dict[str, Any]): the envelope to mutate (and return).

    Returns:
        dict[str, Any]: the same envelope with `workers_scaling` populated for a multiprocess run with knee at n=4.
    """
    env["workers_scaling"] = {
        "ramp": [1, 2, 4],
        "per_step": [
            _workers_step(n_workers=1, total=1000,
                          per_worker_rps=200.0, efficiency_pct=100.0),
            _workers_step(n_workers=2, total=1900,
                          per_worker_rps=190.0, efficiency_pct=95.0),
            _workers_step(n_workers=4, total=3200,
                          per_worker_rps=160.0, efficiency_pct=80.0),
        ],
        "rate_per_worker": 200,
        "per_step_s": 5.0,
        "min_eff_pct": 80.0,
        "stable_workers": 4,
        "reason": "all steps within efficiency band (max n=4)",
    }
    return env


class TestCharacterization:
    """Smoke tests for every plotter in `src.view.characterization`."""

    @pytest.mark.parametrize("plotter,fname", [
        (plot_timer, "timer"),
        (plot_jitter, "jitter"),
        (plot_loopback, "loopback"),
        (plot_handler_scaling, "scaling"),
        (plot_rate_sweep, "rate"),
        (plot_workers_scaling, "workers"),
        (plot_calibration_summary, "summary"),
    ])
    def test_single(self,
                    tmp_path: Path,
                    plotter: Any,
                    fname: str) -> None:
        """Each per-envelope plotter returns a Figure and saves PNG+SVG when `file_path` is given.

        Args:
            tmp_path (Path): pytest-provided scratch directory.
            plotter (Any): the public plotter under test (parametrised).
            fname (str): file stem the plotter saves under.
        """
        _env = _add_workers_block(_synth_envelope(dpl="multiprocess"))
        _fig = plotter(_env, file_path=str(tmp_path), fname=fname)
        assert isinstance(_fig, Figure)
        assert (tmp_path / f"{fname}.png").is_file()
        assert (tmp_path / f"{fname}.svg").is_file()

    def test_summary_localhost(self, tmp_path: Path) -> None:
        """A localhost envelope renders with handler_scaling in the bottom-left slot.

        Args:
            tmp_path (Path): pytest-provided scratch directory.
        """
        _env = _synth_envelope(dpl="localhost")
        _fig = plot_calibration_summary(_env,
                                        file_path=str(tmp_path),
                                        fname="summary_lo")
        assert isinstance(_fig, Figure)
        assert (tmp_path / "summary_lo.png").is_file()

    def test_overlay_mixed(self, tmp_path: Path) -> None:
        """Localhost-vs-multiprocess overlay falls back to handler_scaling in the bottom-left slot.

        Args:
            tmp_path (Path): pytest-provided scratch directory.
        """
        _envs = {
            "localhost": _synth_envelope(dpl="localhost"),
            "multiprocess": _synth_envelope(dpl="multiprocess"),
        }
        _fig = plot_envelope_overlay(_envs,
                                     file_path=str(tmp_path),
                                     fname="overlay")
        assert isinstance(_fig, Figure)
        assert (tmp_path / "overlay.png").is_file()
        assert (tmp_path / "overlay.svg").is_file()

    def test_overlay_all_workers(self, tmp_path: Path) -> None:
        """When all envelopes carry workers data, overlay renders workers_scaling in the bottom-left slot.

        Args:
            tmp_path (Path): pytest-provided scratch directory.
        """
        _envs = {
            "multi-A": _add_workers_block(_synth_envelope(dpl="multiprocess")),
            "multi-B": _add_workers_block(_synth_envelope(dpl="multiprocess")),
        }
        _fig = plot_envelope_overlay(_envs,
                                     file_path=str(tmp_path),
                                     fname="overlay_w")
        assert isinstance(_fig, Figure)
        assert (tmp_path / "overlay_w.png").is_file()

    def test_no_save_without_file_path(self) -> None:
        """When file_path is omitted the plotter still returns a Figure (no on-disk side effect)."""
        _env = _synth_envelope()
        _fig = plot_timer(_env)
        assert isinstance(_fig, Figure)

    def test_handles_missing_blocks(self) -> None:
        """When a probe block is empty, the plotter still returns a Figure rather than raising."""
        _empty: dict[str, Any] = {}
        for _plotter in (plot_timer,
                         plot_jitter,
                         plot_loopback,
                         plot_handler_scaling,
                         plot_rate_sweep,
                         plot_workers_scaling):
            _fig = _plotter(_empty)
            assert isinstance(_fig, Figure)
