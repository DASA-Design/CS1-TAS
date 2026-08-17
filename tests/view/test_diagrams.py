"""Tests for `src.view.diagrams` heatmap / diffmap behaviour.

Covers two contracts that callers rely on:

- `plot_node_heatmap` aligns rows by position; each panel's y-axis labels come from its own `cname` column. Shorter panels NaN-pad so heights stay aligned across scenarios.
- `plot_node_diffmap` accepts a per-row `y_labels` override for the case where the adaptation deploys different keys than `nodes` at swap slots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.view.diagrams import (plot_method_metric_bars,
                                plot_node_diffmap,
                                plot_node_heatmap,
                                plot_pareto_front,
                                plot_verdict_grid)


matplotlib.use("Agg")


def _ndss_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two per-scenario node frames that share three keys and disagree on one swap slot.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: `(baseline_df, swap_df)`. The swap frame
            replaces `MAS_{3}` with `MAS_{4}` at the same row position.
    """
    _common = [
        {"key": "TAS_{1}", "rho": 0.1, "L": 1.0, "W": 0.01},
        {"key": "MAS_{1}", "rho": 0.2, "L": 2.0, "W": 0.02},
    ]
    _baseline = pd.DataFrame(_common + [{"key": "MAS_{3}", "rho": 0.3, "L": 3.0, "W": 0.03}])
    _swap = pd.DataFrame(_common + [{"key": "MAS_{4}", "rho": 0.4, "L": 4.0, "W": 0.04}])
    return _baseline, _swap


class TestPlotNodeHeatmap:
    """**TestPlotNodeHeatmap** contracts for `src.view.plot_node_heatmap`.

    - *test_panel_heights_stay_aligned()* rows align by position; every panel renders the same row count.
    - *test_per_panel_keys_label_y_axis()* each panel's y-axis tick labels come from its own `cname` column, so swap-slot rows carry the panel's actual service key.
    - *test_persists_png_and_svg()* both formats land when `file_path` + `fname` are given.
    """

    def test_panel_heights_stay_aligned(self) -> None:
        """*test_panel_heights_stay_aligned()* swap panel renders the same row count as the baseline panel even though it carries different keys."""
        _bl, _sw = _ndss_pair()
        _fig = plot_node_heatmap(
            ndss=[_bl, _sw],
            names=["baseline", "swap"],
            metrics=["rho", "L", "W"],
        )
        assert isinstance(_fig, Figure)
        _fig.canvas.draw()
        # build_stacked_figure prepends a title axis; body panels start at index 1.
        _panel_axes = _fig.axes[1:3]
        for _ax in _panel_axes:
            _labels = [_t.get_text() for _t in _ax.get_yticklabels() if _t.get_text()]
            assert len(_labels) == len(_bl)

    def test_per_panel_keys_label_y_axis(self, tmp_path: Path) -> None:
        """*test_per_panel_keys_label_y_axis()* each panel's y-axis carries its own `key` column values; swap-slot rows differ per panel."""
        _bl, _sw = _ndss_pair()
        _fig = plot_node_heatmap(
            ndss=[_bl, _sw],
            names=["baseline", "swap"],
            metrics=["rho", "L", "W"],
            file_path=str(tmp_path),
            fname="heatmap_pair",
        )
        _fig.canvas.draw()
        _bl_labels = [_t.get_text() for _t in _fig.axes[1].get_yticklabels()]
        _sw_labels = [_t.get_text() for _t in _fig.axes[2].get_yticklabels()]
        # Mathtext-wrapped keys appear per panel: baseline -> MAS_{3}, swap -> MAS_{4}.
        assert "$MAS_{3}$" in _bl_labels
        assert "$MAS_{4}$" in _sw_labels

    def test_persists_png_and_svg(self, tmp_path: Path) -> None:
        """*test_persists_png_and_svg()* both formats written when `file_path` is set."""
        _bl, _sw = _ndss_pair()
        plot_node_heatmap(ndss=[_bl, _sw],
                          names=["baseline", "swap"],
                          metrics=["rho", "L", "W"],
                          file_path=str(tmp_path),
                          fname="hm_smoke")
        assert (tmp_path / "hm_smoke.png").exists()
        assert (tmp_path / "hm_smoke.svg").exists()


class TestPlotNodeDiffmap:
    """**TestPlotNodeDiffmap** contracts for `src.view.plot_node_diffmap`.

    - *test_y_labels_overrides_default()* `y_labels` replaces the default y-axis ticks (which come from `nodes`).
    - *test_y_labels_validates_length()* a `y_labels` length mismatch raises `ValueError`.
    """

    def test_y_labels_overrides_default(self, tmp_path: Path) -> None:
        """*test_y_labels_overrides_default()* the y-axis shows the override labels in mathtext form."""
        _deltas = pd.DataFrame([
            {"key": "slot_0", "rho": 0.1, "L": 0.2, "W": 0.05},
            {"key": "slot_1", "rho": -0.1, "L": -0.2, "W": -0.05},
        ])
        _fig = plot_node_diffmap(deltas=_deltas,
                                 nodes=["slot_0", "slot_1"],
                                 metrics=["rho", "L", "W"],
                                 y_labels=["TAS_{1}", "MAS_{4}"],
                                 file_path=str(tmp_path),
                                 fname="dm_labels")
        _fig.canvas.draw()
        # axes[0] is the title strip; the body panel is at index 1.
        _ax = _fig.axes[1]
        _texts = [_t.get_text() for _t in _ax.get_yticklabels()]
        assert "$TAS_{1}$" in _texts
        assert "$MAS_{4}$" in _texts

    def test_y_labels_validates_length(self) -> None:
        """*test_y_labels_validates_length()* mismatched `y_labels` length raises `ValueError`."""
        _deltas = pd.DataFrame([
            {"key": "slot_0", "rho": 0.1},
            {"key": "slot_1", "rho": -0.1},
        ])
        with pytest.raises(ValueError, match="y_labels length"):
            plot_node_diffmap(deltas=_deltas,
                              nodes=["slot_0", "slot_1"],
                              y_labels=["only_one"])


def _stub_verdicts() -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Three-method × four-adaptation × two-requirement verdict map mirroring the cross-method shape consumed by `plot_verdict_grid`.

    Schema: `{method -> adp -> req -> {value, threshold, pass, units}}`. All four adaptations carry R1=FAIL; only S2 carries R2=PASS (matches the 2026-06-14 dissertation finding).
    """
    _r1_fail = {"value": 0.1732, "threshold": 0.01, "pass": False, "units": "fraction"}
    _r2_fail = {"value": 0.0288, "threshold": 0.026, "pass": False, "units": "seconds"}
    _r2_pass = {"value": 0.0204, "threshold": 0.026, "pass": True, "units": "seconds"}
    _stub: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for _method in ("analytic", "stochastic", "dimensional"):
        _stub[_method] = {}
        for _adp in ("baseline", "s1", "s2", "aggregate"):
            if _adp == "s2":
                _r2 = dict(_r2_pass)
            else:
                _r2 = dict(_r2_fail)
            _stub[_method][_adp] = {"R1": dict(_r1_fail), "R2": _r2}
    return _stub


class TestPlotVerdictGrid:
    """**TestPlotVerdictGrid** contracts for `src.view.diagrams.plot_verdict_grid`.

    - *test_renders_multi_panel_grid()* returns a `Figure` from a three-method × four-adaptation × two-requirement stub.
    - *test_persists_png_and_svg()* both formats written when `file_path` + `fname` are given.
    - *test_missing_method_raises()* a method key absent from `verdicts` triggers `ValueError`.
    - *test_missing_adaptation_raises()* an adaptation key absent from one method's map triggers `ValueError`.
    - *test_missing_requirement_raises()* a requirement key absent from one (method, adp) leaf triggers `ValueError`.
    """

    def test_renders_multi_panel_grid(self) -> None:
        """*test_renders_multi_panel_grid()* returns a `Figure` with one body subpanel per requirement."""
        _verdicts = _stub_verdicts()
        _fig = plot_verdict_grid(_verdicts,
                                 methods=["analytic", "stochastic", "dimensional"],
                                 adaptations=["baseline", "s1", "s2", "aggregate"],
                                 requirements=["R1", "R2"])
        assert isinstance(_fig, Figure)

    def test_persists_png_and_svg(self, tmp_path: Path) -> None:
        """*test_persists_png_and_svg()* both formats land when `file_path` + `fname` are given."""
        _verdicts = _stub_verdicts()
        plot_verdict_grid(_verdicts,
                          methods=["analytic", "stochastic", "dimensional"],
                          adaptations=["baseline", "s1", "s2", "aggregate"],
                          requirements=["R1", "R2"],
                          file_path=str(tmp_path),
                          fname="vg_smoke")
        assert (tmp_path / "vg_smoke.png").exists()
        assert (tmp_path / "vg_smoke.svg").exists()

    def test_missing_method_raises(self) -> None:
        """*test_missing_method_raises()* `methods=[..., "experimental"]` triggers `ValueError` because `verdicts` does not carry that method."""
        _verdicts = _stub_verdicts()
        with pytest.raises(ValueError, match="missing method"):
            plot_verdict_grid(_verdicts,
                              methods=["analytic", "experimental"],
                              adaptations=["baseline"],
                              requirements=["R1", "R2"])

    def test_missing_adaptation_raises(self) -> None:
        """*test_missing_adaptation_raises()* an unknown adaptation key triggers `ValueError`."""
        _verdicts = _stub_verdicts()
        with pytest.raises(ValueError, match="missing adaptation"):
            plot_verdict_grid(_verdicts,
                              methods=["analytic"],
                              adaptations=["s3"],
                              requirements=["R1"])

    def test_missing_requirement_raises(self) -> None:
        """*test_missing_requirement_raises()* `requirements=["R1", "R3"]` triggers `ValueError` because no leaf carries `R3`."""
        _verdicts = _stub_verdicts()
        with pytest.raises(ValueError, match="missing requirement"):
            plot_verdict_grid(_verdicts,
                              methods=["analytic"],
                              adaptations=["baseline"],
                              requirements=["R1", "R3"])


def _stub_method_nets() -> dict[str, dict[str, pd.DataFrame]]:
    """Three-method × four-adaptation network-frame fixture for `plot_method_metric_bars` smokes.

    Each leaf is a single-row DataFrame with `W_e2e` (seconds) and `eps_e2e` (fraction). Numbers match the dissertation's 2026-06-14 locked-envelope values for a representative inspection.
    """
    _rows = {
        "analytic":    [(0.0288, 0.1732), (0.0747, 0.2535), (0.0204, 0.1146), (0.0281, 0.1525)],
        "stochastic":  [(0.0286, 0.1729), (0.0720, 0.2478), (0.0208, 0.1152), (0.0282, 0.1521)],
        "dimensional": [(0.0288, 0.1732), (0.0747, 0.2535), (0.0204, 0.1146), (0.0281, 0.1525)],
    }
    _adps = ("baseline", "s1", "s2", "aggregate")
    _stub: dict[str, dict[str, pd.DataFrame]] = {}
    for _m, _vals in _rows.items():
        _stub[_m] = {}
        for _a, (_w, _e) in zip(_adps, _vals):
            _stub[_m][_a] = pd.DataFrame([{"W_e2e": _w, "eps_e2e": _e}])
    return _stub


class TestPlotMethodMetricBars:
    """**TestPlotMethodMetricBars** contracts for `src.view.diagrams.plot_method_metric_bars`.

    - *test_renders_two_panel_grid()* returns a `Figure` from a three-method × four-adaptation stub with two metrics.
    - *test_persists_png_and_svg()* both formats land when `file_path` + `fname` are given.
    - *test_missing_method_raises()* a method key absent from `nets` triggers `ValueError`.
    - *test_missing_adaptation_raises()* an adaptation key absent from one method's map triggers `ValueError`.
    - *test_missing_metric_column_raises()* a metric column absent from a network frame triggers `ValueError`.
    - *test_scales_length_mismatch_raises()* a `scales` length that does not match `metrics` triggers `ValueError`.
    """

    def test_renders_two_panel_grid(self) -> None:
        """*test_renders_two_panel_grid()* returns a `Figure` with one body subpanel per metric."""
        _nets = _stub_method_nets()
        _fig = plot_method_metric_bars(
            _nets,
            methods=["analytic", "stochastic", "dimensional"],
            adaptations=["baseline", "s1", "s2", "aggregate"],
            metrics=["W_e2e", "eps_e2e"],
            scales=[1000.0, 100.0],
            thresholds=[0.026, 0.01],
        )
        assert isinstance(_fig, Figure)

    def test_persists_png_and_svg(self, tmp_path: Path) -> None:
        """*test_persists_png_and_svg()* both formats land when `file_path` + `fname` are given."""
        _nets = _stub_method_nets()
        plot_method_metric_bars(
            _nets,
            methods=["analytic", "stochastic", "dimensional"],
            adaptations=["baseline", "s1", "s2", "aggregate"],
            metrics=["W_e2e"],
            scales=[1000.0],
            thresholds=[0.026],
            file_path=str(tmp_path),
            fname="mmb_smoke",
        )
        assert (tmp_path / "mmb_smoke.png").exists()
        assert (tmp_path / "mmb_smoke.svg").exists()

    def test_missing_method_raises(self) -> None:
        """*test_missing_method_raises()* `methods=[..., "experimental"]` triggers `ValueError` because `nets` does not carry that method."""
        _nets = _stub_method_nets()
        with pytest.raises(ValueError, match="missing method"):
            plot_method_metric_bars(_nets,
                                    methods=["analytic", "experimental"],
                                    adaptations=["baseline"],
                                    metrics=["W_e2e"])

    def test_missing_adaptation_raises(self) -> None:
        """*test_missing_adaptation_raises()* an unknown adaptation key triggers `ValueError`."""
        _nets = _stub_method_nets()
        with pytest.raises(ValueError, match="missing adaptation"):
            plot_method_metric_bars(_nets,
                                    methods=["analytic"],
                                    adaptations=["s3"],
                                    metrics=["W_e2e"])

    def test_missing_metric_column_raises(self) -> None:
        """*test_missing_metric_column_raises()* an unknown metric column triggers `ValueError`."""
        _nets = _stub_method_nets()
        with pytest.raises(ValueError, match="missing metric column"):
            plot_method_metric_bars(_nets,
                                    methods=["analytic"],
                                    adaptations=["baseline"],
                                    metrics=["chi_out"])

    def test_scales_length_mismatch_raises(self) -> None:
        """*test_scales_length_mismatch_raises()* `scales` length must match `metrics` length."""
        _nets = _stub_method_nets()
        with pytest.raises(ValueError, match="scales length"):
            plot_method_metric_bars(_nets,
                                    methods=["analytic"],
                                    adaptations=["baseline"],
                                    metrics=["W_e2e", "eps_e2e"],
                                    scales=[1000.0])


def _pareto_points() -> pd.DataFrame:
    """Small candidate cloud spanning two retry groups for the Pareto smokes."""
    _rows = [
        {"eps_e2e": 0.08, "W_e2e": 0.0188, "retry_label": "selection-only"},
        {"eps_e2e": 0.05, "W_e2e": 0.0192, "retry_label": "selection-only"},
        {"eps_e2e": 0.0003, "W_e2e": 0.0198, "retry_label": "retry-all"},
        {"eps_e2e": 0.0006, "W_e2e": 0.0195, "retry_label": "retry-all"},
    ]
    return pd.DataFrame(_rows)


class TestPlotParetoFront:
    """**TestPlotParetoFront** contracts for `src.view.diagrams.plot_pareto_front`."""

    def test_renders_figure(self) -> None:
        """*test_renders_figure()* returns a `Figure` for a two-group cloud with a winner marker."""
        _fig = plot_pareto_front(
            _pareto_points(),
            x_col="eps_e2e",
            y_col="W_e2e",
            group_col="retry_label",
            x_label="eps", y_label="W",
            x_scale=100.0, y_scale=1000.0, x_log=True,
            x_threshold=0.01, y_threshold=0.026,
            winner_xy=(0.0003, 0.0198),
        )
        assert isinstance(_fig, Figure)

    def test_persists_png_and_svg(self, tmp_path: Path) -> None:
        """*test_persists_png_and_svg()* writes both formats under `file_path`."""
        plot_pareto_front(
            _pareto_points(),
            x_col="eps_e2e", y_col="W_e2e", group_col="retry_label",
            x_label="eps", y_label="W",
            file_path=str(tmp_path), fname="pareto_smoke",
        )
        assert (tmp_path / "pareto_smoke.png").exists()
        assert (tmp_path / "pareto_smoke.svg").exists()

    def test_missing_column_raises(self) -> None:
        """*test_missing_column_raises()* a missing axis / group column is rejected."""
        with pytest.raises(ValueError, match="missing column"):
            plot_pareto_front(_pareto_points(),
                              x_col="nope", y_col="W_e2e", group_col="retry_label",
                              x_label="x", y_label="y")
