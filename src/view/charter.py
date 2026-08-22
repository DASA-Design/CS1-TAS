# -*- coding: utf-8 -*-
"""
Module view/charter.py
======================

Dimensionless-coefficient charters (yoly family) for the CS-01 TAS case study. Sibling to `src.view.diagrams` (queueing topology, heatmaps, network bars); this module renders the coefficient cloud (theta, sigma, eta, phi) across a configuration sweep.

Thin orchestrator: every helper, constant, palette, scatter primitive, and layout helper lives in `src.view.common`. Each public function picks a `FigureLayout` and calls `build_stacked_figure(layout)`. The body is populated via the migrated `_paint_*_yoly` + `_compute_*` helpers; the footer is applied via `render_footer_*`. Axis cosmetics are caller-driven through `AxisSpec` kwargs.

Five plotters, all conform to the design contract (title strip / body grid / footer strip; footer width clipped to body width):

    - `plot_yoly_chart(coeff_data, ...)` single-queue 2D yoly chart: 2x2 grid of coefficient planes (theta-sigma, theta-eta, sigma-eta, theta-phi).
    - `plot_yoly_space(coeff_data, ...)` single 3D yoly cloud (theta x sigma x eta) for one artifact / system.
    - `plot_yoly_arts_hist(coeff_data, ...)` per-node coefficient distributions: 3 x ceil(N/3) outer grid of nodes, each cell a 2x2 histogram subgrid.
    - `plot_yoly_arts_behaviour(coeff_data, ...)` per-node 3D yoly clouds laid out in a 3 x ceil(N/3) outer grid.
    - `plot_yoly_arts_charts(coeff_data, ...)` per-node 2D yoly planes laid out in a 3 x ceil(N/3) outer grid; each cell carries a 2x2 inner subgrid of coefficient planes.

`plot_yoly_space` was named `plot_system_behaviour` in the OLD module; the rename matches the `plot_yoly_*` family. `plot_yoly_arts_hist` was named `plot_arts_distributions`; same family-prefix consistency. Backwards-compatible aliases live in `src/view/__init__.py`.
"""
# native python modules
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# scientific stack
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# shared view helpers (every helper + constant lives in common; this module only orchestrates)
from src.view.common import (
    BodySpec,
    FigureLayout,
    _BAR_BLUE,
    _BAR_ORANGE,
    _DEFAULT_LABELS,
    _GRID_PANEL,
    _GRID_PANEL_3D,
    _LBL_STY_2D_GRID,
    _LBL_STY_2D_SINGLE,
    _LBL_STY_3D_GRID,
    _LBL_STY_3D_SINGLE,
    _LBL_STYLE,
    _SUPTITLE_STYLE,
    _TEXT_BLACK,
    _TICK_STY_3D_GRID,
    _TICK_STY_3D_SINGLE,
    _TICK_STYLE,
    _YOLY_PANELS,
    _apply_logscale,
    _apply_sci_format,
    _build_coef_map,
    _compute_grid_dims,
    _compute_node_pos,
    _format_node_header,
    _generate_color_map,
    _paint_groups_2d_yoly,
    _paint_groups_3d_yoly,
    _paint_single_2d_yoly,
    _paint_single_3d_yoly,
    _pick_layout,
    _resolve_groups,
    _save_figure,
    _style_3d_panes,
    build_stacked_figure,
    render_footer_legend,
)


# Requirement-bound line colour for the DASA selection surface; firebrick, matching the
# threshold idiom in src.view.diagrams (kept local so charter does not import from diagrams).
_SELECT_BOUND_RED = "#B22222"
# de-emphasis tint for candidates that fall outside the viable box
_SELECT_OUT_GREY = "#B0B0B0"


# ---------------------------------------------------------------------------
# Local helpers (charter-specific; not shared with characterization or diagrams)
# ---------------------------------------------------------------------------


def _fmt_sci_mathtext(value: float, decimals: int = 2) -> str:
    """*_fmt_sci_mathtext()* format `value` as `mantissa \\times 10^{exp}` mathtext fragment.

    Renders inside an outer `$...$` so the exponent goes through matplotlib's mathtext parser as a proper superscript instead of the raw `e-02` alphanumeric suffix `:.Ne` would produce.

    Args:
        value (float): scalar to format. NaN/inf return the literal `"\\mathrm{NaN}"` / `"\\infty"` string.
        decimals (int): mantissa decimals. Defaults to 2.

    Returns:
        str: e.g. `r"1.23 \\times 10^{-2}"` for `value=0.0123`. Wrap in `$...$` at the call site.
    """
    if not np.isfinite(value):
        if np.isnan(value):
            return r"\mathrm{NaN}"
        return r"\infty"
    if value == 0.0:
        return "0"
    _exp = int(np.floor(np.log10(abs(value))))
    _mantissa = value / (10.0 ** _exp)
    return rf"{_mantissa:.{decimals}f} \times 10^{{{_exp}}}"


def _resolve_yoly_inputs(labels: Optional[Dict[str, str]],
                         paths: Optional[Dict[str, str]],
                         scenarios: Optional[Dict[str, str]]
                         ) -> Tuple[Optional[Dict[str, str]],
                                    str,
                                    Dict[str, str]]:
    """*_resolve_yoly_inputs()* return the trio every yoly plotter needs at the top of its body.

    Args:
        labels (Optional[Dict[str, str]]): caller-supplied coefficient-label override.
        paths (Optional[Dict[str, str]]): multi-path grouping.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping.

    Raises:
        ValueError: If both `paths=` and `scenarios=` are provided (propagated from `_resolve_groups`).

    Returns:
        Tuple[Optional[Dict[str, str]], str, Dict[str, str]]: `(groups, legend_title, lbl_map)`.
    """
    _groups, _legend_title = _resolve_groups(paths, scenarios)
    _lbl_map = {**_DEFAULT_LABELS, **(labels or {})}
    return _groups, _legend_title, _lbl_map


def _resolve_name_map(names: Optional[Dict[str, str]]) -> Dict[str, str]:
    """*_resolve_name_map()* return `names` when given, else an empty dict.

    Args:
        names (Optional[Dict[str, str]]): caller-supplied node-name override.

    Returns:
        Dict[str, str]: the caller's dict or an empty fallback.
    """
    if names is not None:
        return names
    return {}


def _handle_empty_meta_grid(fig: Figure,
                            body_ax: Any,
                            file_path: Optional[str],
                            fname: Optional[str],
                            verbose: bool) -> Figure:
    """*_handle_empty_meta_grid()* close an empty meta-grid figure cleanly when the caller passed an empty `coeff_data`.

    Args:
        fig (Figure): figure returned by `build_stacked_figure`.
        body_ax (Any): body host axis (will be hidden).
        file_path (Optional[str]): directory to save into; persistence delegated to `_save_figure`.
        fname (Optional[str]): filename stem.
        verbose (bool): if True, prints one save-path message per format.

    Returns:
        Figure: the (now-empty) figure.
    """
    body_ax.axis("off")
    _save_figure(fig, file_path, fname, verbose=verbose)
    return fig


def _anchor_cell_header(fig: Figure,
                        gs_cell: Any,
                        text: str,
                        *,
                        fontsize: int,
                        dy: float = 0.008) -> None:
    """*_anchor_cell_header()* draw a per-cell header anchored to the gridspec cell's top edge in figure coordinates.

    Args:
        fig (Figure): figure to draw on.
        gs_cell (Any): gridspec cell (`gs_main[row, col]`); must support `.get_position(fig)`.
        text (str): header text (mathtext OK).
        fontsize (int): text font size.
        dy (float): vertical offset above the cell's top edge in figure-fraction units. Defaults to 0.008.
    """
    _cell_pos = gs_cell.get_position(fig)
    _title_x = (_cell_pos.x0 + _cell_pos.x1) / 2.0
    _title_y = _cell_pos.y1 + dy
    fig.text(_title_x, _title_y, text,
             ha="center",
             va="bottom",
             fontsize=fontsize,
             fontweight="bold",
             color=_TEXT_BLACK,
             transform=fig.transFigure)


def _apply_yoly_panel_axes(ax: Any,
                           x_key: str,
                           y_key: str,
                           lbl_map: Dict[str, str],
                           lbl_style: Dict[str, Any],
                           panel_title: str,
                           logscale: Union[bool, List[bool]]) -> None:
    """*_apply_yoly_panel_axes()* set grid, ticks, spines, sci format, log toggle, axis labels, and panel title on a single yoly 2D panel.

    Sigma values track theta closely (Little's law: lambda*W = L on a static system). Bumping sigma's scientific format to 4 significant figures keeps small values from collapsing to "1.0e+00".

    Args:
        ax (Any): matplotlib 2D axes.
        x_key (str): short coefficient name on the x-axis (`"theta"`, `"sigma"`, ...).
        y_key (str): short coefficient name on the y-axis.
        lbl_map (Dict[str, str]): coefficient -> display-label map.
        lbl_style (Dict[str, Any]): style dict for axis labels (`_LBL_STY_2D_SINGLE` or `_LBL_STY_2D_GRID`).
        panel_title (str): panel title string.
        logscale (Union[bool, List[bool]]): per-axis log toggle.
    """
    ax.grid(True, **_GRID_PANEL)
    ax.tick_params(**_TICK_STYLE)
    for _spine in ax.spines.values():
        _spine.set_edgecolor(_TEXT_BLACK)

    # uniform 2-sig-fig sci format on every axis of every panel; the legacy sig=4 special-case for sigma was needed when sigma ~ theta on closed-form solves (Little's law identity), but after the sigma = lambda*W/K formula correction sigma differs enough to read clearly at sig=2
    _apply_sci_format(ax, axes_list=["x", "y"])
    _apply_logscale(ax, logscale)

    ax.set_xlabel(lbl_map[x_key], **lbl_style)
    ax.set_ylabel(lbl_map[y_key], **lbl_style)
    ax.set_title(panel_title,
                 fontsize=17,
                 pad=-10,
                 **_LBL_STYLE)


def _apply_yoly_3d_axes(ax: Any,
                        lbl_map: Dict[str, str],
                        lbl_style: Dict[str, Any],
                        tick_style: Dict[str, Any],
                        elev: float,
                        azim: float,
                        logscale: Union[bool, List[bool]]) -> None:
    """*_apply_yoly_3d_axes()* set axis labels, log toggle, view angle, pane styling, grid, sci format, and ticks on a 3D yoly axes.

    Args:
        ax (Any): matplotlib 3D axes.
        lbl_map (Dict[str, str]): coefficient -> display-label map.
        lbl_style (Dict[str, Any]): label-style dict (`_LBL_STY_3D_SINGLE` or `_LBL_STY_3D_GRID`).
        tick_style (Dict[str, Any]): tick-style dict (`_TICK_STY_3D_SINGLE` or `_TICK_STY_3D_GRID`).
        elev (float): viewing elevation passed to `ax.view_init`.
        azim (float): viewing azimuth passed to `ax.view_init`.
        logscale (Union[bool, List[bool]]): per-axis log toggle.
    """
    ax.set_xlabel(lbl_map["theta"], **lbl_style)
    ax.set_ylabel(lbl_map["sigma"], **lbl_style)
    ax.set_zlabel(lbl_map["eta"], **lbl_style)

    _apply_logscale(ax, logscale, axes_list=["x", "y", "z"])
    ax.view_init(elev=elev, azim=azim)
    _style_3d_panes(ax)
    ax.grid(True, **_GRID_PANEL_3D)
    # uniform 2-sig-fig sci format across all three 3D axes; matches the 2D panel formatter
    _apply_sci_format(ax, axes_list=["x", "y", "z"])
    for _axis_name in ("x", "y", "z"):
        ax.tick_params(axis=_axis_name, **tick_style)


def _estimate_single_mode_count(coeff_data: Dict[str, Any]) -> int:
    """*_estimate_single_mode_count()* number of distinct (c, mu) combos in single-mode yoly data.

    Used to size the legend strip when the caller didn't pass `paths` / `scenarios`. Looks up the first `c_*` and `\\mu_*` arrays via the same prefix-search the painters use, then returns `len(unique(c)) * len(unique(mu))`. Falls back to 1 when the data isn't shaped for single-mode (so the layout still has a sane minimum).

    Args:
        coeff_data (Dict[str, Any]): single-mode coefficient dict.

    Returns:
        int: estimated combo count for legend-row budgeting.
    """
    try:
        _c_full = next(_k for _k in coeff_data if _k.startswith("c_"))
        _mu_full = next(_k for _k in coeff_data if _k.startswith("\\mu"))
    except StopIteration:
        return 1
    _c_arr = np.asarray(coeff_data.get(_c_full, []), dtype=float)
    _mu_arr = np.asarray(coeff_data.get(_mu_full, []), dtype=float)
    if _c_arr.size == 0 or _mu_arr.size == 0:
        return 1
    return int(len(np.unique(_c_arr)) * len(np.unique(_mu_arr)))


def _lift_legend_to_footer(legend_axes: Optional[Any],
                           footer_ax: Optional[Any],
                           legend_title: str,
                           ncol_cap: int = 6) -> None:
    """*_lift_legend_to_footer()* copy legend handles + labels from a body axis into the footer axis.

    Args:
        legend_axes (Optional[Any]): the body axis whose handles should be lifted; None is a no-op.
        footer_ax (Optional[Any]): footer axis returned by `build_stacked_figure`; None is a no-op.
        legend_title (str): legend title (e.g. "Scenario", "Path").
        ncol_cap (int): maximum column count for the footer legend. Defaults to 6.
    """
    if legend_axes is None or footer_ax is None:
        return
    _handles, _labels = legend_axes.get_legend_handles_labels()
    if not _handles:
        return
    _ncol = min(len(_labels), ncol_cap)
    footer_ax.legend(_handles, _labels,
                     loc="center",
                     ncol=_ncol,
                     fontsize=12,
                     framealpha=0.9,
                     title=legend_title,
                     title_fontsize=13)


# ---------------------------------------------------------------------------
# Public plotters
# ---------------------------------------------------------------------------


def plot_yoly_chart(coeff_data: Dict[str, Any],
                    *,
                    labels: Optional[Dict[str, str]] = None,
                    paths: Optional[Dict[str, str]] = None,
                    scenarios: Optional[Dict[str, str]] = None,
                    logscale: Union[bool, List[bool]] = False,
                    layout: Optional[FigureLayout] = None,
                    title: Optional[str] = None,
                    file_path: Optional[str] = None,
                    fname: Optional[str] = None,
                    legend_ncol_cap: int = 4,
                    verbose: bool = False) -> Figure:
    """*plot_yoly_chart()* single-queue 2D yoly chart: 2x2 grid of coefficient planes.

    Panels (row-major): (theta, sigma), (theta, eta), (sigma, eta), (theta, phi). Three rendering modes selected by the caller's grouping kwarg:

        - **Single-queue** (default, both groupings None): looks up theta / sigma / eta / phi / c / mu / K arrays in `coeff_data` by semantic prefix. Each point is coloured by its `c` value and shaped by its `mu` value; K-endpoints get inline annotations.
        - **Multi-path** (`paths=`, multi-path idiom): one colour + marker per named path; per-path arrays keyed by `\\<coef>_{<path_tag>}`.
        - **Multi-scenario** (`scenarios=`, TAS idiom): same as multi-path under different naming. Mutually exclusive with `paths=`.

    Args:
        coeff_data (Dict[str, Any]): sweep dict keyed by LaTeX-subscripted symbols.
        labels (Optional[Dict[str, str]]): display labels per short coefficient name. Missing keys fall back to `_DEFAULT_LABELS`.
        paths (Optional[Dict[str, str]]): multi-path grouping `{display_name: path_tag}`.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping; aliases `paths=`.
        logscale (Union[bool, List[bool]]): per-axis log toggle, applied to every panel.
        layout (Optional[FigureLayout]): full layout override. Defaults to a 2x2 2D body with a centred legend footer.
        title (Optional[str]): figure title.
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Raises:
        ValueError: If both `paths=` and `scenarios=` are provided.

    Returns:
        Figure: the matplotlib figure.

    Example::

        plot_yoly_chart(coeff_data,
                        scenarios={"Before": "baseline_{TAS_{1}}",
                                   "After":  "aggregate_{TAS_{1}}"},
                        title="TAS_{1} adaptation trajectory",
                        file_path="data/img/dimensional/yoly",
                        fname="trajectory")
    """
    _groups, _legend_title, _lbl_map = _resolve_yoly_inputs(labels, paths, scenarios)

    # legend strip auto-sizes to the row count: `legend_ncol_cap`-col cap means N entries -> ceil(N/cap) rows; rough budget is 0.018 figure-fraction per row + 0.04 padding/title. Floor at 0.18 keeps the small-legend look unchanged; grows so the legend never overlaps the body even with verbose multi-scenario labels.
    _n_legend_entries = len(_groups) if _groups else _estimate_single_mode_count(coeff_data)
    _rows_needed = max(1, (_n_legend_entries + legend_ncol_cap - 1) // legend_ncol_cap)
    _footer_h = max(0.18, 0.04 + _rows_needed * 0.018)

    # narrower + taller. title_h and footer_h trimmed so each strip wraps its text content tightly: the title strip ends just below the suptitle, and the footer strip ends just below the legend. outer_hspace near zero closes the inter-region gap. Footer legend uses mode="expand" (in render_footer_legend) to clip to body width.
    _default_layout = FigureLayout(title=title,
                                   title_h=0.045,
                                   body=BodySpec(shape=(2, 2),
                                                 panel_kind="2d",
                                                 wspace=0.32,
                                                 hspace=0.22),
                                   footer_h=_footer_h,
                                   footer_kind="legend",
                                   figsize=(16, 22),
                                   outer_hspace=0.025)
    _layout = _pick_layout(layout, _default_layout)

    _fig, _regions = build_stacked_figure(_layout)
    _axes = _regions["body_axes"]

    _legend_axes: Optional[Any] = None
    for _idx, (_panel_title, _x_key, _y_key) in enumerate(_YOLY_PANELS):
        _ax = _axes[_idx]
        if _groups:
            _has_legend = _paint_groups_2d_yoly(_ax, coeff_data,
                                                _x_key, _y_key, _groups)
        else:
            _has_legend = _paint_single_2d_yoly(_ax, coeff_data,
                                                _x_key, _y_key)
        if _has_legend and _legend_axes is None:
            _legend_axes = _ax

        _apply_yoly_panel_axes(_ax,
                               _x_key, _y_key,
                               _lbl_map,
                               _LBL_STY_2D_SINGLE,
                               _panel_title,
                               logscale)

    # cap columns at `legend_ncol_cap` so verbose labels (e.g. multi-adp scenarios) wrap rather than overflow.
    _lift_legend_to_footer(_legend_axes,
                           _regions["footer_ax"],
                           _legend_title,
                           ncol_cap=legend_ncol_cap)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


def plot_yoly_space(coeff_data: Dict[str, Any],
                    *,
                    labels: Optional[Dict[str, str]] = None,
                    paths: Optional[Dict[str, str]] = None,
                    scenarios: Optional[Dict[str, str]] = None,
                    logscale: Union[bool, List[bool]] = False,
                    layout: Optional[FigureLayout] = None,
                    title: Optional[str] = None,
                    subtitle: Optional[str] = None,
                    file_path: Optional[str] = None,
                    fname: Optional[str] = None,
                    legend_ncol_cap: int = 6,
                    verbose: bool = False) -> Figure:
    """*plot_yoly_space()* single 3D yoly cloud (theta x sigma x eta) for one artifact / system.

    Three rendering modes share the 3D axes:

        - **Single-queue** (both groupings None): colour by `c`, marker by `mu`, K-endpoints annotated.
        - **Multi-path** (`paths=`, multi-path idiom): one colour + marker per named path.
        - **Multi-scenario** (`scenarios=`, TAS idiom): one colour + marker per named adaptation. Mutually exclusive with `paths=`.

    Args:
        coeff_data (Dict[str, Any]): sweep dict keyed by LaTeX-subscripted symbols.
        labels (Optional[Dict[str, str]]): display labels per short coefficient name (`"theta"`, `"sigma"`, `"eta"`).
        paths (Optional[Dict[str, str]]): multi-path grouping.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping; aliases `paths=`.
        logscale (Union[bool, List[bool]]): bool toggles log scale on all three axes; 3-list selects `[x_log, y_log, z_log]`.
        layout (Optional[FigureLayout]): full layout override. Defaults to a 1x1 3D body with a centred legend footer.
        title (Optional[str]): figure title.
        subtitle (Optional[str]): axes-level subtitle (drawn inside the body axis).
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Raises:
        ValueError: If both `paths=` and `scenarios=` are provided.

    Returns:
        Figure: the matplotlib figure.

    Example::

        plot_yoly_space(coeff_data,
                        title="System behaviour cloud",
                        subtitle="TAS_{1} (baseline)",
                        file_path="data/img/dimensional/space",
                        fname="cloud_baseline")
    """
    _groups, _legend_title, _lbl_map = _resolve_yoly_inputs(labels, paths, scenarios)

    # title strip carries either the suptitle alone or suptitle stacked above subtitle. We pass title=None to build_stacked_figure when both are set so it doesn't auto-draw at strip centre; both lines are then drawn manually into the same dedicated title_ax with explicit y-positions in axes coords (no figure-coord arithmetic, no overlap risk).
    _has_subtitle = bool(subtitle)
    _title_h = 0.10 if _has_subtitle else 0.05
    _default_layout = FigureLayout(title=None if _has_subtitle else title,
                                   title_h=_title_h,
                                   body=BodySpec(shape=(1, 1),
                                                 panel_kind="3d"),
                                   footer_h=0.10,
                                   footer_kind="legend",
                                   figsize=(17, 14),
                                   outer_hspace=0.02)
    _layout = _pick_layout(layout, _default_layout)

    _fig, _regions = build_stacked_figure(_layout)
    _ax = _regions["body_axes"][0]

    if _has_subtitle and title:
        # main title in the upper half of the title strip; subtitle in the lower half. Both in axes coords on the dedicated title_ax (transparent, off) so they never collide with the body or each other
        _title_ax = _regions["title_ax"]
        _title_ax.text(0.5, 0.72, title,
                       ha="center", va="center",
                       transform=_title_ax.transAxes,
                       **_SUPTITLE_STYLE)
        _title_ax.text(0.5, 0.22, subtitle,
                       ha="center", va="center",
                       transform=_title_ax.transAxes,
                       fontsize=18,
                       fontstyle="italic",
                       **_LBL_STYLE)

    if _groups:
        _has_legend = _paint_groups_3d_yoly(_ax, coeff_data, _groups)
    else:
        _has_legend = _paint_single_3d_yoly(_ax, coeff_data)

    _apply_yoly_3d_axes(_ax,
                        _lbl_map,
                        _LBL_STY_3D_SINGLE,
                        _TICK_STY_3D_SINGLE,
                        elev=30,
                        azim=110,
                        logscale=logscale)

    if _has_legend:
        _lift_legend_to_footer(_ax,
                               _regions["footer_ax"],
                               _legend_title,
                               ncol_cap=legend_ncol_cap)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


# Dim-grey dashed trajectory line connecting baseline -> s1 -> s2 -> aggregate; X
# marker style for each operating point. Marker colour matches the swept cloud's
# per-scenario colour (both come from `_generate_color_map` keyed by index) so the
# X reads as "the operating point of THIS cloud" at a glance. The annotation text
# uses a dark contrasting colour (deep forest green) so the adp tag stays
# legible against any cloud or marker colour.
_OP_POINT_LINE_COLOR = "#888888"
_OP_POINT_LINE_WIDTH = 1.0
_OP_POINT_LINE_STYLE = "--"
_OP_POINT_LINE_ZORDER = 2.0
_OP_POINT_MARKER = "X"
_OP_POINT_SIZE = 180
_OP_POINT_EDGE = "black"
_OP_POINT_EDGEWIDTH = 0.8
_OP_POINT_ZORDER = 3.0
_OP_POINT_LABEL_OFFSET = (6, 4)
_OP_POINT_LABEL_FONTSIZE = 9
_OP_POINT_LABEL_COLOR = "#1B5E20"
_OP_POINT_LABEL_BG = "white"
_OP_POINT_LABEL_BG_ALPHA = 0.6
_OP_POINT_LABEL_PAD = 1.5
_OP_POINT_LEGEND_SUFFIX = " (op)"


def _op_point_label_bbox() -> Dict[str, Any]:
    """*_op_point_label_bbox()* matplotlib bbox dict for the op-point annotation background.

    White semi-transparent (60 % opaque) so the adp tag stays legible when it lands on top of a dense sweep cloud or directly over a marker.

    Returns:
        Dict[str, Any]: kwargs for `ax.annotate(..., bbox=...)` / `ax.text(..., bbox=...)`.
    """
    return {
        "facecolor": _OP_POINT_LABEL_BG,
        "alpha": _OP_POINT_LABEL_BG_ALPHA,
        "edgecolor": "none",
        "pad": _OP_POINT_LABEL_PAD,
    }


def _resolve_op_point_colors(adps: List[str],
                             scenarios: Optional[Dict[str, str]],
                             paths: Optional[Dict[str, str]]) -> Dict[str, Any]:
    """*_resolve_op_point_colors()* mirror the sweep plotter's sort-based palette so op-point markers match their swept cloud's colour 1:1.

    The yoly sweep painter assigns colours by `sorted(scenarios.keys())`, not by insertion order, so a naive `_generate_color_map(op_points.keys())` puts the wrong colour under each marker. This helper rebuilds the same `display_name -> color` mapping the sweep used, then pairs op-point entries to the scenarios by insertion order (the natural way the notebook constructs both dicts from the same `for adp in adps` loop).

    When no `scenarios=` / `paths=` is provided (single-mode plotting), falls back to a plain `_generate_color_map(adps)` per insertion order.

    Args:
        adps (List[str]): op-point keys in caller-provided order (the trajectory direction).
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping `{display_name: tag}`.
        paths (Optional[Dict[str, str]]): multi-path grouping (alias of `scenarios`).

    Returns:
        Dict[str, Any]: `{adp: matplotlib_color}` aligned with the swept cloud's colours.
    """
    _group_map = scenarios or paths
    if _group_map and len(_group_map) == len(adps):
        _sorted_displays = sorted(_group_map.keys())
        _palette = _generate_color_map(_sorted_displays)
        _display_to_color = dict(zip(_sorted_displays, _palette))
        # Pair op-point keys to scenario keys by insertion order. The notebook
        # convention builds both dicts from the same `for a in adps` loop, so
        # `op_points["s1"]` lines up with `scenarios[DISPLAY["s1"]]`.
        return {_adp: _display_to_color[_display]
                for _adp, _display in zip(adps, _group_map.keys())}
    _palette = _generate_color_map(adps)
    return {_a: _palette[_i] for _i, _a in enumerate(adps)}


def _is_3d_axis(ax: Any) -> bool:
    """*_is_3d_axis()* True iff `ax` is a matplotlib 3D axes object."""
    return getattr(ax, "name", "") == "3d"


def _yoly_body_axes(fig: Figure, *, kind: str) -> List[Any]:
    """*_yoly_body_axes()* find the body panels of a yoly figure for overlay.

    `build_stacked_figure` prepends a title axis (`axis("off")`) and appends a footer axis (also `axis("off")`); the body panels live in between. For `kind="chart"` the four body axes are rectilinear 2x2 panels; for `kind="space"` the single body axis is a 3D projection.

    Args:
        fig (Figure): the figure returned by `plot_yoly_chart` or `plot_yoly_space`.
        kind (str): `"chart"` (expects 4 rectilinear body axes) or `"space"` (expects one 3D body axis).

    Returns:
        List[Any]: body axes only. 4 entries for `"chart"`; 1 entry for `"space"`.

    Raises:
        RuntimeError: when the expected body-axis count is not present (defends against base-plotter layout drift).
    """
    if kind == "space":
        _body = [_ax for _ax in fig.axes if _is_3d_axis(_ax)]
        _expected = 1
    else:
        _body = [_ax for _ax in fig.axes
                 if (not _is_3d_axis(_ax)) and getattr(_ax, "axison", False)]
        _expected = 4
    if len(_body) != _expected:
        _msg = (f"plot_yoly_with_op_points expected {_expected} body axes "
                f"for kind={kind!r}; found {len(_body)}")
        raise RuntimeError(_msg)
    return _body


def _overlay_op_points_chart(axes: List[Any],
                             op_points: Dict[str, Dict[str, float]],
                             colors: Dict[str, Any]) -> None:
    """*_overlay_op_points_chart()* scatter star markers + trajectory line + labels on the 4 yoly chart panels.

    Args:
        axes (List[Any]): the 4 body axes from `_yoly_body_axes(fig, kind="chart")`, in row-major order matching `_YOLY_PANELS`.
        op_points (Dict[str, Dict[str, float]]): `{adp: {theta, sigma, eta, phi}}` in adaptation order.
        colors (Dict[str, Any]): `{adp: matplotlib_color}` from `_generate_color_map`.
    """
    _adps = list(op_points.keys())
    for _ax, (_title, _x_key, _y_key) in zip(axes, _YOLY_PANELS):
        _xs = [op_points[_a][_x_key] for _a in _adps]
        _ys = [op_points[_a][_y_key] for _a in _adps]
        _ax.plot(_xs, _ys,
                 color=_OP_POINT_LINE_COLOR,
                 linewidth=_OP_POINT_LINE_WIDTH,
                 linestyle=_OP_POINT_LINE_STYLE,
                 zorder=_OP_POINT_LINE_ZORDER)
        for _adp, _x, _y in zip(_adps, _xs, _ys):
            _ax.scatter([_x], [_y],
                        marker=_OP_POINT_MARKER,
                        s=_OP_POINT_SIZE,
                        color=colors[_adp],
                        edgecolor=_OP_POINT_EDGE,
                        linewidth=_OP_POINT_EDGEWIDTH,
                        zorder=_OP_POINT_ZORDER)
            _ax.annotate(_adp.capitalize(),
                         xy=(_x, _y),
                         xytext=_OP_POINT_LABEL_OFFSET,
                         textcoords="offset points",
                         fontsize=_OP_POINT_LABEL_FONTSIZE,
                         color=_OP_POINT_LABEL_COLOR,
                         fontweight="bold",
                         bbox=_op_point_label_bbox())


def _overlay_op_points_space(ax: Any,
                             op_points: Dict[str, Dict[str, float]],
                             colors: Dict[str, Any]) -> None:
    """*_overlay_op_points_space()* scatter star markers + 3D trajectory line + labels on the yoly space (theta x sigma x eta).

    Args:
        ax (Any): the 3D body axis from `_yoly_body_axes(fig, kind="space")`.
        op_points (Dict[str, Dict[str, float]]): `{adp: {theta, sigma, eta, phi}}`.
        colors (Dict[str, Any]): `{adp: matplotlib_color}`.
    """
    _adps = list(op_points.keys())
    _xs = [op_points[_a]["theta"] for _a in _adps]
    _ys = [op_points[_a]["sigma"] for _a in _adps]
    _zs = [op_points[_a]["eta"] for _a in _adps]
    ax.plot(_xs, _ys, _zs,
            color=_OP_POINT_LINE_COLOR,
            linewidth=_OP_POINT_LINE_WIDTH,
            linestyle=_OP_POINT_LINE_STYLE,
            zorder=_OP_POINT_LINE_ZORDER)
    for _adp, _x, _y, _z in zip(_adps, _xs, _ys, _zs):
        ax.scatter([_x], [_y], [_z],
                   marker=_OP_POINT_MARKER,
                   s=_OP_POINT_SIZE,
                   color=colors[_adp],
                   edgecolor=_OP_POINT_EDGE,
                   linewidth=_OP_POINT_EDGEWIDTH,
                   zorder=_OP_POINT_ZORDER)
        # 3D axes: text() instead of annotate(); place tag slightly offset along x.
        ax.text(_x, _y, _z, f"  {_adp.capitalize()}",
                fontsize=_OP_POINT_LABEL_FONTSIZE,
                color=_OP_POINT_LABEL_COLOR,
                fontweight="bold",
                bbox=_op_point_label_bbox())


def _append_op_points_to_footer_legend(fig: Figure,
                                       adps: List[str],
                                       colors: Dict[str, Any],
                                       legend_ncol_cap: int) -> None:
    """*_append_op_points_to_footer_legend()* extend the footer legend with one op-point entry per adaptation.

    The base yoly plotters lift the body axis's swept-cloud legend onto a dedicated footer axis. After the overlay paints X markers on top, append a `Line2D` proxy per adaptation to that footer legend so the marker's meaning is explicit (matched colour-and-tag with the per-adaptation cloud entry).

    No-op when the figure has no footer legend (e.g. single-mode plotting without `scenarios=`).

    Args:
        fig (Figure): the matplotlib figure produced by the base plotter (already has its footer legend).
        adps (List[str]): adaptation labels in iteration order; matches the colour assignment in the overlay.
        colors (Dict[str, Any]): `{adp: matplotlib_color}` used for the X markers; reused here for the legend proxies so the colour story stays consistent.
        legend_ncol_cap (int): max column count; same value the base plotter used so the augmented legend wraps cleanly.
    """
    from matplotlib.lines import Line2D

    _footer_ax = None
    for _ax in reversed(fig.axes):
        if _ax.get_legend() is not None:
            _footer_ax = _ax
            break
    if _footer_ax is None:
        return

    _existing = _footer_ax.get_legend()
    _handles: List[Any] = list(_existing.legend_handles)
    _labels: List[str] = [_t.get_text() for _t in _existing.get_texts()]
    _title = _existing.get_title().get_text() or None

    for _adp in adps:
        _proxy = Line2D([0], [0],
                        marker=_OP_POINT_MARKER,
                        linestyle="None",
                        markerfacecolor=colors[_adp],
                        markeredgecolor=_OP_POINT_EDGE,
                        markeredgewidth=_OP_POINT_EDGEWIDTH,
                        markersize=11)
        _handles.append(_proxy)
        _labels.append(f"{_adp.capitalize()}{_OP_POINT_LEGEND_SUFFIX}")

    _existing.remove()
    _footer_ax.legend(_handles, _labels,
                      loc="center",
                      ncol=min(len(_labels), legend_ncol_cap),
                      fontsize=12,
                      framealpha=0.9,
                      title=_title,
                      title_fontsize=13)


def plot_yoly_with_op_points(coeff_data: Dict[str, Any],
                             op_points: Dict[str, Dict[str, float]],
                             *,
                             kind: Literal["chart", "space"] = "chart",
                             title: Optional[str] = None,
                             file_path: Optional[str] = None,
                             fname: Optional[str] = None,
                             verbose: bool = False,
                             **kwargs: Any) -> Figure:
    """*plot_yoly_with_op_points()* yoly sweep + dimensional operating-point overlay.

    Calls `plot_yoly_chart` (`kind="chart"`) or `plot_yoly_space` (`kind="space"`) with `coeff_data` (the swept design space), then scatters one star marker per adaptation at its `(theta, sigma, eta, phi)` operating point. A dim-grey trajectory line connects the adaptations in iteration order so the reader sees the adaptation path through coefficient space; each star is labelled with the adaptation tag.

    For `kind="chart"` the overlay lands on every panel of the 2x2 layout: `(theta, sigma)`, `(theta, eta)`, `(sigma, eta)`, `(theta, phi)`. For `kind="space"` the overlay lands as N 3D scatter points on the single `(theta, sigma, eta)` axis.

    Args:
        coeff_data (Dict[str, Any]): same sweep dict the base plotter consumes.
        op_points (Dict[str, Dict[str, float]]): `{adp_label: {"theta", "sigma", "eta", "phi"}}` from `src.dimensional.load_dim_op_points`. Insertion order drives both the legend order and the trajectory direction.
        kind (Literal["chart", "space"]): which base plotter to wrap. Defaults to `"chart"`.
        title (Optional[str]): figure title; forwarded to the base plotter.
        file_path (Optional[str]): directory to save into; forwarded to the base plotter.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.
        **kwargs: any additional kwarg the base plotter accepts (labels, paths, scenarios, logscale, layout). `paths` / `scenarios` apply to the sweep cloud only; the op-point overlay is independent.

    Raises:
        ValueError: when `kind` is not `"chart"` or `"space"`.
        RuntimeError: when the wrapped figure does not expose the expected body-axis count.

    Returns:
        Figure: the matplotlib figure produced by the base plotter, with the overlay added.

    Example::

        op = load_dim_op_points(["baseline", "s1", "s2", "aggregate"])
        plot_yoly_with_op_points(base_arch, op, kind="chart",
                                 file_path="data/img/dimensional/yoly/cmp",
                                 fname="yc_with_op_points")
    """
    if kind not in ("chart", "space"):
        _msg = f"kind must be 'chart' or 'space'; got {kind!r}"
        raise ValueError(_msg)

    # Defer the save until after the overlay so the persisted PNG / SVG capture the markers.
    if kind == "chart":
        _fig = plot_yoly_chart(coeff_data,
                               title=title,
                               file_path=None,
                               fname=None,
                               verbose=False,
                               **kwargs)
    else:
        _fig = plot_yoly_space(coeff_data,
                               title=title,
                               file_path=None,
                               fname=None,
                               verbose=False,
                               **kwargs)

    _adps = list(op_points.keys())
    _colors = _resolve_op_point_colors(adps=_adps,
                                       scenarios=kwargs.get("scenarios"),
                                       paths=kwargs.get("paths"))

    _body = _yoly_body_axes(_fig, kind=kind)
    if kind == "chart":
        _overlay_op_points_chart(_body, op_points, _colors)
    else:
        _overlay_op_points_space(_body[0], op_points, _colors)

    # Augment the existing footer legend with one X-marker proxy per adaptation
    # so the marker's meaning is explicit and the colour-cloud / colour-marker
    # mapping reads from the same legend.
    _legend_ncol_cap = int(kwargs.get("legend_ncol_cap", 4 if kind == "chart" else 6))
    _append_op_points_to_footer_legend(_fig, _adps, _colors, _legend_ncol_cap)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


def plot_selection_surface(cloud: Dict[str, np.ndarray],
                           bounds: Dict[str, float],
                           *,
                           tag: str = "req",
                           winner: Optional[Dict[str, float]] = None,
                           feasible_mask: Optional[np.ndarray] = None,
                           title: Optional[str] = None,
                           subtitle: Optional[str] = None,
                           file_path: Optional[str] = None,
                           fname: Optional[str] = None,
                           verbose: bool = False) -> Figure:
    """*plot_selection_surface()* the DASA selection surface: the requirement-frame candidate cloud, the viable box, and the winner deepest inside it.

    A single 2D `(sigma_arch, eta_arch)` panel in the requirement coordinate where the DASA viable box is defined (`sigma_R2` is the R2 axis bound, `eta_R1` the R1 axis bound). Each cloud point is one search candidate; the shaded rectangle from the origin to `(sigma_R2, eta_R1)` is the viable region the search descends into, and the star is the winner. This is the sigma-eta panel of the yoly chart re-axed into the frame where the bound box is numerically valid, so it is kept separate from the sum-frame `plot_yoly_chart` family.

    Args:
        cloud (Dict[str, np.ndarray]): requirement-frame cloud from `src.dimensional.search.cloud_to_selection_surface`; must carry `\\sigma_{<tag>}` and `\\eta_{<tag>}` arrays.
        bounds (Dict[str, float]): output of `src.dimensional.search.compute_bounds`; must carry `sigma_R2` and `eta_R1`.
        tag (str): architecture subscript on the cloud keys. Defaults to `"req"`.
        winner (Optional[Dict[str, float]]): winning candidate carrying `sigma_arch` and `eta_arch`; drawn as a star when set.
        feasible_mask (Optional[np.ndarray]): boolean mask aligned to the cloud arrays; True points (inside the viable box) are drawn solid, False points are de-emphasised. When None, every point uses the single in-box colour.
        title (Optional[str]): figure title.
        subtitle (Optional[str]): second title line drawn below `title` (italic); ignored when `title` is None.
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Raises:
        KeyError: when the cloud lacks the `\\sigma_{<tag>}` / `\\eta_{<tag>}` keys or `bounds` lacks `sigma_R2` / `eta_R1`.

    Returns:
        Figure: the matplotlib figure.

    Example::

        surface = cloud_to_selection_surface(cloud_df)
        plot_selection_surface(surface, bounds,
                               winner={"sigma_arch": 0.40, "eta_arch": 34.5},
                               file_path="data/img/search", fname="selection_surface")
    """
    _sigma_key = f"\\sigma_{{{tag}}}"
    _eta_key = f"\\eta_{{{tag}}}"
    for _k in (_sigma_key, _eta_key):
        if _k not in cloud:
            _msg = f"cloud missing {_k!r}; got {list(cloud.keys())}"
            raise KeyError(_msg)
    for _k in ("sigma_R2", "eta_R1"):
        if _k not in bounds:
            _msg = f"bounds missing {_k!r}; got {list(bounds.keys())}"
            raise KeyError(_msg)

    _x = np.asarray(cloud[_sigma_key], dtype=float)
    _y = np.asarray(cloud[_eta_key], dtype=float)
    _sigma_r2 = float(bounds["sigma_R2"])
    _eta_r1 = float(bounds["eta_R1"])

    # subtitle case: pass title=None so build_stacked_figure does not auto-draw the suptitle at
    # strip centre; both lines are then drawn manually into the dedicated title_ax (axes coords)
    _has_subtitle = bool(subtitle)
    _title_h = 0.12 if _has_subtitle else 0.08
    _layout = FigureLayout(title=None if _has_subtitle else title,
                           title_h=_title_h,
                           body=BodySpec(shape=(1, 1), panel_kind="2d"),
                           footer_h=0.16,
                           footer_kind="legend",
                           figsize=(9, 7.5),
                           outer_hspace=0.32)
    _fig, _regions = build_stacked_figure(_layout)
    _ax = _regions["body_axes"][0]

    if _has_subtitle and title:
        # main title in the upper half of the strip, subtitle in the lower half
        _title_ax = _regions["title_ax"]
        _title_ax.text(0.5, 0.70, title,
                       ha="center", va="center",
                       transform=_title_ax.transAxes,
                       **_SUPTITLE_STYLE)
        _title_ax.text(0.5, 0.22, subtitle,
                       ha="center", va="center",
                       transform=_title_ax.transAxes,
                       fontsize=13,
                       fontstyle="italic",
                       **_LBL_STYLE)

    # viable region: shaded rectangle from the origin to the (sigma_R2, eta_R1) corner, drawn
    # behind the scatter so the candidate cloud stays readable on top of it
    _ax.add_patch(mpatches.Rectangle((0.0, 0.0), _sigma_r2, _eta_r1,
                                     facecolor=_BAR_BLUE, alpha=0.10,
                                     edgecolor="none", zorder=0))

    # candidate scatter: split solid (inside the box) vs de-emphasised (outside) when a mask is given
    _footer_handles: List[Any] = []
    _footer_labels: List[str] = []
    if feasible_mask is not None:
        _mask = np.asarray(feasible_mask, dtype=bool)
        _ax.scatter(_x[~_mask], _y[~_mask], s=40, alpha=0.30,
                    color=_SELECT_OUT_GREY, edgecolor="none", zorder=2)
        _ax.scatter(_x[_mask], _y[_mask], s=55, alpha=0.80,
                    color=_BAR_BLUE, edgecolor=_TEXT_BLACK, linewidth=0.4, zorder=3)
        _footer_handles = [
            mpatches.Patch(facecolor=_BAR_BLUE, edgecolor=_TEXT_BLACK, label="Inside viable box"),
            mpatches.Patch(facecolor=_SELECT_OUT_GREY, edgecolor="none", label="Outside viable box"),
        ]
        _footer_labels = ["Inside viable box", "Outside viable box"]
    else:
        _ax.scatter(_x, _y, s=55, alpha=0.80,
                    color=_BAR_BLUE, edgecolor=_TEXT_BLACK, linewidth=0.4, zorder=3)
        _footer_handles = [
            mpatches.Patch(facecolor=_BAR_BLUE, edgecolor=_TEXT_BLACK, label="Candidate"),
        ]
        _footer_labels = ["Candidate"]

    # bound lines + viable-region proxy live in the in-axis legend
    _ref_handles: List[Any] = [
        mpatches.Patch(facecolor=_BAR_BLUE, alpha=0.10, edgecolor="none",
                       label="DASA viable region"),
        Line2D([0], [0], color=_SELECT_BOUND_RED, ls="--",
               label=r"$\mathbf{\sigma_{R2}}$ bound (R2)"),
        Line2D([0], [0], color=_SELECT_BOUND_RED, ls="--",
               label=r"$\mathbf{\eta_{R1}}$ bound (R1)"),
    ]
    _ax.axvline(_sigma_r2, ls="--", color=_SELECT_BOUND_RED, lw=1.4, zorder=1)
    _ax.axhline(_eta_r1, ls="--", color=_SELECT_BOUND_RED, lw=1.4, zorder=1)

    if winner is not None:
        _wx = float(winner["sigma_arch"])
        _wy = float(winner["eta_arch"])
        _ax.scatter([_wx], [_wy], s=320, marker="*",
                    color=_BAR_ORANGE, edgecolor=_TEXT_BLACK, linewidth=0.8, zorder=5)
        _ref_handles.append(Line2D([0], [0], marker="*", linestyle="none",
                                   markerfacecolor=_BAR_ORANGE, markeredgecolor=_TEXT_BLACK,
                                   markersize=15, label="Winner"))

    _ax.set_xlabel(r"Stall ($\mathbf{\sigma_{arch}}$, $\propto$ R2)",
                   fontweight="bold", color=_TEXT_BLACK, fontsize=12)
    _ax.set_ylabel(r"Effective-yield ($\mathbf{\eta_{arch}}$, $\propto$ R1)",
                   fontweight="bold", color=_TEXT_BLACK, fontsize=12)
    _ax.grid(alpha=0.3, linestyle=":")
    _ax.set_axisbelow(True)
    _ax.legend(handles=_ref_handles, loc="best", framealpha=0.9, fontsize=9)

    render_footer_legend(_regions["footer_ax"],
                         _footer_handles,
                         _footer_labels,
                         ncol=max(1, len(_footer_labels)))

    _save_figure(_fig, file_path, fname or "selection_surface", verbose=verbose)
    return _fig


def plot_yoly_arts_hist(coeff_data: Dict[str, Dict[str, Any]],
                        *,
                        labels: Optional[Dict[str, str]] = None,
                        names: Optional[Dict[str, str]] = None,
                        layout: Optional[FigureLayout] = None,
                        title: Optional[str] = None,
                        file_path: Optional[str] = None,
                        fname: Optional[str] = None,
                        verbose: bool = False) -> Figure:
    """*plot_yoly_arts_hist()* per-node coefficient distributions as histograms arranged in a 3 x ceil(N/3) meta-grid; each node cell carries a 2x2 inner subgrid (one histogram per derived coefficient).

    Each histogram uses 50 bins, draws a mean reference line, and annotates the title with mean + std. The outer grid centres a short last row.

    Args:
        coeff_data (Dict[str, Dict[str, Any]]): nested `{node_key: {full_symbol: array}}`.
        labels (Optional[Dict[str, str]]): display labels per short coefficient name. Missing keys fall back to `_DEFAULT_LABELS`.
        names (Optional[Dict[str, str]]): human display-name override per node key (e.g. `{"TAS_{1}": "Dispatch"}`).
        layout (Optional[FigureLayout]): full layout override. Defaults to a 1x1 2D body with no footer (the meta-grid lives inside the body axis via subgridspec).
        title (Optional[str]): figure title.
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Returns:
        Figure: the matplotlib figure.

    Example::

        plot_yoly_arts_hist(coeff_data,
                            title="Per-node coefficient distributions",
                            file_path="data/img/dimensional/hist",
                            fname="dist_baseline")
    """
    # histogram x-axis stays symbol-only (no parenthesised name) to keep the dense per-comp grid readable; the operational name lives in the legend label / subplot title instead
    _hist_symbols = {
        "theta": r"$\mathbf{\theta}$",
        "sigma": r"$\mathbf{\sigma}$",
        "eta": r"$\mathbf{\eta}$",
        "phi": r"$\mathbf{\phi}$",
    }
    _lbl_map = {**_hist_symbols, **(labels or {})}
    _name_map = _resolve_name_map(names)

    _node_keys = list(coeff_data.keys())
    _n_nodes = len(_node_keys)

    _default_layout = FigureLayout(title=title,
                                   title_h=0.045,
                                   body=BodySpec(shape=(1, 1),
                                                 panel_kind="2d"),
                                   footer_h=0.0,
                                   footer_kind="none",
                                   figsize=(26, 26),
                                   outer_hspace=0.025)
    _layout = _pick_layout(layout, _default_layout)

    _fig, _regions = build_stacked_figure(_layout)
    _body_ax = _regions["body_axes"][0]

    if _n_nodes == 0:
        return _handle_empty_meta_grid(_fig, _body_ax,
                                       file_path, fname, verbose)

    # the body axis acts as a host for the subgridspec; hide its frame, keep its bbox
    _body_ax.axis("off")

    _n_rows, _n_cols, _last_row_idx, _n_last_row = _compute_grid_dims(_n_nodes)
    _gs_main = _body_ax.get_subplotspec().subgridspec(_n_rows, _n_cols,
                                                      hspace=0.30,
                                                      wspace=0.25)

    for _nd_idx, _node in enumerate(_node_keys):
        _nd_row, _nd_col = _compute_node_pos(_nd_idx,
                                             _n_rows, _n_cols,
                                             _last_row_idx, _n_last_row)

        _node_block = coeff_data[_node]
        _coef_map = _build_coef_map(_node_block)
        _n_coeffs = len(_coef_map)
        if _n_coeffs == 0:
            continue

        _col_lt = _generate_color_map(list(range(_n_coeffs)))
        _n_inner_cols = (_n_coeffs + 1) // 2
        # extra hspace on the inner grid so the two-line subplot title (mean + sigma) clears the histogram body above it
        _gs_node = _gs_main[_nd_row, _nd_col].subgridspec(2, _n_inner_cols,
                                                          hspace=0.85,
                                                          wspace=0.40)

        _anchor_cell_header(_fig,
                            _gs_main[_nd_row, _nd_col],
                            _format_node_header(_node, _name_map),
                            fontsize=15,
                            dy=0.008)

        _short_names = list(_coef_map.keys())
        for _c_idx, _short in enumerate(_short_names):
            _row = _c_idx // _n_inner_cols
            _col = _c_idx % _n_inner_cols
            _ax = _fig.add_subplot(_gs_node[_row, _col])
            _ax.set_facecolor("white")

            _full = _coef_map[_short]
            _data = np.asarray(_node_block[_full], dtype=float)
            _color = _col_lt[_c_idx]

            _ax.hist(_data,
                     bins=50,
                     color=_color,
                     alpha=0.7,
                     edgecolor=_TEXT_BLACK)

            # vertical reference at the sample median (X-tilde, more robust than mean to K-block tail clustering); the subplot title carries the sample mean (X-bar) and sample variance (s^2) for distribution shape
            _median = float(np.median(_data))
            _mean = float(np.mean(_data))
            _var = float(np.var(_data))
            _ax.axvline(_median,
                        color=_color,
                        linestyle="-",
                        linewidth=2,
                        label=rf"$\widetilde{{X}}={_fmt_sci_mathtext(_median)}$")

            _ax.set_xlabel(_lbl_map.get(_short, _short),
                           fontsize=11,
                           fontweight="bold",
                           color=_TEXT_BLACK)
            _ax.set_ylabel("Frequency",
                           fontsize=11,
                           fontweight="bold",
                           color=_TEXT_BLACK)
            # two-line title: mean on top, variance below; pad bumped to clear the body
            _ax.set_title(
                rf"$\overline{{X}}={_fmt_sci_mathtext(_mean)}$" "\n"
                rf"$s^{{2}}={_fmt_sci_mathtext(_var)}$",
                fontsize=10,
                fontweight="bold",
                color=_TEXT_BLACK,
                pad=8)

            _ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            _ax.tick_params(**_TICK_STYLE)
            _ax.legend(loc="best", fontsize=11, framealpha=0.9)
            _ax.grid(True,
                     alpha=0.8,
                     color=_TEXT_BLACK,
                     linewidth=1.0)
            for _spine in _ax.spines.values():
                _spine.set_edgecolor(_TEXT_BLACK)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


def plot_yoly_arts_behaviour(coeff_data: Dict[str, Dict[str, Any]],
                             *,
                             labels: Optional[Dict[str, str]] = None,
                             names: Optional[Dict[str, str]] = None,
                             paths: Optional[Dict[str, str]] = None,
                             scenarios: Optional[Dict[str, str]] = None,
                             logscale: Union[bool, List[bool]] = False,
                             layout: Optional[FigureLayout] = None,
                             title: Optional[str] = None,
                             file_path: Optional[str] = None,
                             fname: Optional[str] = None,
                             verbose: bool = False) -> Figure:
    """*plot_yoly_arts_behaviour()* per-node 3D yoly clouds laid out in a 3 x ceil(N/3) outer grid.

    Each cell carries a 3D scatter of theta x sigma x eta for one artifact. Three rendering modes (single / paths / scenarios) flow through to every cell with the same vocabulary.

    Args:
        coeff_data (Dict[str, Dict[str, Any]]): nested `{node_key: {full_symbol: array}}`.
        labels (Optional[Dict[str, str]]): display labels per short coefficient name.
        names (Optional[Dict[str, str]]): node display-name override.
        paths (Optional[Dict[str, str]]): multi-path grouping.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping; aliases `paths=`.
        logscale (Union[bool, List[bool]]): per-axis log toggle, applied to every cell.
        layout (Optional[FigureLayout]): full layout override.
        title (Optional[str]): figure title.
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Raises:
        ValueError: If both `paths=` and `scenarios=` are provided.

    Returns:
        Figure: the matplotlib figure.

    Example::

        plot_yoly_arts_behaviour(coeff_data,
                                 scenarios={"Before": "baseline",
                                            "After":  "aggregate"},
                                 title="Per-node 3D yoly trajectories",
                                 file_path="data/img/dimensional/beh",
                                 fname="beh_overlay")
    """
    _groups, _legend_title, _lbl_map = _resolve_yoly_inputs(labels, paths, scenarios)
    _name_map = _resolve_name_map(names)

    _node_keys = list(coeff_data.keys())
    _n_nodes = len(_node_keys)

    _default_layout = FigureLayout(title=title,
                                   title_h=0.045,
                                   body=BodySpec(shape=(1, 1),
                                                 panel_kind="2d"),
                                   footer_h=0.06,
                                   footer_kind="legend",
                                   figsize=(34, 29),
                                   outer_hspace=0.025)
    _layout = _pick_layout(layout, _default_layout)

    _fig, _regions = build_stacked_figure(_layout)
    _body_ax = _regions["body_axes"][0]

    if _n_nodes == 0:
        return _handle_empty_meta_grid(_fig, _body_ax,
                                       file_path, fname, verbose)

    _body_ax.axis("off")
    _n_rows, _n_cols, _last_row_idx, _n_last_row = _compute_grid_dims(_n_nodes)
    _gs_main = _body_ax.get_subplotspec().subgridspec(_n_rows, _n_cols,
                                                      hspace=0.10,
                                                      wspace=0.08)

    _legend_axes: Optional[Any] = None
    for _nd_idx, _node in enumerate(_node_keys):
        _nd_row, _nd_col = _compute_node_pos(_nd_idx,
                                             _n_rows, _n_cols,
                                             _last_row_idx, _n_last_row)

        _ax = _fig.add_subplot(_gs_main[_nd_row, _nd_col], projection="3d")
        _ax.set_facecolor("white")

        _node_block = coeff_data[_node]

        if _groups:
            _has_legend = _paint_groups_3d_yoly(_ax, _node_block, _groups)
        else:
            try:
                _has_legend = _paint_single_3d_yoly(_ax, _node_block)
            except KeyError:
                _has_legend = False
        if _has_legend and _legend_axes is None:
            _legend_axes = _ax

        _apply_yoly_3d_axes(_ax,
                            _lbl_map,
                            _LBL_STY_3D_GRID,
                            _TICK_STY_3D_GRID,
                            elev=25,
                            azim=105,
                            logscale=logscale)

        _ax.set_title(_format_node_header(_node, _name_map),
                      fontsize=19,
                      pad=10,
                      **_LBL_STYLE)

    _lift_legend_to_footer(_legend_axes,
                           _regions["footer_ax"],
                           _legend_title,
                           ncol_cap=6)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


def plot_yoly_arts_with_op_points(coeff_data: Dict[str, Dict[str, Any]],
                                  op_points_per_node: Dict[str, Dict[str, Dict[str, float]]],
                                  *,
                                  title: Optional[str] = None,
                                  file_path: Optional[str] = None,
                                  fname: Optional[str] = None,
                                  verbose: bool = False,
                                  **kwargs: Any) -> Figure:
    """*plot_yoly_arts_with_op_points()* per-node 3D yoly clouds + per-node operating-point overlay.

    Wraps `plot_yoly_arts_behaviour` (the 3 x ceil(N/3) per-node 3D grid) and drops one star marker per adaptation at every node cell's coordinates. Each cell sees `(theta, sigma, eta)` for that artifact alone, so the overlay anchors `baseline`, `s1`, `s2`, `aggregate` to the same node's swept cloud. A dim-grey trajectory line connects the per-adaptation points within each cell.

    Args:
        coeff_data (Dict[str, Dict[str, Any]]): nested `{node_key: {full_symbol: array}}`, same shape `plot_yoly_arts_behaviour` consumes.
        op_points_per_node (Dict[str, Dict[str, Dict[str, float]]]): `{adp: {node_key: {"theta", "sigma", "eta", "phi"}}}` from `src.dimensional.load_dim_op_points_per_node`. Adaptation order drives the trajectory direction.
        title (Optional[str]): figure title; forwarded.
        file_path (Optional[str]): directory to save into; forwarded.
        fname (Optional[str]): filename stem.
        verbose (bool): forwarded to `_save_figure`.
        **kwargs: any additional kwarg the base plotter accepts (`labels`, `names`, `paths`, `scenarios`, `logscale`, `layout`).

    Raises:
        RuntimeError: when the wrapped figure does not expose one 3D body axis per node.

    Returns:
        Figure: the matplotlib figure produced by the base plotter, with per-node overlays added.

    Example::

        op_per_node = load_dim_op_points_per_node(
            ["baseline", "s1", "s2", "aggregate"])
        plot_yoly_arts_with_op_points(
            base_sweep, op_per_node,
            title="Per-node 3D yoly + operating points",
            file_path="data/img/dimensional/yoly/cmp",
            fname="yab_per_node_with_op_points")
    """
    _fig = plot_yoly_arts_behaviour(coeff_data,
                                    title=title,
                                    file_path=None,
                                    fname=None,
                                    verbose=False,
                                    **kwargs)

    _adps = list(op_points_per_node.keys())
    _palette = _generate_color_map(_adps)
    _colors: Dict[str, Any] = {_a: _palette[_i] for _i, _a in enumerate(_adps)}

    _cells = [_ax for _ax in _fig.axes if _is_3d_axis(_ax)]
    _node_keys = list(coeff_data.keys())
    if len(_cells) != len(_node_keys):
        _msg = (f"plot_yoly_arts_with_op_points expected {len(_node_keys)} "
                f"3D body cells; found {len(_cells)}")
        raise RuntimeError(_msg)

    for _ax, _node in zip(_cells, _node_keys):
        _xs: List[float] = []
        _ys: List[float] = []
        _zs: List[float] = []
        for _adp in _adps:
            _node_op = op_points_per_node[_adp].get(_node)
            if _node_op is None:
                # Adaptation does not deploy this node (swap-slot mismatch); skip.
                continue
            _xs.append(_node_op["theta"])
            _ys.append(_node_op["sigma"])
            _zs.append(_node_op["eta"])
        if not _xs:
            continue
        _ax.plot(_xs, _ys, _zs,
                 color=_OP_POINT_LINE_COLOR,
                 linewidth=_OP_POINT_LINE_WIDTH,
                 linestyle=_OP_POINT_LINE_STYLE,
                 zorder=_OP_POINT_LINE_ZORDER)
        _i = 0
        for _adp in _adps:
            _node_op = op_points_per_node[_adp].get(_node)
            if _node_op is None:
                continue
            _ax.scatter([_xs[_i]], [_ys[_i]], [_zs[_i]],
                        marker=_OP_POINT_MARKER,
                        s=_OP_POINT_SIZE,
                        color=_colors[_adp],
                        edgecolor=_OP_POINT_EDGE,
                        linewidth=_OP_POINT_EDGEWIDTH,
                        zorder=_OP_POINT_ZORDER)
            _ax.text(_xs[_i], _ys[_i], _zs[_i], f"  {_adp.capitalize()}",
                     fontsize=_OP_POINT_LABEL_FONTSIZE,
                     color=_OP_POINT_LABEL_COLOR,
                     fontweight="bold",
                     bbox=_op_point_label_bbox())
            _i += 1

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig


def plot_yoly_arts_charts(coeff_data: Dict[str, Dict[str, Any]],
                          *,
                          labels: Optional[Dict[str, str]] = None,
                          names: Optional[Dict[str, str]] = None,
                          paths: Optional[Dict[str, str]] = None,
                          scenarios: Optional[Dict[str, str]] = None,
                          logscale: Union[bool, List[bool]] = False,
                          layout: Optional[FigureLayout] = None,
                          title: Optional[str] = None,
                          file_path: Optional[str] = None,
                          fname: Optional[str] = None,
                          verbose: bool = False) -> Figure:
    """*plot_yoly_arts_charts()* per-node 2D yoly planes laid out in a 3 x ceil(N/3) outer grid; each cell carries a 2x2 inner subgrid of coefficient planes.

    Inner panels per node match `plot_yoly_chart`: (theta, sigma), (theta, eta), (sigma, eta), (theta, phi). Three rendering modes (single / paths / scenarios) flow through every panel of every node.

    Args:
        coeff_data (Dict[str, Dict[str, Any]]): nested `{node_key: {full_symbol: array}}`.
        labels (Optional[Dict[str, str]]): display labels per short coefficient name.
        names (Optional[Dict[str, str]]): node display-name override.
        paths (Optional[Dict[str, str]]): multi-path grouping.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping; aliases `paths=`.
        logscale (Union[bool, List[bool]]): per-axis log toggle, applied to every panel of every cell.
        layout (Optional[FigureLayout]): full layout override.
        title (Optional[str]): figure title.
        file_path (Optional[str]): directory to save into.
        fname (Optional[str]): filename stem; both PNG and SVG written.
        verbose (bool): if True, prints one save-path message per format.

    Raises:
        ValueError: If both `paths=` and `scenarios=` are provided.

    Returns:
        Figure: the matplotlib figure.

    Example::

        plot_yoly_arts_charts(coeff_data,
                              scenarios={"Before": "baseline",
                                         "After":  "aggregate"},
                              title="Per-node 2D yoly planes",
                              file_path="data/img/dimensional/charts",
                              fname="planes_overlay")
    """
    _groups, _legend_title, _lbl_map = _resolve_yoly_inputs(labels, paths, scenarios)
    _name_map = _resolve_name_map(names)

    _node_keys = list(coeff_data.keys())
    _n_nodes = len(_node_keys)

    _default_layout = FigureLayout(title=title,
                                   title_h=0.045,
                                   body=BodySpec(shape=(1, 1),
                                                 panel_kind="2d"),
                                   footer_h=0.06,
                                   footer_kind="legend",
                                   figsize=(34, 29),
                                   outer_hspace=0.025)
    _layout = _pick_layout(layout, _default_layout)

    _fig, _regions = build_stacked_figure(_layout)
    _body_ax = _regions["body_axes"][0]

    if _n_nodes == 0:
        return _handle_empty_meta_grid(_fig, _body_ax,
                                       file_path, fname, verbose)

    _body_ax.axis("off")
    _n_rows, _n_cols, _last_row_idx, _n_last_row = _compute_grid_dims(_n_nodes)
    _gs_main = _body_ax.get_subplotspec().subgridspec(_n_rows, _n_cols,
                                                      hspace=0.25,
                                                      wspace=0.22)

    _legend_axes: Optional[Any] = None
    for _nd_idx, _node in enumerate(_node_keys):
        _nd_row, _nd_col = _compute_node_pos(_nd_idx,
                                             _n_rows, _n_cols,
                                             _last_row_idx, _n_last_row)

        _gs_node = _gs_main[_nd_row, _nd_col].subgridspec(2, 2,
                                                          hspace=0.45,
                                                          wspace=0.45)
        _node_block = coeff_data[_node]

        for _p_idx, (_panel_title, _x_key, _y_key) in enumerate(_YOLY_PANELS):
            _row = _p_idx // 2
            _col = _p_idx % 2
            _ax = _fig.add_subplot(_gs_node[_row, _col])
            _ax.set_facecolor("white")

            if _groups:
                _has_legend = _paint_groups_2d_yoly(_ax, _node_block,
                                                    _x_key, _y_key, _groups)
            else:
                try:
                    _has_legend = _paint_single_2d_yoly(_ax, _node_block,
                                                        _x_key, _y_key)
                except KeyError:
                    _has_legend = False

            if _has_legend and _legend_axes is None:
                _legend_axes = _ax

            _apply_yoly_panel_axes(_ax,
                                   _x_key, _y_key,
                                   _lbl_map,
                                   _LBL_STY_2D_GRID,
                                   _panel_title,
                                   logscale)

        _anchor_cell_header(_fig,
                            _gs_main[_nd_row, _nd_col],
                            _format_node_header(_node, _name_map),
                            fontsize=17,
                            dy=0.012)

    _lift_legend_to_footer(_legend_axes,
                           _regions["footer_ax"],
                           _legend_title,
                           ncol_cap=6)

    _save_figure(_fig, file_path, fname, verbose=verbose)
    return _fig
