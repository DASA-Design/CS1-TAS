# -*- coding: utf-8 -*-
"""
Module view/common.py
=====================

Single source for cosmetics, math, layout primitives, and family-private helpers used by every plotter under `src/view/`. The four plotter modules (`characterization.py`, `charter.py`, `diagrams.py`) become thin orchestrators that import from this module and call `build_stacked_figure` plus the matching family helpers.

Section map:

    - shared constants and `_apply_view_rcparams` (idempotent rcParams seed)
    - DataFrame + numeric helpers (`_resolve_metrics`, `_resolve_labels`, `_format_value`, `_generate_color_map`)
    - I/O (`_save_figure`)
    - design-contract primitives (`AxisSpec`, `BodySpec`, `FigureLayout`, `build_stacked_figure`, `attach_axis_spec`, `render_footer_*`)
    - calibration helpers (`_sort_n_con_usr_items`, `_draw_stat_bars`, `_draw_handler_scaling_axis`)
    - yoly helpers (`_sci_tick_fmt`, `_eng_tick_fmt`, `_style_3d_panes`, `_apply_sci_format`, `_apply_logscale`, `_format_path_legend`, `_generate_marker_map`, `_find_key_starting_with`, `_paint_single_2d_yoly`, `_paint_groups_2d_yoly`, `_paint_single_3d_yoly`, `_paint_groups_3d_yoly`, `_resolve_groups`, `_pick_coef_short_name`, `_build_coef_map`, `_compute_grid_dims`, `_compute_node_pos`, `_format_node_header`)
    - topology helpers (`_bfs_layout_shared`, `_build_topology_graph`, `_draw_qn_topology_axis`, `_draw_dim_topology_axis`, `_add_param_glossary`, `_add_qn_network_summary`, `_add_dim_network_summary`, `_add_qn_node_table`, `_add_dim_node_table`, plus the QN / dim glossary defaults and coefficient symbol maps)

Public symbols (consumed by plotter modules and re-exported by `src/view/__init__.py`): the design-contract primitives + the two glossary defaults. Everything else is leading-underscore private; plotter modules import the helpers they need by name.

Typical usage::

    from src.view.common import (
        AxisSpec, BodySpec, FigureLayout,
        build_stacked_figure, attach_axis_spec,
        render_footer_legend, _save_figure,
    )

    _layout = FigureLayout(
        title="Empty-handler scaling",
        body=BodySpec(shape=(1, 1), panel_kind="2d"),
        footer_h=0.10,
        footer_kind="legend",
        figsize=(10, 7),
    )
    _fig, _regions = build_stacked_figure(_layout)
    _ax = _regions["body_axes"][0]

    _ax.plot(xs, ys, marker="o")
    attach_axis_spec(_ax,
                     AxisSpec(scale="log", label="n_con_usr"),
                     AxisSpec(scale="log", label="Latency [us]"))

    _h, _l = _ax.get_legend_handles_labels()
    render_footer_legend(_regions["footer_ax"], _h, _l, ncol=3)

    _save_figure(_fig, "data/img/calibration", "scaling")
"""
# native python modules
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# scientific stack
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import cm, colormaps
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------------------------------
# Section 1 -- shared constants + rcParams seed
# ---------------------------------------------------------------------------


# Near-black; matplotlib's SVG backend OMITS the fill attribute on pure black, which makes dark-theme SVG viewers render text in their inherited foreground colour (often white -> invisible on white paper). `#010101` is visually identical but non-default, forcing matplotlib to emit `style="fill:#010101"`.
_TEXT_BLACK = "#010101"

# Sign-only delta colours (R15 neutral): blue for decrease, orange for increase. Whether a delta is "good" / "bad" is a domain concern left to the caller.
_BAR_BLUE = "#33ACD1"
_BAR_ORANGE = "#EDD175"

# Topology + heatmap palette (cool = low value, warm = high value).
_TOPOLOGY_CMAP = cm.coolwarm
_HEATMAP_CMAP = "coolwarm"

# Calibration palette (sequential, no signed delta).
_NEUTRAL_BAR = "#7445DA"
_PCTL_GRADIENT = (cm.Purples(0.45), cm.Purples(0.70), cm.Purples(0.92))

# Style dicts (single source of truth; replaces the three drifting copies in the OLD modules).
_LBL_STYLE = dict(fontweight="bold",
                  color=_TEXT_BLACK)
_TITLE_STYLE = dict(fontsize=16,
                    fontweight="bold",
                    pad=20,
                    color=_TEXT_BLACK)
_SUPTITLE_STYLE = dict(fontsize=22,
                       fontweight="bold",
                       color=_TEXT_BLACK)
_GRID_BARS = dict(axis="y",
                  linestyle="--",
                  alpha=0.7,
                  color="#555555")
_GRID_PANEL = dict(alpha=0.5,
                   color=_TEXT_BLACK,
                   linewidth=0.8,
                   linestyle="--")
_GRID_PANEL_3D = dict(alpha=0.5,
                      color=_TEXT_BLACK,
                      linewidth=0.8)

# Yoly-panel tick / label style variants ("GRID" for meta-grid cells, "SINGLE" for standalone figures).
_TICK_STYLE = dict(colors=_TEXT_BLACK,
                   which="both",
                   labelsize=11)
_TICK_STY_3D_GRID = dict(colors=_TEXT_BLACK,
                         which="both",
                         labelsize=10,
                         pad=8)
_TICK_STY_3D_SINGLE = dict(colors=_TEXT_BLACK,
                           which="both",
                           labelsize=11,
                           pad=10)
_LBL_STY_3D_GRID = dict(fontsize=13,
                        labelpad=22,
                        fontweight="bold",
                        color=_TEXT_BLACK)
_LBL_STY_3D_SINGLE = dict(fontsize=15,
                          labelpad=24,
                          fontweight="bold",
                          color=_TEXT_BLACK)
_LBL_STY_2D_GRID = dict(fontsize=12,
                        labelpad=12,
                        fontweight="bold",
                        color=_TEXT_BLACK)
_LBL_STY_2D_SINGLE = dict(fontsize=15,
                          labelpad=14,
                          fontweight="bold",
                          color=_TEXT_BLACK)

# Yoly per-point K annotation style + bounding box (3D vs 2D variants).
_K_LBL_STY_3D = dict(fontsize=14,
                     color=_TEXT_BLACK,
                     fontweight="bold",
                     alpha=0.95,
                     ha="center",
                     va="bottom")
_K_LBL_STY_2D = dict(fontsize=12,
                     color=_TEXT_BLACK,
                     fontweight="bold",
                     alpha=0.9,
                     ha="center",
                     va="bottom")
_K_BBOX = dict(facecolor="white",
               edgecolor="gray",
               alpha=0.8,
               pad=1.5,
               boxstyle="round,pad=0.2")

# Calibration glossary: compact statistical symbols for the bar charts and line legend.
# Mean uses X-hat ($\hat{X}$), median uses $\phi$ (project convention), standard deviation uses $s^2$ (project convention -- treated as a label, not the strict variance definition). Percentiles use the $p_{N}$ subscript form.
_STAT_NAMES = {
    "min": "Min",
    "mean": r"$\hat{X}$",
    "median": r"$\phi$",
    "p50": r"$\phi$",
    "p95": r"$p_{95}$",
    "p99": r"$p_{99}$",
    "max": "Max",
    "std": r"$s^{2}$",
}

# Yoly default coefficient labels: operational name with symbol in parentheses; consumed by every yoly plotter as axis labels.
_DEFAULT_LABELS: Dict[str, str] = {
    "theta": r"Occupancy ($\mathbf{\theta}$)",
    "sigma": r"Stall ($\mathbf{\sigma}$)",
    "eta": r"Effective-Yield ($\mathbf{\eta}$)",
    "phi": r"Memory-Use ($\mathbf{\phi}$)",
}

# The 4 panels of a single-queue 2D yoly chart, ordered (panel_title, x_key, y_key). Titles use the bare LaTeX symbols since the operational names already sit on the axis labels.
_YOLY_PANELS = [
    (r"$\mathbf{\theta}$ vs. $\mathbf{\sigma}$", "theta", "sigma"),
    (r"$\mathbf{\theta}$ vs. $\mathbf{\eta}$", "theta", "eta"),
    (r"$\mathbf{\sigma}$ vs. $\mathbf{\eta}$", "sigma", "eta"),
    (r"$\mathbf{\theta}$ vs. $\mathbf{\phi}$", "theta", "phi"),
]

# Per-node dimensionless coefficient columns + symbols + names (declaration order drives label lines and table columns).
_DIM_COEF_COLS = ("theta", "sigma", "eta", "phi")
_DIM_COEF_SYMS = {
    "theta": r"\theta",
    "sigma": r"\sigma",
    "eta": r"\eta",
    "phi": r"\phi",
}
_DIM_COEF_NAMES = {
    "theta": "Occupancy",
    "sigma": "Stall",
    "eta": "Effective-yield",
    "phi": "Memory-usage",
}

# Confidence -> z-score lookup for two-sided CI bands (consumed by the per-node CI plotter).
_Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

# Public default glossaries (callers import these and pass through to the topology plotters).
QN_GLOSSARY_DEFAULT = [
    "LEGEND",
    r"$\lambda$: Arrival rate (req/s)",
    r"$\mu$: Service rate (req/s)",
    r"$\rho$: Utilisation",
    r"$L$: Avg number in system",
    r"$L_q$: Avg queue length",
    r"$W$: Avg time in system (s)",
    r"$W_q$: Avg wait time (s)",
]

DIM_GLOSSARY_DEFAULT = [
    "LEGEND",
    r"$\theta = \frac{L}{K}$: Occupancy (queue fill ratio)",
    r"$\sigma = \frac{W\lambda}{K}$: Stall (queueing share of capacity)",
    r"$\eta = \frac{\chi \cdot K}{\mu \cdot c}$: Effective-yield (utilisation headroom)",
    r"$\phi = \frac{M_{act}}{M_{buf}}$: Memory-usage (buffer fill)",
]


def _apply_view_rcparams() -> None:
    """*_apply_view_rcparams()* sets the project-wide matplotlib rcParams (white canvas, near-black text, dashed grey grid, project font sizes).

    Called once at module import time; idempotent so re-importing or calling explicitly from a notebook is safe.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "text.color": _TEXT_BLACK,
        "axes.labelcolor": _TEXT_BLACK,
        "axes.edgecolor": _TEXT_BLACK,
        "axes.titlecolor": _TEXT_BLACK,
        "xtick.color": _TEXT_BLACK,
        "ytick.color": _TEXT_BLACK,
        "grid.color": "lightgray",
        "font.size": 10,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


# Apply once at import time so any consumer of `src.view.common` (and transitively, every plotter module) inherits the project look without explicit setup.
_apply_view_rcparams()


# ---------------------------------------------------------------------------
# Section 2 -- DataFrame + numeric helpers
# ---------------------------------------------------------------------------


def _resolve_metrics(df: pd.DataFrame,
                     metrics: Optional[List[str]]) -> List[str]:
    """*_resolve_metrics()* default to every numeric column in `df` when `metrics` is None.

    Args:
        df (pd.DataFrame): frame to inspect.
        metrics (Optional[List[str]]): caller-supplied selection or None.

    Returns:
        List[str]: caller's list (copied) when given, else the numeric columns of `df`.
    """
    if metrics is not None:
        return list(metrics)
    return df.select_dtypes(include="number").columns.tolist()


def _resolve_labels(metrics: List[str],
                    labels: Optional[List[str]]) -> List[str]:
    """*_resolve_labels()* default `labels` to the metric names when the caller does not supply a custom mapping.

    Args:
        metrics (List[str]): metric column names (used as fallback labels).
        labels (Optional[List[str]]): caller-supplied display labels or None.

    Returns:
        List[str]: caller's list (copied) when given, else `metrics`.
    """
    if labels is not None:
        return list(labels)
    return list(metrics)


def _format_value(value: float) -> str:
    """*_format_value()* pick a reasonable format string for a bar label based on the value magnitude.

    Args:
        value (float): numeric value to format.

    Returns:
        str: scientific notation for very small / very large magnitudes, else 2-3 decimal places.
    """
    if np.isnan(value):
        return "nan"
    _abs = abs(value)
    if _abs < 0.01:
        return f"{value:.2e}"
    if _abs < 1:
        return f"{value:.3f}"
    if _abs > 10000:
        return f"{value:.2e}"
    return f"{value:.2f}"


def _generate_color_map(values: List) -> List[str]:
    """*_generate_color_map()* build a vibrant colour palette for N distinct values.

    Selection rule:

        - n <= 12  -> rainbow   (high saturation, wide hue spread)
        - n <= 20  -> Spectral  (perceptually smoother)
        - n >  20  -> turbo     (dense distinct steps)

    The RGB tuples round-trip through HSV as a hook for future saturation / value boosting; the alpha channel is preserved.

    Args:
        values (List): items needing one distinct colour each. Length drives the colormap choice.

    Returns:
        List[str]: hex colour strings aligned 1:1 with `values` (same order the caller supplied).
    """
    _n = len(values)
    if _n <= 12:
        _cmap = colormaps["rainbow"]
    elif _n <= 20:
        _cmap = colormaps["Spectral"]
    else:
        _cmap = colormaps["turbo"]

    # warm end first; matches OLD display.py spacing
    _rgba = _cmap(np.linspace(1.0, 0.0, max(_n, 1)))

    # RGB -> HSV -> RGB pass (no-op today; left as a hook for future saturation boosting)
    _rgb = mcolors.rgb_to_hsv(_rgba[:, :3])
    _boosted = mcolors.hsv_to_rgb(_rgb)

    _rgba_out = np.column_stack([_boosted, _rgba[:, 3]])
    return [mcolors.rgb2hex(_c) for _c in _rgba_out]


# ---------------------------------------------------------------------------
# Section 3 -- I/O
# ---------------------------------------------------------------------------


def _save_figure(fig: Figure,
                 file_path: Optional[str],
                 fname: Optional[str],
                 verbose: bool = False) -> None:
    """*_save_figure()* persist the figure as BOTH `.png` (raster, 300 dpi) and `.svg` (vector) when both `file_path` and `fname` are given; no-op otherwise. Any extension on `fname` is stripped; the stem is reused for both formats.

    Args:
        fig (Figure): matplotlib figure to save.
        file_path (Optional[str]): destination directory (created if missing).
        fname (Optional[str]): output filename; extension (if any) is ignored, the stem drives both `.png` and `.svg` outputs.
        verbose (bool): if True, prints one save message per format.

    Raises:
        ValueError: If `fig.savefig` fails for either format.
    """
    if not (file_path and fname):
        return

    os.makedirs(file_path, exist_ok=True)

    _stem = Path(fname).with_suffix("").name

    for _ext, _extra in (("png", {"dpi": 300}), ("svg", {})):
        _full_path = os.path.join(file_path, f"{_stem}.{_ext}")
        if verbose:
            print(f"Saving plot to: {_full_path}")
        try:
            fig.savefig(_full_path,
                        facecolor="white",
                        bbox_inches="tight",
                        **_extra)
        except (OSError, ValueError, RuntimeError) as _e:
            _msg = f"Error saving {_ext}: {_e}. "
            _msg += f"file_path: {file_path!r}, stem: {_stem!r}"
            raise ValueError(_msg) from _e

    if verbose:
        print(f"Plot saved successfully ({_stem}.png + {_stem}.svg)")


# ---------------------------------------------------------------------------
# Section 4 -- design-contract primitives (stacked figure layout)
# ---------------------------------------------------------------------------


@dataclass
class AxisSpec:
    """Per-axis cosmetic spec for a body panel.

    Fields:
        scale: `"linear"` (default), `"log"`, `"symlog"`, or `"logit"`.
        lim: optional `(lo, hi)` clamp; `None` = autoscale.
        label: axis label (mathtext OK); `None` = no label.
        ticks: explicit list of tick positions; `None` = matplotlib default.
        tick_format: `"plain"` (default), `"sci"`, `"eng"`, or a callable accepting `(value, position) -> str`.
        grid: True (default) draws this axis's grid; False suppresses it.

    Example::

        xspec = AxisSpec(scale="log",
                         label=r"$\\theta$",
                         lim=(1e-3, 1.0),
                         tick_format="sci")
        yspec = AxisSpec(scale="linear",
                         label=r"$\\eta$",
                         tick_format="plain",
                         grid=True)
    """
    scale: str = "linear"
    lim: Optional[Tuple[float, float]] = None
    label: Optional[str] = None
    ticks: Optional[List[float]] = None
    tick_format: Union[str, Callable[[float, int], str]] = "plain"
    grid: bool = True


@dataclass
class BodySpec:
    """Body-grid spec.

    Fields:
        shape: `(n_rows, n_cols)` of the panel grid; default `(1, 1)`.
        panel_kind: `"2d"` (default) or `"3d"`.
        share_x / share_y / share_z: panel-axis sharing flags. `share_z` only honoured for 3D bodies.
        wspace / hspace: inter-panel spacing fractions. Tight defaults (0.05) per the design contract.

    Example::

        # 2x2 grid of 2D panels (yoly chart)
        body = BodySpec(shape=(2, 2), panel_kind="2d", wspace=0.20, hspace=0.30)

        # single 3D panel (system-behaviour cloud)
        body = BodySpec(shape=(1, 1), panel_kind="3d")
    """
    shape: Tuple[int, int] = (1, 1)
    panel_kind: str = "2d"
    share_x: bool = False
    share_y: bool = False
    share_z: bool = False
    wspace: float = 0.05
    hspace: float = 0.05


@dataclass
class FigureLayout:
    """Top-level layout spec: title strip / body grid / footer strip.

    Fields:
        title: figure title text (None renders an empty title strip with the same height so multi-figure grids align).
        title_h: title-strip height fraction of the figure; default 0.06.
        body: `BodySpec` for the panel grid; default `BodySpec()`.
        footer_h: footer-strip height fraction; default 0.18. Set to 0.0 to suppress the footer entirely.
        footer_kind: `"table"` | `"legend"` | `"summary"` | `"none"`; selects which `render_footer_*` helper the plotter is expected to call.
        figsize: `(w, h)` in inches; default `(12, 14)`.

    Example::

        # 1x1 2D figure with a centred legend footer (calibration scaling plot)
        layout = FigureLayout(
            title="Empty-handler scaling",
            body=BodySpec(shape=(1, 1), panel_kind="2d"),
            footer_h=0.10,
            footer_kind="legend",
            figsize=(10, 7),
        )
    """
    title: Optional[str] = None
    title_h: float = 0.06
    body: BodySpec = field(default_factory=BodySpec)
    footer_h: float = 0.18
    footer_kind: str = "table"
    figsize: Tuple[float, float] = (12.0, 14.0)
    # outer_hspace: vertical gap between title strip <-> body grid <-> footer strip, as a fraction of the row height. Default 0.40 keeps a generous gap; per-plotter callers can pass a smaller value (e.g. 0.28 = 30 % less) when the figure already has tight margins.
    outer_hspace: float = 0.40


def _pick_axis_spec(spec: Optional["AxisSpec"],
                    default: "AxisSpec") -> "AxisSpec":
    """*_pick_axis_spec()* return `spec` when the caller supplied one, else `default`.

    Sibling of `_pick_layout` for `AxisSpec` kwargs (`xspec` / `yspec` / `zspec`); replaces the `_xspec = xspec or AxisSpec(...)` short-circuit pattern.

    Args:
        spec (Optional[AxisSpec]): caller-supplied per-axis spec.
        default (AxisSpec): family-specific default.

    Returns:
        AxisSpec: the resolved spec.
    """
    if spec is not None:
        return spec
    return default


def _pick_title(title: Optional[str], default: str) -> str:
    """*_pick_title()* return `title` when the caller supplied one, else `default`.

    Replaces the `title or default` short-circuit pattern that recurs in every plotter that composes a default suptitle.

    Args:
        title (Optional[str]): caller-supplied figure title.
        default (str): family-specific default title.

    Returns:
        str: the resolved title.
    """
    if title is not None:
        return title
    return default


def _pick_layout(layout: Optional["FigureLayout"],
                 default: "FigureLayout") -> "FigureLayout":
    """*_pick_layout()* return `layout` when the caller supplied one, else `default`.

    Replaces the `_layout = layout if layout is not None else _default` pattern that recurs in every plotter; matching the explicit if/else form keeps the edge-case branch (None) visible.

    Args:
        layout (Optional[FigureLayout]): caller-supplied layout override.
        default (FigureLayout): family-specific default.

    Returns:
        FigureLayout: the resolved layout.
    """
    if layout is not None:
        return layout
    return default


def build_stacked_figure(layout: FigureLayout
                         ) -> Tuple[Figure, Dict[str, Any]]:
    """*build_stacked_figure()* construct the title / body / footer scaffolding per the design contract; the body grid anchors title and footer width.

    The returned figure has three vertically-stacked regions sized by `(layout.title_h, body_h, layout.footer_h)` (body_h = remainder). The title strip is one centred axis; the body is `layout.body.shape[0] x layout.body.shape[1]` panel axes (2D or 3D per `layout.body.panel_kind`); the footer is one axis whose horizontal extent matches the body so legends, tables, and summary boxes can never render wider than the body. When `layout.footer_h <= 0.0` the outer grid drops to two rows and `footer_ax` is None.

    Args:
        layout (FigureLayout): full layout spec.

    Raises:
        ValueError: If `layout.body.panel_kind` is not in `{"2d", "3d"}`.

    Returns:
        Tuple[Figure, Dict[str, Any]]: `(fig, regions)` with `regions = {"title_ax": Axes, "body_axes": List[Axes], "footer_ax": Optional[Axes]}`. `body_axes` is a flat list in row-major order. `footer_ax` is None when `layout.footer_h <= 0.0`.

    Example::

        layout = FigureLayout(title="my figure",
                              body=BodySpec(shape=(2, 2), panel_kind="2d"))
        fig, regions = build_stacked_figure(layout)
        for _ax, _data in zip(regions["body_axes"], panel_data):
            _ax.plot(_data["x"], _data["y"])
            attach_axis_spec(_ax, AxisSpec(label="x"), AxisSpec(label="y"))
        render_footer_legend(regions["footer_ax"], handles, labels)
    """
    _kind = layout.body.panel_kind
    if _kind not in ("2d", "3d"):
        raise ValueError(f"BodySpec.panel_kind must be '2d' or '3d'; got {_kind!r}")

    _fig = plt.figure(figsize=layout.figsize, facecolor="white")

    # F1: when the footer is suppressed, drop the row entirely instead of allocating a zero-height slot
    _has_footer = layout.footer_h > 0.0
    if _has_footer:
        _body_h = max(1.0 - layout.title_h - layout.footer_h, 0.05)
        _outer = GridSpec(3, 1,
                          height_ratios=[layout.title_h, _body_h, layout.footer_h],
                          hspace=layout.outer_hspace,
                          figure=_fig)
        _title_slot = _outer[0, 0]
        _body_slot = _outer[1, 0]
        _footer_slot = _outer[2, 0]
    else:
        _body_h = max(1.0 - layout.title_h, 0.05)
        _outer = GridSpec(2, 1,
                          height_ratios=[layout.title_h, _body_h],
                          hspace=layout.outer_hspace,
                          figure=_fig)
        _title_slot = _outer[0, 0]
        _body_slot = _outer[1, 0]
        _footer_slot = None

    # title strip
    _title_ax = _fig.add_subplot(_title_slot)
    _title_ax.set_facecolor("white")
    _title_ax.axis("off")
    if layout.title:
        _title_ax.text(0.5, 0.5, layout.title,
                       ha="center",
                       va="center",
                       transform=_title_ax.transAxes,
                       **_SUPTITLE_STYLE)

    # body grid
    _n_rows, _n_cols = layout.body.shape
    _body_grid = GridSpecFromSubplotSpec(_n_rows, _n_cols,
                                         subplot_spec=_body_slot,
                                         wspace=layout.body.wspace,
                                         hspace=layout.body.hspace)
    _body_axes: List[Any] = []
    for _r in range(_n_rows):
        for _c in range(_n_cols):
            if _kind == "3d":
                _ax = _fig.add_subplot(_body_grid[_r, _c], projection="3d")
            else:
                _ax = _fig.add_subplot(_body_grid[_r, _c])
            _ax.set_facecolor("white")
            _body_axes.append(_ax)

    # footer subgridspec inherits the outer column width, so the footer is clipped to body width by construction
    _footer_ax: Optional[Any] = None
    if _footer_slot is not None:
        _footer_grid = GridSpecFromSubplotSpec(1, 1, subplot_spec=_footer_slot)
        _footer_ax = _fig.add_subplot(_footer_grid[0, 0])
        _footer_ax.set_facecolor("white")
        _footer_ax.axis("off")

    return _fig, {"title_ax": _title_ax,
                  "body_axes": _body_axes,
                  "footer_ax": _footer_ax}


def attach_axis_spec(ax: Any,
                     x: AxisSpec,
                     y: AxisSpec,
                     z: Optional[AxisSpec] = None) -> None:
    """*attach_axis_spec()* apply `scale + lim + label + ticks + tick_format + grid` to a body axis in one call.

    Args:
        ax: matplotlib 2D or 3D axes.
        x (AxisSpec): x-axis spec.
        y (AxisSpec): y-axis spec.
        z (Optional[AxisSpec]): z-axis spec; only honoured for 3D axes.

    Raises:
        ValueError: If `z` is not None on a 2D axis.
    """
    _is_3d = hasattr(ax, "set_zscale")
    if z is not None and not _is_3d:
        raise ValueError("attach_axis_spec: z spec given but ax is 2D")

    _apply_one_axis_spec(ax, x, "x")
    _apply_one_axis_spec(ax, y, "y")
    if _is_3d and z is not None:
        _apply_one_axis_spec(ax, z, "z")

    # one ax.grid call per axis so callers can mix on/off per axis cleanly
    _grid_axes = []
    if x.grid:
        _grid_axes.append("x")
    if y.grid:
        _grid_axes.append("y")
    if _grid_axes:
        _grid_kwargs = {
            k: v for k, v in _GRID_PANEL.items()
            if k in {"alpha", "color", "linewidth", "linestyle"}
        }
        for _g in _grid_axes:
            ax.grid(True, axis=_g, **_grid_kwargs)


def _eng_tick_fmt(value: float, _pos: int = 0) -> str:
    """*_eng_tick_fmt()* return `value` as a 3-significant-figure tick string ("0" when `value == 0`).

    Args:
        value (float): the tick value.
        _pos (int): unused position argument supplied by `matplotlib.ticker.FuncFormatter`.

    Returns:
        str: `"0"` when `value == 0`, else `f"{value:.3g}"`.
    """
    if value == 0:
        return "0"
    return f"{value:.3g}"


def _apply_one_axis_spec(ax: Any, spec: AxisSpec, which: str) -> None:
    """*_apply_one_axis_spec()* set scale, lim, label, ticks, and tick formatter on a single axis of `ax`.

    Notes:
        Tick-format dispatch order: a callable in `spec.tick_format` wins over named formats; `"sci"` uses `_sci_tick_fmt`; `"eng"` uses `_eng_tick_fmt`; `"plain"` leaves matplotlib's default formatter in place.

    Args:
        ax: matplotlib axes (2D or 3D).
        spec (AxisSpec): per-axis spec.
        which (str): `"x"`, `"y"`, or `"z"`.
    """
    _set_scale = getattr(ax, f"set_{which}scale")
    _set_lim = getattr(ax, f"set_{which}lim")
    _set_label = getattr(ax, f"set_{which}label")
    _set_ticks = getattr(ax, f"set_{which}ticks")
    _axis_obj = getattr(ax, f"{which}axis")

    # scale BEFORE lim so log-axis lim validation runs with the right scale; "linear" must override an upstream set_xscale("log") from a body-painting helper.
    _set_scale(spec.scale)
    if spec.lim is not None:
        _set_lim(*spec.lim)
    if spec.label:
        _set_label(spec.label, **_LBL_STYLE)
    if spec.ticks is not None:
        _set_ticks(list(spec.ticks))

    # tick_format dispatch: callable / "sci" / "eng" / "plain" (default keeps matplotlib's formatter)
    if callable(spec.tick_format):
        _axis_obj.set_major_formatter(FuncFormatter(spec.tick_format))
    elif spec.tick_format == "sci":
        _axis_obj.set_major_formatter(FuncFormatter(lambda v, _p: _sci_tick_fmt(v)))
    elif spec.tick_format == "eng":
        _axis_obj.set_major_formatter(FuncFormatter(_eng_tick_fmt))


def render_footer_legend(footer_ax: Any,
                         handles: List[Any],
                         labels: List[str],
                         *,
                         ncol: Optional[int] = None,
                         title: Optional[str] = None) -> None:
    """*render_footer_legend()* draw a centred horizontal legend inside the footer axis (clipped to body width).

    The legend uses the footer axis's bbox as anchor so its width is clipped to the body width by construction (the footer axis was sized that way in `build_stacked_figure`).

    Args:
        footer_ax: footer axis returned by `build_stacked_figure`.
        handles (List): legend handles.
        labels (List[str]): legend labels (1:1 with handles).
        ncol (Optional[int]): column count; defaults to `min(len(labels), 8)`.
        title (Optional[str]): legend title (e.g. "Scenario", "Path").
    """
    if not handles:
        return
    if ncol is not None:
        _ncol = ncol
    else:
        _ncol = min(len(labels), 8)
    # bbox_to_anchor + mode="expand" force the legend to fill the footer axis exactly; combined with the footer axis being body-width by construction, the legend can never overflow horizontally past the main panel grid even when the label set is wide
    footer_ax.legend(handles, labels,
                     loc="center",
                     bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                     mode="expand",
                     ncol=_ncol,
                     fontsize=12,
                     framealpha=0.9,
                     title=title,
                     title_fontsize=13)


def render_footer_table(footer_ax: Any,
                        header: List[str],
                        rows: List[List[str]],
                        *,
                        col_widths: Optional[List[float]] = None) -> None:
    """*render_footer_table()* draw a per-row table inside the footer axis (header row styled bold).

    Args:
        footer_ax: footer axis returned by `build_stacked_figure`.
        header (List[str]): column headers (mathtext OK).
        rows (List[List[str]]): data rows; every row length must equal `len(header)`.
        col_widths (Optional[List[float]]): column-width fractions; defaults to equal widths summing to 1.
    """
    if not rows:
        return
    _ncols = len(header)
    if col_widths is None:
        col_widths = [1.0 / _ncols] * _ncols
    _table = footer_ax.table(cellText=[header] + rows,
                             loc="center",
                             cellLoc="center",
                             colWidths=col_widths)
    _table.auto_set_font_size(False)
    _table.set_fontsize(12)
    _table.scale(1, 1.25)
    for _j in range(_ncols):
        _table[(0, _j)].set_facecolor("#E4EBF1")
        _table[(0, _j)].set_text_props(weight="bold")


def render_footer_summary(footer_ax: Any,
                          lines: List[str],
                          *,
                          anchor: str = "left") -> None:
    """*render_footer_summary()* draw a multi-line summary box anchored left, centre, or right inside the footer axis.

    Args:
        footer_ax: footer axis returned by `build_stacked_figure`.
        lines (List[str]): one entry per visual line (mathtext OK).
        anchor (str): `"left"` (default), `"center"`, or `"right"`.
    """
    if not lines:
        return
    _x = {"left": 0.02, "center": 0.5, "right": 0.98}.get(anchor, 0.02)
    _ha = {"left": "left", "center": "center", "right": "right"}.get(anchor, "left")
    _props = dict(boxstyle="round,pad=0.5",
                  facecolor="lightblue",
                  alpha=0.85,
                  edgecolor="steelblue")
    footer_ax.text(_x, 0.5, "\n".join(lines),
                   transform=footer_ax.transAxes,
                   fontsize=12,
                   color=_TEXT_BLACK,
                   ha=_ha,
                   va="center",
                   bbox=_props)


# ---------------------------------------------------------------------------
# Section 5 -- calibration helpers
# ---------------------------------------------------------------------------


def _sort_n_con_usr_items(handler: Dict[str, Dict[str, float]]
                          ) -> List[Tuple[int, Dict[str, float]]]:
    """*_sort_n_con_usr_items()* iterate `(int(n_con_usr), stats)` pairs sorted by concurrent-user load.

    Args:
        handler (dict): the `handler_scaling` block of a calibration envelope.

    Returns:
        list[tuple[int, dict]]: `[(n_con_usr, stats_dict), ...]` sorted ascending by `n_con_usr`.
    """
    _pairs = []
    for _k, _stats in handler.items():
        _pairs.append((int(_k), _stats))
    _pairs.sort(key=lambda _p: _p[0])
    return _pairs


def _draw_stat_bars(ax: plt.Axes,
                    values: Dict[str, float],
                    order: List[str],
                    unit: str,
                    title: str) -> None:
    """*_draw_stat_bars()* horizontal bar chart of `values` in `order`; each bar labelled with its full stat name and numeric value in `unit`.

    Args:
        ax: matplotlib axis to draw into.
        values (dict): stat-key -> numeric value (e.g. `{"mean_us": 627.7, ...}`).
        order (list): ordered list of stat short keys (`["min", "median", "p95", "p99"]`).
        unit (str): unit string appended to the value (`"ns"` / `"us"`).
        title (str): axis title.
    """
    _ys = list(range(len(order)))
    _labels = []
    _vals = []
    for _key in order:
        _long = _STAT_NAMES.get(_key, _key)
        _labels.append(_long)
        _full_key = f"{_key}_{unit.replace('µ', 'u')}"
        if _full_key in values:
            _v = float(values[_full_key])
        elif _key in values:
            _v = float(values[_key])
        elif _key == "median" and f"p50_{unit}" in values:
            _v = float(values[f"p50_{unit}"])
        else:
            _v = float("nan")
        _vals.append(_v)

    ax.barh(_ys, _vals,
            color=_NEUTRAL_BAR,
            edgecolor=_TEXT_BLACK,
            linewidth=0.6)
    ax.set_yticks(_ys)
    ax.set_yticklabels(_labels, fontsize=10, color=_TEXT_BLACK)
    ax.invert_yaxis()
    # render "us" as the proper $\mu$s mathtext glyph; leave "ns" / other units as plain text
    if unit == "us":
        _unit_lbl = r"$\mu$s"
    else:
        _unit_lbl = unit
    ax.set_xlabel(f"[{_unit_lbl}]", **_LBL_STYLE)
    ax.set_title(title, **_TITLE_STYLE)
    ax.grid(True,
            axis="x",
            linestyle="--",
            alpha=0.5,
            color="#555555")

    for _y, _v in zip(_ys, _vals):
        if not np.isfinite(_v):
            continue
        ax.text(_v, _y, f"  {_v:,.1f} {_unit_lbl}",
                va="center",
                ha="left",
                fontsize=9,
                color=_TEXT_BLACK)


def _draw_handler_scaling_axis(ax: plt.Axes,
                               handler: Dict[str, Dict[str, float]],
                               *,
                               log_y: bool = True) -> None:
    """*_draw_handler_scaling_axis()* plot median / p95 / p99 latency vs `n_con_usr` on one axis.

    Args:
        ax: matplotlib axis to draw into.
        handler (dict): `handler_scaling` block from the calibration envelope.
        log_y (bool): log-scale y-axis; recommended because latencies span several decades.
    """
    _pairs = _sort_n_con_usr_items(handler)
    _xs = [_c for _c, _ in _pairs]
    _median = [float(_s.get("median_us", np.nan)) for _, _s in _pairs]
    _p95 = [float(_s.get("p95_us", np.nan)) for _, _s in _pairs]
    _p99 = [float(_s.get("p99_us", np.nan)) for _, _s in _pairs]

    _c_median, _c_p95, _c_p99 = _PCTL_GRADIENT
    ax.plot(_xs, _median,
            marker="o",
            linestyle="--",
            linewidth=2.0,
            color=_c_median,
            label=_STAT_NAMES["median"])
    ax.plot(_xs, _p95,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            color=_c_p95,
            label=_STAT_NAMES["p95"])
    ax.plot(_xs, _p99,
            marker="^",
            linestyle="--",
            linewidth=2.0,
            color=_c_p99,
            label=_STAT_NAMES["p99"])

    ax.set_xlabel("concurrent users, in-flight requests",
                  **_LBL_STYLE)
    ax.set_ylabel(r"Latency [$\mu$s]", **_LBL_STYLE)
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True,
            which="both",
            linestyle="--",
            alpha=0.5,
            color="#555555")
    ax.legend(loc="upper left", framealpha=0.9)

    for _x, _m in zip(_xs, _median):
        if not np.isfinite(_m):
            continue
        ax.annotate(rf"{_m:,.0f} $\mu$s",
                    xy=(_x, _m),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=9,
                    color=_TEXT_BLACK)


# ---------------------------------------------------------------------------
# Section 6 -- yoly helpers (2D + 3D scatter primitives, marker / path lookup)
# ---------------------------------------------------------------------------


def _sci_tick_fmt(x: float, sig: int = 2) -> str:
    """*_sci_tick_fmt()* return x as scientific-notation text with `sig` significant figures.

    Args:
        x (float): the tick value.
        sig (int): significant figures. Defaults to 2.

    Returns:
        str: `"0"` when `x == 0`, else e-notation like `"3.4e-02"`.
    """
    if x == 0:
        return "0"
    _decimals = max(sig - 1, 0)
    return f"{x:.{_decimals}e}"


def _style_3d_panes(ax: Any) -> None:
    """*_style_3d_panes()* harmonise the look of the three panes of a matplotlib 3D axes.

    Sets pane facecolor to whitesmoke, edge to project-black, and forces the 3D grid colour + linestyle to match the 2D convention.

    Args:
        ax (Any): matplotlib 3D axes object.
    """
    for _axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        _axis.pane.set_facecolor("whitesmoke")
        _axis.pane.set_edgecolor(_TEXT_BLACK)
        _axis._axinfo["grid"]["color"] = _TEXT_BLACK
        _axis._axinfo["grid"]["linewidth"] = 0.8
        _axis._axinfo["grid"]["linestyle"] = "--"


def _apply_sci_format(ax: Any,
                      *,
                      axes_list: Optional[List[str]] = None,
                      sig: int = 2) -> None:
    """*_apply_sci_format()* set a scientific-notation tick formatter on the selected axes of `ax`.

    Args:
        ax (Any): matplotlib axes (2D or 3D).
        axes_list (Optional[List[str]]): which axes to format. Defaults to `["x", "y"]`; pass `["x", "y", "z"]` for 3D.
        sig (int): significant figures passed through to `_sci_tick_fmt`. Defaults to 2.
    """
    if axes_list is not None:
        _axes = axes_list
    else:
        _axes = ["x", "y"]
    for _axis_name in _axes:
        _fmt = FuncFormatter(lambda x, _, s=sig: _sci_tick_fmt(x, s))
        getattr(ax, f"{_axis_name}axis").set_major_formatter(_fmt)


def _apply_logscale(ax: Any,
                    logscale: Union[bool, List[bool]],
                    *,
                    axes_list: Optional[List[str]] = None) -> None:
    """*_apply_logscale()* toggle log scale on selected axes from a bool or per-axis list.

    Args:
        ax (Any): matplotlib axes (2D or 3D).
        logscale (Union[bool, List[bool]]): True logs every axis in `axes_list`; a list of bools sets each axis independently.
        axes_list (Optional[List[str]]): axes to consider. Defaults to `["x", "y"]`; pass `["x", "y", "z"]` for 3D.
    """
    if axes_list is not None:
        _axes = axes_list
    else:
        _axes = ["x", "y"]
    if isinstance(logscale, bool):
        _flags = [logscale] * len(_axes)
    else:
        _flags = list(logscale)
    for _axis_name, _flag in zip(_axes, _flags):
        if _flag:
            getattr(ax, f"set_{_axis_name}scale")("log")


def _format_path_legend(data: Dict[str, Any], path_tag: str) -> str:
    """*_format_path_legend()* build a legend label summarising `c` and `mu` values for one path in a multi-path coefficient dict.

    Args:
        data (Dict[str, Any]): coefficient dict (per-path keys like `c_{<tag>}` / `\\mu_{<tag>}`).
        path_tag (str): path subscript (e.g. `"R"` for read-path).

    Returns:
        str: `"c=1,2,4, mu=900,1800"`. Empty string if neither key is populated.
    """
    _c_key = f"c_{{{path_tag}}}"
    _mu_key = f"\\mu_{{{path_tag}}}"
    _c_vals = np.array(data.get(_c_key, []))
    _mu_vals = np.array(data.get(_mu_key, []))

    _parts: List[str] = []
    if len(_c_vals) > 0:
        _uniq_c = np.unique(_c_vals)
        _c_str = ",".join(str(round(_v)) for _v in _uniq_c)
        _parts.append(rf"$\mathbf{{c}}={_c_str}$")
    if len(_mu_vals) > 0:
        _uniq_mu = np.unique(_mu_vals)
        _mu_str = ",".join(str(round(_v)) for _v in _uniq_mu)
        _parts.append(rf"$\mathbf{{\overline{{\mu}}}}={_mu_str}$")
    return ", ".join(_parts)


def _generate_marker_map(uniq_vals: Union[List[Any], np.ndarray]
                         ) -> Dict[Any, str]:
    """*_generate_marker_map()* assign a matplotlib marker shape to each unique value, cycling through a 14-shape palette.

    Args:
        uniq_vals: distinct values to map.

    Returns:
        Dict[Any, str]: `{value: marker}` in sorted order.
    """
    _markers = ["o", "s", "^", "v", "<", ">", "D",
                "p", "*", "h", "+", "x", "|", "_"]
    return {_v: _markers[_i % len(_markers)]
            for _i, _v in enumerate(sorted(uniq_vals))}


def _find_key_starting_with(data: Dict[str, Any], prefix: str) -> str:
    """*_find_key_starting_with()* return the first key in `data` that starts with `prefix`.

    Lets yoly plotters look up coefficient arrays by semantic prefix (`"\\theta"`, `"c_"`, etc.) without knowing the artifact subscript in advance.

    Args:
        data (Dict[str, Any]): coefficient / sweep dict.
        prefix (str): required key prefix.

    Raises:
        KeyError: If no key matches the prefix.

    Returns:
        str: the matched key.
    """
    for _k in data.keys():
        if _k.startswith(prefix):
            return _k
    raise KeyError(f"no key in data starts with {prefix!r}")


def _pick_coef_short_name(full_sym: str) -> Optional[str]:
    """*_pick_coef_short_name()* return `"theta"` / `"sigma"` / `"eta"` / `"phi"` for a backslash-prefixed coefficient symbol, or `None` when none of the four match.

    Args:
        full_sym (str): full LaTeX symbol (e.g. `\\theta_{TAS_{1}}`).

    Returns:
        Optional[str]: short coefficient name, or `None` if unmatched.
    """
    for _short in ("theta", "sigma", "eta", "phi"):
        if f"\\{_short}" in full_sym:
            return _short
    return None


def _build_coef_map(node_block: Dict[str, Any]) -> Dict[str, str]:
    """*_build_coef_map()* return `{short_name: full_symbol}` over the four derived coefficients present on one node block.

    Args:
        node_block (Dict[str, Any]): per-node dict (keys are coefficient symbols, values are sweep arrays).

    Returns:
        Dict[str, str]: e.g. `{"theta": "\\theta_{TAS_{1}}", ...}`. Missing coefficients are simply omitted.
    """
    _derived = [_k for _k in node_block.keys() if _k.startswith("\\")]
    _coef_map: Dict[str, str] = {}
    for _full in _derived:
        _short = _pick_coef_short_name(_full)
        if _short is not None and _short not in _coef_map:
            _coef_map[_short] = _full
    return _coef_map


def _resolve_groups(paths: Optional[Dict[str, str]],
                    scenarios: Optional[Dict[str, str]]
                    ) -> Tuple[Optional[Dict[str, str]], str]:
    """*_resolve_groups()* choose between `paths=` (PACS idiom) and `scenarios=` (TAS idiom) kwargs.

    Both kwargs drive identical plotter behaviour but live under different names so each case study reads in its own vocabulary:

        - PACS: `paths={"Read": "R_{PACS}", "Write": "W_{PACS}"}`
        - CS-01 TAS: `scenarios={"Before": "baseline_{TAS_{1}}", "After": "aggregate_{TAS_{1}}"}`

    Args:
        paths (Optional[Dict[str, str]]): PACS-idiom grouping.
        scenarios (Optional[Dict[str, str]]): TAS-idiom grouping.

    Raises:
        ValueError: If both kwargs are non-None.

    Returns:
        Tuple[Optional[Dict[str, str]], str]: `(groups, legend_title)`; `groups` is the chosen dict (or None for single-mode) and `legend_title` is the label to use on the panel legend.
    """
    if paths is not None and scenarios is not None:
        raise ValueError(
            "plot_yoly_*: pass `paths=` OR `scenarios=`, not both "
            "(they are aliases; pick the one matching your case-study idiom)")
    if paths is not None:
        return paths, "Path"
    if scenarios is not None:
        return scenarios, "Scenario"
    return None, "System Configuration"


def _compute_grid_dims(n_nodes: int,
                       n_rows: int = 3) -> Tuple[int, int, int, int]:
    """*_compute_grid_dims()* return a row-major `(n_rows, n_cols)` grid plus last-row centring offsets.

    Args:
        n_nodes (int): number of nodes to lay out.
        n_rows (int): outer-grid row count; column count is `ceil(n_nodes / n_rows)`. Defaults to 3.

    Returns:
        Tuple[int, int, int, int]: `(n_rows, n_cols, last_row_idx, n_last_row)`. `n_last_row` is the count of nodes that land in the final row (== `n_cols` when the grid is full).
    """
    _n_cols = (n_nodes + n_rows - 1) // n_rows
    _last_row_idx = n_rows - 1
    _n_last_row = n_nodes - _last_row_idx * _n_cols
    if _n_last_row <= 0:
        _n_last_row = _n_cols
    return n_rows, _n_cols, _last_row_idx, _n_last_row


def _compute_node_pos(nd_idx: int,
                      n_rows: int,
                      n_cols: int,
                      last_row_idx: int,
                      n_last_row: int) -> Tuple[int, int]:
    """*_compute_node_pos()* return `(row, col)` for the n-th node, applying horizontal centring on a short last row.

    Args:
        nd_idx (int): node index in the outer iteration.
        n_rows (int): outer-grid row count.
        n_cols (int): outer-grid column count.
        last_row_idx (int): row index of the final outer row.
        n_last_row (int): number of nodes that land in the final row.

    Returns:
        Tuple[int, int]: `(row, col)` matplotlib gridspec coordinates.
    """
    _row = nd_idx // n_cols
    _raw_col = nd_idx % n_cols
    if _row == last_row_idx and n_last_row < n_cols:
        _col_offset = (n_cols - n_last_row) // 2
        return _row, _raw_col + _col_offset
    return _row, _raw_col


def _format_node_header(node_key: str,
                        name_map: Dict[str, str]) -> str:
    """*_format_node_header()* assemble the per-node header string (mathtext-wrapped node key, with an optional human display name).

    Args:
        node_key (str): artifact identifier (e.g. `"TAS_{1}"`).
        name_map (Dict[str, str]): optional `{node_key: human_name}`.

    Returns:
        str: header line ready for `fig.text(...)`.
    """
    _node_math = f"${node_key}$"
    if node_key in name_map:
        return f"{name_map[node_key]} ({_node_math})"
    return _node_math


def _split_on_K_change(*arrays: np.ndarray
                       ) -> Tuple[np.ndarray, ...]:
    """*_split_on_K_change()* split parallel arrays into NaN-separated sub-sweeps wherever K changes (the last positional array is K).

    Within a (c, mu) group the natural Cartesian sweep order keeps K constant for one sub-sweep (lambda iterates), then K jumps to the next K_factor and lambda restarts. Inserting NaN at every K change breaks the trajectory between sub-sweeps so the line never jumps from a high-theta endpoint back to the next sub-sweep's low-theta start.

    Args:
        *arrays (np.ndarray): parallel 1-D float arrays of the same length; the last one is treated as K.

    Returns:
        Tuple[np.ndarray, ...]: each input array with a `np.nan` inserted at every index where K differs from the previous element. The returned arrays are still parallel and same-length.
    """
    if not arrays:
        return ()
    _K_arr = arrays[-1]
    if len(_K_arr) <= 1:
        return tuple(np.asarray(a, dtype=float) for a in arrays)

    # boundary indices where K[i] != K[i-1] -> insert NaN BEFORE that index in the output (any change, not just decrease)
    _diff = np.diff(_K_arr)
    _break_at = np.where(_diff != 0)[0] + 1

    if len(_break_at) == 0:
        return tuple(np.asarray(a, dtype=float) for a in arrays)

    _out: List[np.ndarray] = []
    for _arr in arrays:
        _arr = np.asarray(_arr, dtype=float)
        _result = np.insert(_arr, _break_at, np.nan)
        _out.append(_result)
    return tuple(_out)


def _paint_single_2d_yoly(ax: Any,
                          coeff_data: Dict[str, Any],
                          x_key: str,
                          y_key: str) -> bool:
    """*_paint_single_2d_yoly()* populate one 2D yoly panel in single-queue mode (colour by `c`, marker by `mu`); within each (c, mu) group the K-sweep points are connected by a dashed trajectory line.

    Args:
        ax (Any): matplotlib 2D axes.
        coeff_data (Dict[str, Any]): single-queue sweep dict.
        x_key (str): short coefficient name for the x-axis.
        y_key (str): short coefficient name for the y-axis.

    Returns:
        bool: True when at least one label was registered.
    """
    _x_full = _find_key_starting_with(coeff_data, f"\\{x_key}")
    _y_full = _find_key_starting_with(coeff_data, f"\\{y_key}")
    _c_full = _find_key_starting_with(coeff_data, "c_")
    _mu_full = _find_key_starting_with(coeff_data, "\\mu")
    _K_full = _find_key_starting_with(coeff_data, "K_")

    _x = np.asarray(coeff_data[_x_full], dtype=float)
    _y = np.asarray(coeff_data[_y_full], dtype=float)
    _c = np.asarray(coeff_data[_c_full], dtype=float)
    _mu = np.asarray(coeff_data[_mu_full], dtype=float)
    _K = np.asarray(coeff_data[_K_full], dtype=float)

    _uniq_c = np.unique(_c)
    _uniq_mu = np.unique(_mu)

    _sorted_c = _uniq_c.tolist()
    _cmap = dict(zip(_sorted_c, _generate_color_map(_sorted_c)))
    _mmap = _generate_marker_map(_uniq_mu.tolist())

    if len(_y) > 0:
        _y_range = float(_y.max() - _y.min())
    else:
        _y_range = 1.0
    _y_off = _y_range * 0.04

    _seen_combos: set = set()
    _seen_K: set = set()

    # one dashed K-trajectory per (c, mu) group, using the data's natural order so each hidden sub-sweep (e.g. fourth dim like lambda) stays paired correctly. NaN-break wherever K decreases so the line separates into sub-sweeps instead of jumping back to origin.
    for _c_val in _uniq_c:
        for _mu_val in _uniq_mu:
            _c_hit = np.abs(_c - _c_val) < 0.1
            _mu_hit = np.abs(_mu - _mu_val) < 0.1
            _group_mask = _c_hit & _mu_hit
            if not np.any(_group_mask):
                continue
            _idx_group = np.where(_group_mask)[0]

            _xs_seg, _ys_seg, _ks_seg = _split_on_K_change(
                _x[_idx_group], _y[_idx_group], _K[_idx_group])

            _combo = (_c_val, _mu_val)
            _label = rf"$\mathbf{{c}}={round(_c_val)},\,\mathbf{{\overline{{\mu}}}}={round(_mu_val)}$"
            _seen_combos.add(_combo)

            ax.plot(_xs_seg, _ys_seg,
                    color=_cmap.get(_c_val, _TEXT_BLACK),
                    marker=_mmap.get(_mu_val, "o"),
                    linestyle="--",
                    linewidth=1.0,
                    markersize=6,
                    markerfacecolor=_cmap.get(_c_val, _TEXT_BLACK),
                    markeredgecolor=_TEXT_BLACK,
                    markeredgewidth=0.4,
                    alpha=0.7,
                    label=_label,
                    rasterized=True)

            # annotate every unique K in this group at the LAST data point of that K-block (high-theta end of the lambda-sweep)
            _K_grp = _K[_idx_group]
            for _K_val in [float(v) for v in np.unique(_K_grp)]:
                if _K_val in _seen_K:
                    continue
                _seen_K.add(_K_val)
                _K_match_local = np.where(_K_grp == _K_val)[0]
                _at_idx = _idx_group[int(_K_match_local[-1])]
                ax.text(float(_x[_at_idx]),
                        float(_y[_at_idx]) + _y_off,
                        f"K={int(_K_val)}",
                        bbox=_K_BBOX,
                        **_K_LBL_STY_2D)
    return len(_seen_combos) > 0


def _paint_groups_2d_yoly(ax: Any,
                          coeff_data: Dict[str, Any],
                          x_key: str,
                          y_key: str,
                          paths: Dict[str, str]) -> bool:
    """*_paint_groups_2d_yoly()* populate one 2D yoly panel in multi-path mode (colour + marker per group); each group's K-sweep points are connected by a dashed trajectory line.

    Args:
        ax (Any): matplotlib 2D axes.
        coeff_data (Dict[str, Any]): sweep dict with per-group arrays keyed by `\\<coef>_{<tag>}`.
        x_key (str): short coefficient name for the x-axis.
        y_key (str): short coefficient name for the y-axis.
        paths (Dict[str, str]): `{display_name: path_tag}` map.

    Returns:
        bool: True when at least one label was registered.
    """
    _path_names = sorted(paths.keys())
    _cmap = dict(zip(_path_names, _generate_color_map(_path_names)))
    _mmap = _generate_marker_map(_path_names)

    _has_label = False
    _seen_K: set = set()

    for _name, _tag in paths.items():
        _x = np.asarray(coeff_data.get(f"\\{x_key}_{{{_tag}}}", []), dtype=float)
        _y = np.asarray(coeff_data.get(f"\\{y_key}_{{{_tag}}}", []), dtype=float)
        _K = np.asarray(coeff_data.get(f"K_{{{_tag}}}", []), dtype=float)

        if len(_x) == 0 or len(_y) == 0:
            continue

        _params = _format_path_legend(coeff_data, _tag)
        if _params:
            _path_lbl = f"{_name} ({_params})"
        else:
            _path_lbl = _name

        if len(_y) > 0:
            _y_range = float(_y.max() - _y.min())
        else:
            _y_range = 1.0
        _y_off = _y_range * 0.04

        # NaN-break wherever K decreases so each sub-sweep within a group renders as its own dashed trajectory
        if len(_K) == len(_x):
            _x_seg, _y_seg, _ks_seg = _split_on_K_change(_x, _y, _K)
        else:
            _x_seg, _y_seg = np.asarray(_x, dtype=float), np.asarray(_y, dtype=float)

        ax.plot(_x_seg, _y_seg,
                color=_cmap.get(_name, _TEXT_BLACK),
                marker=_mmap.get(_name, "o"),
                linestyle="--",
                linewidth=1.0,
                markersize=6,
                markerfacecolor=_cmap.get(_name, _TEXT_BLACK),
                markeredgecolor=_TEXT_BLACK,
                markeredgewidth=0.4,
                alpha=0.7,
                label=_path_lbl,
                rasterized=True)

        # annotate every unique K of this group at the LAST data point of that K-block (high-theta trajectory tip)
        if len(_K) > 0:
            for _K_val in [float(v) for v in np.unique(_K)]:
                if _K_val in _seen_K:
                    continue
                _seen_K.add(_K_val)
                _K_match = np.where(_K == _K_val)[0]
                _at_idx = int(_K_match[-1])
                ax.text(float(_x[_at_idx]),
                        float(_y[_at_idx]) + _y_off,
                        f"K={int(_K_val)}",
                        bbox=_K_BBOX,
                        **_K_LBL_STY_2D)
        _has_label = True
    return _has_label


def _paint_single_3d_yoly(ax: Any, coeff_data: Dict[str, Any]) -> bool:
    """*_paint_single_3d_yoly()* populate one 3D yoly axes in single-queue mode (colour by `c`, marker by `mu`).

    Args:
        ax (Any): matplotlib 3D axes.
        coeff_data (Dict[str, Any]): single-queue sweep dict.

    Returns:
        bool: True when at least one (c, mu) combo was plotted.
    """
    _theta = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "\\theta")], dtype=float)
    _sigma = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "\\sigma")], dtype=float)
    _eta = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "\\eta")], dtype=float)
    _c = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "c_")], dtype=float)
    _mu = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "\\mu")], dtype=float)
    _K = np.asarray(coeff_data[_find_key_starting_with(coeff_data, "K_")], dtype=float)

    _uniq_c = np.unique(_c)
    _uniq_mu = np.unique(_mu)

    _sorted_c = _uniq_c.tolist()
    _cmap = dict(zip(_sorted_c, _generate_color_map(_sorted_c)))
    _mmap = _generate_marker_map(_uniq_mu.tolist())

    if len(_eta) > 0:
        _z_range = float(_eta.max() - _eta.min())
    else:
        _z_range = 1.0
    _z_off = _z_range * 0.05

    _seen_combos: set = set()
    _seen_K: set = set()

    # one dashed K-trajectory per (c, mu) group in 3D (theta x sigma x eta); NaN-break wherever K decreases so multi-sub-sweep groups render as distinct trajectories instead of zig-zags
    for _c_val in _uniq_c:
        for _mu_val in _uniq_mu:
            _c_hit = np.abs(_c - _c_val) < 0.1
            _mu_hit = np.abs(_mu - _mu_val) < 0.1
            _group_mask = _c_hit & _mu_hit
            if not np.any(_group_mask):
                continue
            _idx_group = np.where(_group_mask)[0]

            _theta_seg, _sigma_seg, _eta_seg, _ks_seg = _split_on_K_change(
                _theta[_idx_group], _sigma[_idx_group],
                _eta[_idx_group], _K[_idx_group])

            _combo = (_c_val, _mu_val)
            _label = rf"$\mathbf{{c}}={round(_c_val)},\,\mathbf{{\overline{{\mu}}}}={round(_mu_val)}$"
            _seen_combos.add(_combo)

            ax.plot(_theta_seg, _sigma_seg, _eta_seg,
                    color=_cmap.get(_c_val, _TEXT_BLACK),
                    marker=_mmap.get(_mu_val, "o"),
                    linestyle="--",
                    linewidth=0.8,
                    markersize=4,
                    markerfacecolor=_cmap.get(_c_val, _TEXT_BLACK),
                    markeredgecolor=_TEXT_BLACK,
                    markeredgewidth=0.3,
                    alpha=0.7,
                    label=_label)

            # annotate every unique K at the LAST point of that K-block (high-theta trajectory tip)
            _K_grp = _K[_idx_group]
            for _K_val in [float(v) for v in np.unique(_K_grp)]:
                if _K_val in _seen_K:
                    continue
                _seen_K.add(_K_val)
                _K_match_local = np.where(_K_grp == _K_val)[0]
                _at_idx = _idx_group[int(_K_match_local[-1])]
                ax.text(float(_theta[_at_idx]),
                        float(_sigma[_at_idx]),
                        float(_eta[_at_idx]) + _z_off,
                        f"K={int(_K_val)}",
                        bbox=_K_BBOX,
                        **_K_LBL_STY_3D)
    return len(_seen_combos) > 0


def _paint_groups_3d_yoly(ax: Any,
                          coeff_data: Dict[str, Any],
                          groups: Dict[str, str]) -> bool:
    """*_paint_groups_3d_yoly()* populate one 3D yoly axes in grouped mode (paths or scenarios).

    Args:
        ax (Any): matplotlib 3D axes.
        coeff_data (Dict[str, Any]): sweep dict with per-group arrays keyed by `\\<coef>_{<tag>}`.
        groups (Dict[str, str]): `{display_name: tag}` map.

    Returns:
        bool: True when at least one group produced a labelled scatter.
    """
    _names = sorted(groups.keys())
    _cmap = dict(zip(_names, _generate_color_map(_names)))
    _mmap = _generate_marker_map(_names)

    _has_label = False
    _seen_K: set = set()

    _all_eta: List[float] = []
    for _tag in groups.values():
        _eta_arr = coeff_data.get(f"\\eta_{{{_tag}}}", [])
        _all_eta.extend(np.asarray(_eta_arr, dtype=float).tolist())
    if len(_all_eta) > 1:
        _z_range = max(_all_eta) - min(_all_eta)
    else:
        _z_range = 1.0
    _z_off = _z_range * 0.05

    for _name, _tag in groups.items():
        _theta = np.asarray(coeff_data.get(f"\\theta_{{{_tag}}}", []), dtype=float)
        _sigma = np.asarray(coeff_data.get(f"\\sigma_{{{_tag}}}", []), dtype=float)
        _eta = np.asarray(coeff_data.get(f"\\eta_{{{_tag}}}", []), dtype=float)
        _K = np.asarray(coeff_data.get(f"K_{{{_tag}}}", []), dtype=float)

        if len(_theta) == 0 or len(_sigma) == 0 or len(_eta) == 0:
            continue

        _params = _format_path_legend(coeff_data, _tag)
        if _params:
            _group_lbl = f"{_name} ({_params})"
        else:
            _group_lbl = _name

        # NaN-break wherever K decreases so each sub-sweep renders as its own dashed 3D trajectory
        if len(_K) == len(_theta):
            _theta_seg, _sigma_seg, _eta_seg, _ks_seg = _split_on_K_change(
                _theta, _sigma, _eta, _K)
        else:
            _theta_seg = np.asarray(_theta, dtype=float)
            _sigma_seg = np.asarray(_sigma, dtype=float)
            _eta_seg = np.asarray(_eta, dtype=float)

        ax.plot(_theta_seg, _sigma_seg, _eta_seg,
                color=_cmap.get(_name, _TEXT_BLACK),
                marker=_mmap.get(_name, "o"),
                linestyle="--",
                linewidth=0.8,
                markersize=4,
                markerfacecolor=_cmap.get(_name, _TEXT_BLACK),
                markeredgecolor=_TEXT_BLACK,
                markeredgewidth=0.3,
                alpha=0.7,
                label=_group_lbl)

        if len(_K) > 0:
            # annotate every unique K at the LAST point of that K-block (high-theta trajectory tip)
            for _K_val in [float(v) for v in np.unique(_K)]:
                if _K_val in _seen_K:
                    continue
                _seen_K.add(_K_val)
                _K_match = np.where(_K == _K_val)[0]
                _at_idx = int(_K_match[-1])
                ax.text(float(_theta[_at_idx]),
                        float(_sigma[_at_idx]),
                        float(_eta[_at_idx]) + _z_off,
                        f"K={int(_K_val)}",
                        bbox=_K_BBOX,
                        **_K_LBL_STY_3D)
        _has_label = True
    return _has_label


# ---------------------------------------------------------------------------
# Section 7 -- topology helpers (graph construction, axis painting, table + summary)
# ---------------------------------------------------------------------------


def _bfs_layout_shared(routs: List[np.ndarray]) -> dict:
    """*_bfs_layout_shared()* compute a single BFS layout over the union of all routing matrices so every subplot in a grid uses the same node positions.

    `networkx.bfs_layout` requires every node be reachable from `start`. A node with no incoming routing edge (e.g. an atomic that saw zero observed traffic in the empirical-collapsed routing) is unreachable and would raise `NetworkXError`. To keep the topology plot robust to such partial-data cases, any node unreachable from node 0 gets a temporary layout-only edge `(0, node)` before the layout call; that edge positions the orphan node sensibly without appearing in the rendered topology, which draws edges from the routing matrices, not from this union graph.

    Args:
        routs (List[np.ndarray]): list of routing matrices (same N x N shape).

    Returns:
        dict: node-index -> (x, y) position dict from `networkx.bfs_layout(G, start=0)`.
    """
    _n = routs[0].shape[0]
    _union = nx.DiGraph()
    for _i in range(_n):
        _union.add_node(_i)
    for _r in routs:
        for _i in range(_n):
            for _j in range(_n):
                if _r[_i, _j] > 0:
                    _union.add_edge(_i, _j)
    # Add layout-only edges from node 0 to any node not reachable from it, so
    # bfs_layout does not raise on a partially-disconnected routing graph.
    _reachable = set(nx.descendants(_union, 0)) | {0}
    for _node in range(_n):
        if _node not in _reachable:
            _union.add_edge(0, _node)
    return nx.bfs_layout(_union, start=0)


def _build_topology_graph(rout: np.ndarray,
                          nd_names: List[str]) -> nx.DiGraph:
    """*_build_topology_graph()* convert a routing matrix into a `networkx.DiGraph` with edge weights and display labels.

    Args:
        rout (np.ndarray): `(n, n)` routing probability matrix.
        nd_names (List[str]): per-node display names, aligned with the matrix row / column order.

    Returns:
        nx.DiGraph: directed graph with `weight` on every edge.
    """
    _n = rout.shape[0]
    _graph = nx.DiGraph()
    for _i in range(_n):
        _graph.add_node(_i, name=nd_names[_i])
    for _i in range(_n):
        for _j in range(_n):
            if rout[_i, _j] > 0:
                _graph.add_edge(_i, _j, weight=float(rout[_i, _j]))
    return _graph


def _draw_qn_topology_axis(ax: plt.Axes,
                           graph: nx.DiGraph,
                           pos: dict,
                           nds: pd.DataFrame,
                           nd_names: List[str],
                           edge_label_threshold: float = 0.01,
                           rho_max: Optional[float] = None) -> None:
    """*_draw_qn_topology_axis()* render one queue-network topology onto `ax`, colouring nodes by `rho` when the column is present.

    Args:
        ax: matplotlib axis to draw into.
        graph (nx.DiGraph): prebuilt topology graph.
        pos (dict): BFS layout positions.
        nds (pd.DataFrame): per-node metrics frame.
        nd_names (List[str]): display names aligned with the graph.
        edge_label_threshold (float): probabilities below this are drawn without a numeric label.
        rho_max (Optional[float]): shared scale for node colouring in multi-panel plots; defaults to the frame's own `rho.max()`.
    """
    _n = len(nd_names)

    if "rho" in nds.columns:
        _rhos = nds["rho"].to_numpy(dtype=float)
        _scale = rho_max if rho_max is not None else float(_rhos.max())
        _scale = max(_scale, 1e-9)
        _node_colors = [_TOPOLOGY_CMAP(_r / _scale) for _r in _rhos]
    else:
        _node_colors = ["skyblue"] * _n

    # node_size bumped 20 % (1500 -> 1800) per request
    nx.draw_networkx_nodes(graph, pos,
                           node_size=1800,
                           node_color=_node_colors,
                           alpha=0.9,
                           ax=ax)
    nx.draw_networkx_edges(graph, pos,
                           width=1.5,
                           alpha=0.7,
                           edge_color=_TEXT_BLACK,
                           arrows=True,
                           arrowsize=20,
                           arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.2",
                           ax=ax)
    _edge_lbl = {
        (_u, _v): f"{_d['weight']:.2f}"
        for _u, _v, _d in graph.edges(data=True)
        if _d["weight"] >= edge_label_threshold
    }
    _edge_bbox = dict(facecolor="white",
                      edgecolor="none",
                      alpha=0.9,
                      pad=0.3)
    nx.draw_networkx_edge_labels(graph, pos,
                                 edge_labels=_edge_lbl,
                                 font_size=11,
                                 font_color=_TEXT_BLACK,
                                 font_weight="light",
                                 bbox=_edge_bbox,
                                 label_pos=0.5,
                                 connectionstyle="arc3,rad=0.2",
                                 ax=ax)
    _labels = {}
    for _i in range(_n):
        _name = nd_names[_i]
        _parts = [rf"${_name}$"]
        if "L" in nds.columns:
            _parts.append(rf"$L = {nds['L'].iloc[_i]:.2f}$")
        _labels[_i] = "\n".join(_parts)
    # 10 % thinner labels per request: drop the `\mathbf{...}` mathtext-bold wrapping (keeps subscripts intact via plain `${...}$`) AND drop font_weight from bold (700) to semibold (600)
    nx.draw_networkx_labels(graph, pos,
                            labels=_labels,
                            font_size=12,
                            font_weight="semibold",
                            font_color=_TEXT_BLACK,
                            ax=ax)
    ax.set_axis_off()


def _draw_dim_topology_axis(ax: plt.Axes,
                            graph: nx.DiGraph,
                            pos: dict,
                            nds: pd.DataFrame,
                            nd_names: List[str],
                            *,
                            color_by: str = "eta",
                            edge_label_threshold: float = 0.01,
                            color_min: Optional[float] = None,
                            color_max: Optional[float] = None) -> None:
    """*_draw_dim_topology_axis()* render the dimensionless topology onto `ax`; nodes coloured by `color_by` using min-max normalisation, with a 2-line label per node showing the artifact key and its $\\theta$ value.

    Args:
        ax: matplotlib axis to draw into.
        graph (nx.DiGraph): prebuilt topology graph.
        pos (dict): BFS layout positions.
        nds (pd.DataFrame): per-node coefficients frame.
        nd_names (List[str]): display names aligned with the graph.
        color_by (str): column driving the node colour. Defaults to `"eta"`.
        edge_label_threshold (float): probabilities below this are drawn without a numeric label.
        color_min (Optional[float]): shared colour-scale lower bound; defaults to `nds[color_by].min()`.
        color_max (Optional[float]): shared colour-scale upper bound; defaults to `nds[color_by].max()`.
    """
    _n = len(nd_names)
    if color_by in nds.columns:
        _vals = nds[color_by].to_numpy(dtype=float)
        _vmin = color_min if color_min is not None else float(_vals.min())
        _vmax = color_max if color_max is not None else float(_vals.max())
        _span = max(_vmax - _vmin, 1e-9)
        _node_colors = [_TOPOLOGY_CMAP((_v - _vmin) / _span) for _v in _vals]
    else:
        _node_colors = ["skyblue"] * _n

    nx.draw_networkx_nodes(graph, pos,
                           node_size=1800,
                           node_color=_node_colors,
                           alpha=0.9,
                           ax=ax)
    nx.draw_networkx_edges(graph, pos,
                           width=1.5,
                           alpha=0.7,
                           edge_color=_TEXT_BLACK,
                           arrows=True,
                           arrowsize=20,
                           arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.2",
                           ax=ax)
    _edge_lbl = {
        (_u, _v): f"{_d['weight']:.2f}"
        for _u, _v, _d in graph.edges(data=True)
        if _d["weight"] >= edge_label_threshold
    }
    _edge_bbox = dict(facecolor="white",
                      edgecolor="none",
                      alpha=0.9,
                      pad=0.3)
    nx.draw_networkx_edge_labels(graph, pos,
                                 edge_labels=_edge_lbl,
                                 font_size=11,
                                 font_color=_TEXT_BLACK,
                                 font_weight="light",
                                 bbox=_edge_bbox,
                                 label_pos=0.5,
                                 connectionstyle="arc3,rad=0.2",
                                 ax=ax)
    _labels: dict = {}
    for _i in range(_n):
        _name = nd_names[_i]
        _parts = [rf"$\mathbf{{{_name}}}$"]
        if "theta" in nds.columns:
            _val = float(nds["theta"].iloc[_i])
            _parts.append(rf"$\mathbf{{\theta = {_val:.2e}}}$")
        _labels[_i] = "\n".join(_parts)
    nx.draw_networkx_labels(graph, pos,
                            labels=_labels,
                            font_size=12,
                            font_weight="bold",
                            font_color=_TEXT_BLACK,
                            ax=ax)
    ax.set_axis_off()


def _add_param_glossary(ax: plt.Axes,
                        glossary: List[str],
                        *,
                        corner: str = "lower right") -> None:
    """*_add_param_glossary()* overlay a caller-supplied parameter glossary in the chosen corner of a topology axis.

    Args:
        ax: matplotlib axis (graph subplot).
        glossary (List[str]): lines to render verbatim.
        corner (str): `"lower right"`, `"lower left"`, `"upper right"`, or `"upper left"`.
    """
    _text = "\n".join(glossary)
    _y = 0.02 if "lower" in corner else 0.98
    _x = 0.98 if "right" in corner else 0.02
    _ha = "right" if "right" in corner else "left"
    _va = "bottom" if "lower" in corner else "top"
    _props = dict(boxstyle="round,pad=0.4",
                  facecolor="white",
                  alpha=0.85,
                  edgecolor="gray")
    ax.text(_x, _y, _text,
            transform=ax.transAxes,
            fontsize=14,
            color=_TEXT_BLACK,
            verticalalignment=_va,
            horizontalalignment=_ha,
            bbox=_props)


def _add_qn_network_summary(ax: plt.Axes,
                            net: pd.Series,
                            *,
                            corner: str = "upper right") -> None:
    """*_add_qn_network_summary()* overlay the network-wide aggregate metrics as a text box.

    Args:
        ax: matplotlib axis (graph subplot).
        net (pd.Series): single row from `aggregate_net()`.
        corner (str): anchor corner.
    """
    # End-to-end values (R1 / R2 comparable) are the headline. W_net per-visit is included for reference.
    _w_e2e = net.get("W_e2e", float("nan"))
    _eps_e2e = net.get("eps_e2e", float("nan"))
    _chi_out = net.get("chi_out", float("nan"))
    _lines = [
        r"$\mathbf{NETWORK}$",
        rf"$\mathbf{{\overline{{\mu}}}}$: {net['avg_mu']:.2f} [req/s]",
        rf"$\mathbf{{\overline{{\rho}}}}$: {net['avg_rho']:.3f}",
        rf"$\mathbf{{L_{{net}}}}$: {net['L_net']:.2f} [req]",
        rf"$\mathbf{{W_{{e2e}}}}$: {_w_e2e*1000:.2f} [ms/req]",
        rf"$\mathbf{{\varepsilon_{{e2e}}}}$: {_eps_e2e*100:.2f} [%]",
        rf"$\mathbf{{\chi_{{out}}}}$: {_chi_out:.2f} [req/s]",
        rf"$\mathbf{{TP_{{net}}}}$: {net['total_throughput']:.2f} [req/s]",
    ]
    _text = "\n".join(_lines)
    _y = 0.98 if "upper" in corner else 0.02
    _x = 0.98 if "right" in corner else 0.02
    _va = "top" if "upper" in corner else "bottom"
    _ha = "right" if "right" in corner else "left"
    _props = dict(boxstyle="round,pad=0.5",
                  facecolor="lightblue",
                  alpha=0.85,
                  edgecolor="steelblue")
    ax.text(_x, _y, _text,
            transform=ax.transAxes,
            fontsize=14,
            color=_TEXT_BLACK,
            verticalalignment=_va,
            horizontalalignment=_ha,
            bbox=_props)


def _add_dim_network_summary(ax: plt.Axes,
                             nds: pd.DataFrame,
                             *,
                             corner: str = "upper right") -> None:
    """*_add_dim_network_summary()* overlay the architecture-wide coefficient averages as a text box.

    Args:
        ax: matplotlib axis (graph subplot).
        nds (pd.DataFrame): per-node coefficients frame.
        corner (str): anchor corner.
    """
    _lines = [r"$\mathbf{NETWORK}$"]
    for _c in _DIM_COEF_COLS:
        if _c in nds.columns:
            _sym = _DIM_COEF_SYMS[_c]
            _name = _DIM_COEF_NAMES.get(_c, _c)
            _mean = float(nds[_c].mean())
            _lines.append(
                rf"$\mathbf{{\overline{{{_sym}}}}}$ ({_name}): {_mean:.2e}")
    _text = "\n".join(_lines)
    _y = 0.98 if "upper" in corner else 0.02
    _x = 0.98 if "right" in corner else 0.02
    _va = "top" if "upper" in corner else "bottom"
    _ha = "right" if "right" in corner else "left"
    _props = dict(boxstyle="round,pad=0.5",
                  facecolor="lightblue",
                  alpha=0.85,
                  edgecolor="steelblue")
    ax.text(_x, _y, _text,
            transform=ax.transAxes,
            fontsize=14,
            color=_TEXT_BLACK,
            verticalalignment=_va,
            horizontalalignment=_ha,
            bbox=_props)


def _add_qn_node_table(ax: plt.Axes,
                       nds: pd.DataFrame,
                       nd_names: List[str]) -> None:
    """*_add_qn_node_table()* draw a per-node QN metrics table into a dedicated axis below the topology.

    Args:
        ax: matplotlib axis reserved for the table (axis is hidden).
        nds (pd.DataFrame): per-node metrics frame.
        nd_names (List[str]): display names aligned with the rows.
    """
    ax.set_axis_off()
    _header = ["Component",
               r"$\lambda$ [req/s]",
               r"$\mu$ [req/s]",
               r"$\rho$",
               r"$L$ [req]",
               r"$L_q$ [req]",
               r"$W$ [s]",
               r"$W_q$ [s]"]
    _rows = [_header]
    for _i in range(len(nd_names)):
        _row = nds.iloc[_i]
        _rows.append([
            f"${nd_names[_i]}$",
            f"{_row['lambda']:.2f}",
            f"{_row['mu']:.2f}",
            f"{_row['rho']:.3f}",
            f"{_row['L']:.2f}",
            f"{_row['Lq']:.2f}",
            f"{_row['W']:.4e}",
            f"{_row['Wq']:.4e}",
        ])
    # columns 15 % thinner per request: 0.12 -> 0.102, 0.125 -> 0.106; total table width drops from 0.995 to 0.844 (the table is anchored loc="center" so the saved width comes off both sides)
    _table = ax.table(cellText=_rows,
                      loc="center",
                      cellLoc="center",
                      colWidths=[0.102] + [0.106] * 7)
    _table.auto_set_font_size(False)
    _table.set_fontsize(13)
    _table.scale(1, 1.50)
    for _j in range(len(_header)):
        _table[(0, _j)].set_facecolor("#E4EBF1")
        _table[(0, _j)].set_text_props(weight="bold")


def _add_dim_node_table(ax: plt.Axes,
                        nds: pd.DataFrame,
                        nd_names: List[str]) -> None:
    """*_add_dim_node_table()* draw a per-node coefficient table (Component, theta, sigma, eta, phi).

    Args:
        ax: matplotlib axis reserved for the table (axis is hidden).
        nds (pd.DataFrame): per-node coefficients frame.
        nd_names (List[str]): display names aligned with the rows.
    """
    ax.set_axis_off()
    _present_cols = [_c for _c in _DIM_COEF_COLS if _c in nds.columns]
    _header = ["Component"]
    for _c in _present_cols:
        _sym = _DIM_COEF_SYMS[_c]
        _header.append(rf"${_sym}$")
    _rows = [_header]
    for _i in range(len(nd_names)):
        _row = nds.iloc[_i]
        _cells = [f"${nd_names[_i]}$"]
        for _c in _present_cols:
            _cells.append(f"{float(_row[_c]):.2e}")
        _rows.append(_cells)
    # match the QN-table sizing on the large topology canvas: wide cells, taller rows, readable font
    _col_widths = [0.16] + [0.18] * len(_present_cols)
    _table = ax.table(cellText=_rows,
                      loc="center",
                      cellLoc="center",
                      colWidths=_col_widths)
    _table.auto_set_font_size(False)
    _table.set_fontsize(13)
    _table.scale(1, 1.50)
    for _j in range(len(_header)):
        _table[(0, _j)].set_facecolor("#E4EBF1")
        _table[(0, _j)].set_text_props(weight="bold")


# Public symbols (consumed by plotter modules and re-exported by `src/view/__init__.py`).
__all__ = [
    # design-contract primitives
    "AxisSpec",
    "BodySpec",
    "FigureLayout",
    "build_stacked_figure",
    "attach_axis_spec",
    "render_footer_legend",
    "render_footer_table",
    "render_footer_summary",
    # public defaults the plotters expose to callers
    "QN_GLOSSARY_DEFAULT",
    "DIM_GLOSSARY_DEFAULT",
]
