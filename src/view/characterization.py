# -*- coding: utf-8 -*-
"""Plotters for the calibration envelope.

Six per-probe panels plus a 2x3 summary grid (with a Report panel) and an overlay variant for cross-deployment comparison:

- `plot_timer`: clock-tick `Min`, `Phi` (median), `chi-bar` (mean), `Max`; std-dev `s^2` rendered as +/- error caps.
- `plot_jitter`: `chi-bar`, `Phi`, `p_{95}`, `p_{99}` relative to target sleep; `s^2` as caps.
- `plot_loopback`: `Min`, `Phi`, `p_{95}`, `p_{99}`; precision band `[Phi - s^2, Phi + s^2]` shaded behind bars.
- `plot_handler_scaling`: linear latency vs concurrency; three traces (`Phi`, `p_{95}`, `p_{99}`) + p95-p99 fill.
- `plot_workers_scaling`: worker rate (req/s) + efficiency vs `n_workers`; vertical marker at the stable-knee.
- `plot_rate_sweep`: latency + loss-rate vs rate; verifiable region (below saturation) shaded green.
- `plot_calibration_summary`: 2x3 grid; bottom-left slot shows `handler_scaling` on localhost and `workers_scaling` on multiprocess (the more actionable probe per mode); bottom-right is the Report.
- `plot_envelope_overlay`: same 2x3 grid; the bottom-left slot falls back to `handler_scaling` when any envelope lacks workers data, ensuring localhost-vs-multiprocess overlays still produce a coherent panel.

`plot_rate_sweep` and `plot_handler_scaling` subtract the loopback median floor from displayed values by default (`subtract_floor=True`); the panels measure service work on top of the floor, not the floor itself.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.view.common import (
    _NEUTRAL_BAR,
    _PCTL_GRADIENT,
    _TEXT_BLACK,
    _generate_color_map,
    _save_figure,
)


# Per-trace tints for handler scaling (Phi / p99 traces; the p95 trace inherits the panel colour).
_PHI_COLOR, _, _P99_COLOR = _PCTL_GRADIENT

_TIGHT_RECT: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.97)


def plot_timer(envelope: dict[str, Any],
               *,
               title: str | None = None,
               file_path: str | None = None,
               fname: str = "timer",
               verbose: bool = False) -> Figure:
    """Horizontal bar chart of timer min / Phi / chi-bar / s^2 / max (ns).

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `envelope["timer"]`).
        title (str | None, optional): override panel title; default is `f"Timer (n={samples_n})"`.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem used by `_save_figure`. Defaults to `"timer"`.
        verbose (bool, optional): when True, `_save_figure` logs the saved paths.

    Returns:
        Figure: the matplotlib figure (caller closes it; the function never calls `plt.show`).
    """
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _draw_timer(_ax, envelope.get("timer", {}), title=title)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_jitter(envelope: dict[str, Any],
                *,
                title: str | None = None,
                file_path: str | None = None,
                fname: str = "jitter",
                verbose: bool = False) -> Figure:
    """Horizontal bar chart of asyncio.sleep chi-bar / Phi / p_{95} / p_{99} / s^2 against the target sleep.

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `envelope["jitter"]`).
        title (str | None, optional): override panel title.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"jitter"`.
        verbose (bool, optional): log saved paths when True.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _draw_jitter(_ax, envelope.get("jitter", {}), title=title)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_loopback(envelope: dict[str, Any],
                  *,
                  title: str | None = None,
                  file_path: str | None = None,
                  fname: str = "loopback",
                  verbose: bool = False) -> Figure:
    """Horizontal bar chart of TCP loopback Min / Phi / p_{95} / p_{99} (us).

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `envelope["loopback"]`).
        title (str | None, optional): override panel title.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"loopback"`.
        verbose (bool, optional): log saved paths when True.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _ax = plt.subplots(figsize=(6, 3.5))
    _draw_loopback(_ax, envelope.get("loopback", {}), title=title)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_handler_scaling(envelope: dict[str, Any],
                         *,
                         title: str | None = None,
                         file_path: str | None = None,
                         fname: str = "handler_scaling",
                         verbose: bool = False,
                         subtract_floor: bool = True) -> Figure:
    """Linear latency-change (us) vs concurrency, three traces (Phi, p_{95}, p_{99}).

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `handler_scaling` + `loopback`).
        title (str | None, optional): override panel title.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"handler_scaling"`.
        verbose (bool, optional): log saved paths when True.
        subtract_floor (bool, optional): subtract `loopback.median_us` from displayed latencies. Defaults to True.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _draw_handler_scaling(_ax,
                          envelope.get("handler_scaling", {}),
                          loopback=envelope.get("loopback", {}),
                          title=title,
                          subtract_floor=subtract_floor)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_rate_sweep(envelope: dict[str, Any],
                    *,
                    title: str | None = None,
                    file_path: str | None = None,
                    fname: str = "rate_sweep",
                    verbose: bool = False,
                    subtract_floor: bool = True) -> Figure:
    """Twin-axis plot of p_{95} latency (us) + loss percent vs target rate.

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `rate` + `loopback`).
        title (str | None, optional): override panel title.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"rate_sweep"`.
        verbose (bool, optional): log saved paths when True.
        subtract_floor (bool, optional): subtract `loopback.median_us` from displayed latencies.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _draw_rate_sweep(_ax,
                     envelope.get("rate", {}),
                     loopback=envelope.get("loopback", {}),
                     title=title,
                     subtract_floor=subtract_floor)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_workers_scaling(envelope: dict[str, Any],
                         *,
                         title: str | None = None,
                         file_path: str | None = None,
                         fname: str = "workers_scaling",
                         verbose: bool = False) -> Figure:
    """Twin-axis plot of worker rate (req/s) + efficiency percent vs `n_workers`, with the stable-knee marker.

    Args:
        envelope (dict[str, Any]): populated calibration envelope (reads `envelope["workers_scaling"]`).
        title (str | None, optional): override panel title.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"workers_scaling"`.
        verbose (bool, optional): log saved paths when True.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _draw_workers_scaling(_ax, envelope.get("workers_scaling", {}), title=title)
    _fig.tight_layout()
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_calibration_summary(envelope: dict[str, Any],
                             *,
                             title: str | None = None,
                             file_path: str | None = None,
                             fname: str = "summary",
                             verbose: bool = False,
                             subtract_floor: bool = True) -> Figure:
    """Per-envelope summary grid (2x3) with the bottom-left slot driven by deployment mode.

    Localhost runs show `handler_scaling` (the per-handler concurrency probe); multiprocess runs show `workers_scaling` (the parallel-limit probe). Both probes still record data on every run; only the panel slot is conditional. Layout never changes shape.

    Args:
        envelope (dict[str, Any]): populated calibration envelope.
        title (str | None, optional): override the figure suptitle; default is `f"Calibration: {host}, dpl={dpl}"`.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"summary"`.
        verbose (bool, optional): log saved paths when True.
        subtract_floor (bool, optional): subtract `loopback.median_us` from displayed latencies on the handler-scaling and rate-sweep panels.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _fig, _axes = plt.subplots(2, 3, figsize=(18, 9))
    _loopback = envelope.get("loopback", {})
    _draw_timer(_axes[0, 0], envelope.get("timer", {}))
    _draw_jitter(_axes[0, 1], envelope.get("jitter", {}))
    _draw_loopback(_axes[0, 2], _loopback)
    _draw_scaling_slot(_axes[1, 0], envelope,
                       loopback=_loopback, subtract_floor=subtract_floor)
    _draw_rate_sweep(_axes[1, 1],
                     envelope.get("rate", {}),
                     loopback=_loopback,
                     subtract_floor=subtract_floor)
    _draw_report(_axes[1, 2], envelope.get("gate", {}), envelope=envelope)
    if title is None:
        _title = f"Calibration: {envelope.get('dpl', '?')}"
    else:
        _title = title
    _fig.suptitle(_title, color=_TEXT_BLACK, fontsize=15, fontweight="bold")
    _fig.tight_layout(rect=_TIGHT_RECT)
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def plot_envelope_overlay(envelopes: dict[str, dict[str, Any]],
                          *,
                          title: str | None = None,
                          file_path: str | None = None,
                          fname: str = "overlay",
                          verbose: bool = False,
                          subtract_floor: bool = True) -> Figure:
    """Cross-envelope overlay grid (2 cols x 4 rows, portrait); bottom row is the Report (spans both columns).

    Layout:
        Row 0: Timer            | Jitter
        Row 1: Loopback         | Rate sweep
        Row 2: Handler scaling  | Workers scaling
        Row 3: Calibration Reports (spans both columns)

    Both scaling panels render every envelope's data even when only one mode populated each block (cross-mode overlays compare handler_scaling on the localhost line and workers_scaling on the multiprocess line in their respective panels).

    Args:
        envelopes (dict[str, dict[str, Any]]): label-to-envelope mapping. Order is preserved across the legend, Report-table columns, and the per-row offset on the bar panels.
        title (str | None, optional): override the figure suptitle.
        file_path (str | None, optional): when given, saves `<fname>.png` and `<fname>.svg` under this directory.
        fname (str, optional): file stem. Defaults to `"overlay"`.
        verbose (bool, optional): log saved paths when True.
        subtract_floor (bool, optional): subtract `loopback.median_us` from displayed latencies.

    Returns:
        Figure: the matplotlib figure (caller owns its lifecycle).
    """
    _items = list(envelopes.items())
    _fig = plt.figure(figsize=(14, 18))
    _gs = _fig.add_gridspec(4, 2)
    _ax_timer = _fig.add_subplot(_gs[0, 0])
    _ax_jitter = _fig.add_subplot(_gs[0, 1])
    _ax_loopback = _fig.add_subplot(_gs[1, 0])
    _ax_rate = _fig.add_subplot(_gs[1, 1])
    _ax_handler = _fig.add_subplot(_gs[2, 0])
    _ax_workers = _fig.add_subplot(_gs[2, 1])
    _ax_report = _fig.add_subplot(_gs[3, 0:2])

    _palette = _generate_color_map(_items)
    _total = len(_items)
    for _i, (_label, _env) in enumerate(_items):
        _color = _palette[_i]
        _loopback = _env.get("loopback", {})
        _draw_timer(_ax_timer, _env.get("timer", {}),
                    color=_color, label=_label, idx=_i, total=_total)
        _draw_jitter(_ax_jitter, _env.get("jitter", {}),
                     color=_color, label=_label, idx=_i, total=_total)
        _draw_loopback(_ax_loopback, _loopback,
                       color=_color, label=_label, idx=_i, total=_total)
        _draw_rate_sweep(_ax_rate,
                         _env.get("rate", {}),
                         loopback=_loopback,
                         color=_color, label=_label,
                         subtract_floor=subtract_floor)
        _draw_handler_scaling(_ax_handler,
                              _env.get("handler_scaling", {}),
                              loopback=_loopback,
                              color=_color, label=_label,
                              subtract_floor=subtract_floor)
        _draw_workers_scaling(_ax_workers,
                              _env.get("workers_scaling", {}),
                              color=_color, label=_label)
    _draw_report_overlay(_ax_report, envelopes)

    # Legend per panel; skip those with no artists (e.g. an empty workers panel on a
    # localhost-only overlay) to silence matplotlib's "no artists" warnings.
    _has_handler = False
    _has_workers = False
    for _, _env in _items:
        if _env.get("handler_scaling"):
            _has_handler = True
        if _has_workers_data(_env):
            _has_workers = True
    _legend_axes = [_ax_timer, _ax_jitter, _ax_loopback, _ax_rate]
    if _has_handler:
        _legend_axes.append(_ax_handler)
    if _has_workers:
        _legend_axes.append(_ax_workers)
    for _ax in _legend_axes:
        _ax.legend(fontsize=9)

    if title is None:
        _title = "Calibration Overlay"
    else:
        _title = title
    _fig.suptitle(_title, color=_TEXT_BLACK, fontsize=15, fontweight="bold")
    _fig.tight_layout(rect=_TIGHT_RECT)
    _save_figure(_fig, file_path, fname, verbose)
    return _fig


def _has_workers_data(envelope: dict[str, Any]) -> bool:
    """True iff `envelope["workers_scaling"]` carries at least one per-step row.

    Args:
        envelope (dict[str, Any]): one calibration envelope.

    Returns:
        bool: True when the workers ramp has been populated (multiprocess runs); False on localhost or pre-stamp.
    """
    _ws = envelope.get("workers_scaling") or {}
    _per_step = _ws.get("per_step") or []
    return len(_per_step) > 0


def _draw_scaling_slot(ax: Axes,
                       envelope: dict[str, Any],
                       *,
                       loopback: dict[str, Any],
                       subtract_floor: bool) -> None:
    """Draw the bottom-left scaling panel: workers_scaling on multiprocess, handler_scaling otherwise.

    Args:
        ax (Axes): target axis.
        envelope (dict[str, Any]): one calibration envelope.
        loopback (dict[str, Any]): the envelope's loopback block (for floor subtraction in the handler-scaling case).
        subtract_floor (bool): forwarded to `_draw_handler_scaling` when handler_scaling is the chosen panel.
    """
    if _has_workers_data(envelope):
        _draw_workers_scaling(ax, envelope["workers_scaling"])
    else:
        _draw_handler_scaling(ax,
                              envelope.get("handler_scaling", {}),
                              loopback=loopback,
                              subtract_floor=subtract_floor)


# ---- Per-axis drawers (private; shared by single-panel plotters and the grid) ----

def _draw_timer(ax: Axes,
                block: dict[str, Any],
                *,
                title: str | None = None,
                color: Any = _NEUTRAL_BAR,
                label: str | None = None,
                idx: int = 0,
                total: int = 1) -> None:
    """Horizontal bars: Min, Phi, chi-bar, Max in ns; s^2 rendered as +/- error caps on each bar."""
    _labels = ["Min", r"$\Phi$", r"$\overline{\chi}$", "Max"]
    _vals = [block.get("min_ns", 0),
             block.get("median_ns", 0),
             block.get("mean_ns", 0.0),
             block.get("max_ns", 0)]
    _std = float(block.get("std_ns", 0.0))
    _draw_grouped_barh(ax, _labels, _vals, "ns",
                       color=color, label=label, idx=idx, total=total,
                       xerr=_std)
    ax.set_xlabel(r"$[\mathrm{ns}]$  (caps: $s^{2}$)", color=_TEXT_BLACK)
    if title is None:
        _t = f"Timer (n={block.get('samples_n', 0)})"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def _draw_jitter(ax: Axes,
                 block: dict[str, Any],
                 *,
                 title: str | None = None,
                 color: Any = _NEUTRAL_BAR,
                 label: str | None = None,
                 idx: int = 0,
                 total: int = 1) -> None:
    """Horizontal bars: chi-bar, Phi, p_{95}, p_{99} relative to target sleep (us); s^2 rendered as +/- error caps."""
    _target_us = block.get("target_us", 0)
    _labels = [r"$\overline{\chi}$", r"$\Phi$", r"$p_{95}$", r"$p_{99}$"]
    # Show jitter relative to the target so values centre near zero when the host is quiet.
    _vals = [block.get("mean_us", 0.0) - _target_us,
             block.get("median_us", 0.0) - _target_us,
             block.get("p95_us", 0.0) - _target_us,
             block.get("p99_us", 0.0) - _target_us]
    _std = float(block.get("std_us", 0.0))
    _draw_grouped_barh(ax, _labels, _vals, r"$\mu s$",
                       color=color, label=label, idx=idx, total=total,
                       xerr=_std)
    ax.set_xlabel(r"$[\mu s]$ (actual $-$ target; caps: $s^{2}$)", color=_TEXT_BLACK)
    if title is None:
        _t = f"Jitter (target {_target_us / 1000:.1f} ms)"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def _draw_loopback(ax: Axes,
                   block: dict[str, Any],
                   *,
                   title: str | None = None,
                   color: Any = _NEUTRAL_BAR,
                   label: str | None = None,
                   idx: int = 0,
                   total: int = 1) -> None:
    """Horizontal bars: Min, Phi, p_{95}, p_{99} (us); precision band [Phi - s^2, Phi + s^2] shaded behind the bars."""
    _labels = ["Min", r"$\Phi$", r"$p_{95}$", r"$p_{99}$"]
    _phi = float(block.get("median_us", 0.0))
    _std = float(block.get("std_us", 0.0))
    _vals = [block.get("min_us", 0.0),
             _phi,
             block.get("p95_us", 0.0),
             block.get("p99_us", 0.0)]
    # Translucent band centred on the median; only drawn for the first envelope so overlays do not paint twice.
    if idx == 0 and _std > 0.0:
        ax.axvspan(_phi - _std, _phi + _std,
                   color=color, alpha=0.12, zorder=0,
                   label=r"$\Phi \pm s^{2}$")
    _draw_grouped_barh(ax, _labels, _vals, r"$\mu s$",
                       color=color, label=label, idx=idx, total=total,
                       xerr=_std)
    ax.set_xlabel(r"$[\mu s]$  (band + caps: $s^{2}$)", color=_TEXT_BLACK)
    _payload = block.get("payload_bytes", 0)
    if title is None:
        _t = f"Loopback ({_payload:,} B)"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.5)


def _draw_handler_scaling(ax: Axes,
                          block: dict[str, Any],
                          *,
                          loopback: dict[str, Any] | None = None,
                          title: str | None = None,
                          color: Any = _NEUTRAL_BAR,
                          label: str | None = None,
                          subtract_floor: bool = True) -> None:
    """Latency-change vs concurrency, three traces (Phi, p_{95}, p_{99}).

    Linear axes for typical default sweeps (c <= ~32). Log-log makes sense only when concurrency spans multiple decades (e.g. 1..1000); at small ranges it hides the signal. Loopback floor subtracted by default so the panel shows handler work, not kernel noise.
    """
    _stats = block.get("stats", {})
    _cs = sorted(int(_k) for _k in _stats.keys())
    if not _cs:
        ax.set_title(title or "Handler Scaling", color=_TEXT_BLACK, fontweight="bold")
        return
    _floor = _resolve_floor_us(loopback, subtract_floor)
    _phi = [max(_stats[str(_c)].get("median_us", 0.0) - _floor, 0.0) for _c in _cs]
    _p95 = [max(_stats[str(_c)].get("p95_us", 0.0) - _floor, 0.0) for _c in _cs]
    _p99 = [max(_stats[str(_c)].get("p99_us", 0.0) - _floor, 0.0) for _c in _cs]
    if label is None:
        _phi_label, _p95_label, _p99_label = r"$\Phi$", r"$p_{95}$", r"$p_{99}$"
    else:
        _phi_label = f"{label}: $\\Phi$"
        _p95_label = f"{label}: $p_{{95}}$"
        _p99_label = f"{label}: $p_{{99}}$"
    # Translucent fill between p95 and p99 highlights the tail variability as a precision band.
    ax.fill_between(_cs, _p95, _p99, color=color, alpha=0.12,
                    label=r"$p_{95} \rightarrow p_{99}$" if label is None else None)
    ax.plot(_cs, _phi, marker="o", color=_PHI_COLOR,
            linestyle="--", label=_phi_label)
    ax.plot(_cs, _p95, marker="s", color=color,
            linestyle="--", label=_p95_label)
    ax.plot(_cs, _p99, marker="^", color=_P99_COLOR,
            linestyle="--", label=_p99_label)
    # Annotate the highest-c point on the median trace.
    ax.annotate(_fmt_us(_phi[-1]),
                xy=(_cs[-1], _phi[-1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9, color=_TEXT_BLACK)
    # Linear axes; log-log only earns its keep when concurrency spans multiple decades.
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.set_xlabel(r"$c$: concurrent user requests", color=_TEXT_BLACK)
    ax.set_ylabel(r"Latency change $[\mu s]$", color=_TEXT_BLACK)
    if title is None:
        _t = "Handler Scaling"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)


def _draw_rate_sweep(ax: Axes,
                     block: dict[str, Any],
                     *,
                     loopback: dict[str, Any] | None = None,
                     title: str | None = None,
                     color: Any = _NEUTRAL_BAR,
                     label: str | None = None,
                     subtract_floor: bool = True) -> None:
    """Twin-axis: p_{95} latency on left (us), loss percent on right (%). Loopback floor subtracted from latency."""
    _per = block.get("per_rate", [])
    _rates = [_row.get("rate", 0) for _row in _per]
    _floor = _resolve_floor_us(loopback, subtract_floor)
    _p95 = [max(_row.get("p95_us", 0.0) - _floor, 0.0) for _row in _per]
    _loss = [_row.get("loss_pct", 0.0) for _row in _per]
    _sat = block.get("saturation_rate")
    # Verifiable range: rates below saturation are inside the calibrated envelope; shade them.
    if _rates and _sat is not None:
        ax.axvspan(min(_rates), _sat, color="#4CAF50", alpha=0.07, zorder=0)
        ax.axvline(_sat, color="#C44536", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"saturation ({_sat} req/s)")
    if _rates:
        _lat_label = label or r"$p_{95}$ latency"
        ax.plot(_rates, _p95, marker="o", color=color,
                linestyle="-", linewidth=2, label=_lat_label)
    ax.set_xlabel("target rate (req/s)", color=_TEXT_BLACK)
    ax.set_ylabel(r"$p_{95}$ latency $[\mu s]$", color=_TEXT_BLACK)
    if title is None:
        _sat = block.get("saturation_rate")
        if _sat is None:
            _t = "Rate Sweep"
        else:
            _t = f"Rate Sweep (sat. {_sat} req/s)"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)

    # Reuse the twin axis across overlay calls so multiple envelopes share one right y-axis.
    _ax2 = getattr(ax, "_loss_axis", None)
    if _ax2 is None:
        _ax2 = ax.twinx()
        ax._loss_axis = _ax2  # type: ignore[attr-defined]
        _ax2.set_ylabel("loss (%)", color=_TEXT_BLACK)
    if _rates:
        _ax2.plot(_rates, _loss, marker="x", color="#F0AB7E",
                  linestyle="--", alpha=0.85, label="loss %")


def _draw_workers_scaling(ax: Axes,
                          block: dict[str, Any],
                          *,
                          title: str | None = None,
                          color: Any = _NEUTRAL_BAR,
                          label: str | None = None) -> None:
    """Twin-axis: worker rate (req/s) on the left, efficiency percent on the right; vertical marker at the stable knee.

    Args:
        ax (Axes): target axis.
        block (dict[str, Any]): the `workers_scaling` envelope block (`{ramp, per_step, stable_workers, ...}`).
        title (str | None, optional): override title; defaults to "Workers Scaling (stable n=N)" when a knee is known.
        color (Any, optional): primary trace colour.
        label (str | None, optional): legend label for the worker-rate trace.
    """
    _per_step = block.get("per_step") or []
    _ns = [_row.get("n_workers", 0) for _row in _per_step]
    _per_w = [_row.get("per_worker_rps", 0.0) for _row in _per_step]
    _eff = [_row.get("efficiency_pct", 0.0) for _row in _per_step]
    _stable = block.get("stable_workers")

    if _ns and _stable is not None:
        ax.axvspan(min(_ns), _stable, color="#4CAF50", alpha=0.07, zorder=0)
        ax.axvline(_stable, color="#C44536", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"stable (n={_stable})")
    if _ns:
        _lat_label = label or "worker rate"
        ax.plot(_ns, _per_w, marker="o", color=color,
                linestyle="-", linewidth=2, label=_lat_label)
    ax.set_xlabel(r"$w$: concurrent workers", color=_TEXT_BLACK)
    ax.set_ylabel("worker rate (req/s)", color=_TEXT_BLACK)
    if title is None:
        if _stable is None:
            _t = "Workers Scaling"
        else:
            _t = f"Workers Scaling (stable n={_stable})"
    else:
        _t = title
    ax.set_title(_t, color=_TEXT_BLACK, fontweight="bold")
    ax.grid(linestyle=":", alpha=0.5)

    # Reuse the twin axis across overlay calls so multiple envelopes share one right y-axis.
    _ax2 = getattr(ax, "_eff_axis", None)
    if _ax2 is None:
        _ax2 = ax.twinx()
        ax._eff_axis = _ax2  # type: ignore[attr-defined]
        _ax2.set_ylabel("efficiency (%)", color=_TEXT_BLACK)
    if _ns:
        _ax2.plot(_ns, _eff, marker="x", color="#F0AB7E",
                  linestyle="--", alpha=0.85, label="efficiency %")


# Layout constants (axes-fraction units) for the Report panel.
_REPORT_FONTSIZE = 8
_REPORT_LINE_H = 0.034
_REPORT_LABEL_X = 0.02
_REPORT_VALUE_X = 0.36
_REPORT_ROW_LABEL_X = 0.05
_REPORT_LEGEND_BODY_X = 0.12

_REPORT_LEGEND_ROWS = (
    ("Latency:",
     ("Reported figures equal the measured value minus the",
      "loopback floor (median), with the jitter p99 as the",
      "precision band.")),
    ("Floors:",
     ("Background noise sources we cannot control (clock,",
      "scheduler, kernel TCP path); the precision band is",
      "their RMS sum.")),
    ("Envelope:",
     ("Operating limits where the apparatus's measurements",
      "remain trustworthy (concurrency knee + rate",
      "saturation knee).")),
)


def _draw_report(ax: Axes,
                 gate_block: dict[str, Any],
                 *,
                 envelope: dict[str, Any] | None = None) -> None:
    """Text-only single-envelope Report panel: borderless table + static legend.

    The table mirrors the overlay's layout with a single column whose header is the host name. RUN and the allowed noise floor render as the first two rows of the table; the deployment mode is omitted because the figure suptitle already carries it.

    Args:
        ax (Axes): target axis (whole panel).
        gate_block (dict[str, Any]): the `gate` envelope block.
        envelope (dict[str, Any] | None, optional): full envelope for the column header (host) + the metadata rows (run, noise floor).
    """
    ax.axis("off")
    _summary = gate_block.get("summary", {}) or {}
    if envelope is None:
        _host = "?"
    else:
        _host = str(envelope.get("host", "?"))
    _noise = _fmt_noise_floor(gate_block.get("noise_floor_pct"))
    _wrapped = {"gate": gate_block}

    _y = 0.99
    _y = _draw_table_rows(ax,
                          names=[_host],
                          noise_values=[_noise],
                          band_values=[_overlay_band(_wrapped)],
                          c_values=[_overlay_c_max(_wrapped)],
                          r_values=[_overlay_r_max(_wrapped)],
                          w_values=[_overlay_w_max(_wrapped)],
                          summaries=[_summary],
                          y=_y)
    # Divider matches the table column span (label-column left edge to last-column right edge).
    _div_left = _REPORT_LABEL_X
    _div_right = _OVERLAY_COL_START + _OVERLAY_COL_WIDTH
    _draw_legend_rows(ax, _y,
                      div_left=_div_left,
                      div_right=_div_right)
    if envelope is None:
        _dt = "?"
    else:
        _dt = _fmt_run_date(envelope.get("run_id"))
    ax.set_title(f"Calibration Report\n{_host}: {_dt}",
                 color=_NEUTRAL_BAR, fontweight="bold")


def _fmt_noise_floor(value: Any) -> str:
    """Format the gate's noise-floor budget as a `± X %` string.

    Args:
        value (Any): the raw `gate.noise_floor_pct` (number or None).

    Returns:
        str: mathtext-formatted band, or `"n/a"` when the value is missing or non-numeric.
    """
    _ans = "n/a"
    if isinstance(value, (int, float)):
        _ans = rf"$\pm$ {float(value):.1f} %"
    return _ans


def _fmt_run_date(run_id: Any) -> str:
    """Extract the ISO-style timestamp from a run id minted by `make_run_id`.

    Run ids are either `<timestamp>_<nonce>` (no prefix) or `<prefix>_<timestamp>_<nonce>` (with prefix); the timestamp is the only segment matching `YYYYMMDDTHHMMSSZ`. Scans the underscore-split parts for it.

    Args:
        run_id (Any): the envelope's run id.

    Returns:
        str: `YYYY-MM-DD HH:MM:SS` when the timestamp segment is found; otherwise the raw run id (or `"?"` when missing / non-string).
    """
    _ans = "?"
    if isinstance(run_id, str) and run_id:
        _ans = run_id
        _ts: str | None = None
        for _part in run_id.split("_"):
            if _ts is None and _is_run_ts(_part):
                _ts = _part
        if _ts is not None:
            _ans = (f"{_ts[0:4]}-{_ts[4:6]}-{_ts[6:8]}"
                    f" {_ts[9:11]}:{_ts[11:13]}:{_ts[13:15]}")
    return _ans


def _is_run_ts(s: str) -> bool:
    """True iff `s` matches the `YYYYMMDDTHHMMSSZ` shape used by `make_run_id`."""
    return len(s) == 16 and s[8] == "T" and s.endswith("Z")


def _draw_table_rows(ax: Axes,
                     *,
                     names: list[str],
                     noise_values: list[str],
                     band_values: list[str],
                     c_values: list[str],
                     r_values: list[str],
                     w_values: list[str],
                     summaries: list[dict[str, Any]],
                     y: float,
                     x_offset: float = 0.0) -> float:
    """Render the borderless attribute-vs-column table.

    Sections, top to bottom: column header (host names) -> NOISE FLOOR -> PRECISION BAND -> OPERATING RANGE (c / r / workers max) -> FLOORS (timer / jitter / loopback) -> ENVELOPE (scaling / rate / workers). The run-id timestamp lives in the figure suptitle, not in the table.

    Args:
        ax (Axes): target axis.
        names (list[str]): one column header per envelope.
        noise_values (list[str]): per-column noise-floor budget strings.
        band_values (list[str]): per-column precision-band strings.
        c_values (list[str]): per-column `c_max` strings.
        r_values (list[str]): per-column `r_max` strings.
        w_values (list[str]): per-column `w_max` strings.
        summaries (list[dict[str, Any]]): the `gate.summary` block of each envelope.
        y (float): top y-coordinate.

    Returns:
        float: next y-coordinate after the table.
    """
    _y = y
    _y = _put_overlay_columns(ax, names, _y, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    _y = _put_overlay_row(ax, "NOISE FLOOR:", noise_values, _y,
                          indent=False, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    _y = _put_overlay_row(ax, "PRECISION BAND:", band_values, _y,
                          indent=False, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    _y = _put_section(ax, "OPERATING RANGE:", _y, x_offset=x_offset)
    _y = _put_overlay_row(ax, "Concurrency:", c_values, _y,
                          indent=True, x_offset=x_offset)
    _y = _put_overlay_row(ax, "Rate:", r_values, _y,
                          indent=True, x_offset=x_offset)
    _y = _put_overlay_row(ax, "Workers:", w_values, _y,
                          indent=True, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    _y = _put_section(ax, "FLOORS:", _y, x_offset=x_offset)
    for _name, _label in (("timer", "Timer:"), ("jitter", "Jitter:"), ("loopback", "Loopback:")):
        _vals = [_s.get(_name, {}).get("headline", "n/a") for _s in summaries]
        _y = _put_overlay_row(ax, _label, _vals, _y,
                              indent=True, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    _y = _put_section(ax, "ENVELOPE:", _y, x_offset=x_offset)
    for _name, _label in (("scaling", "Scaling:"), ("rate", "Rate sweep:"), ("workers", "Workers:")):
        _vals = [_s.get(_name, {}).get("headline", "n/a") for _s in summaries]
        _y = _put_overlay_row(ax, _label, _vals, _y,
                              indent=True, x_offset=x_offset)
    _y -= _REPORT_LINE_H * 0.3
    return _y


def _draw_legend_rows(ax: Axes,
                      y: float,
                      *,
                      x_offset: float = 0.0,
                      div_left: float | None = None,
                      div_right: float | None = None) -> None:
    """Static three-block legend explaining Latency / Floors / Envelope; bold labels, regular body.

    Args:
        ax (Axes): target axis.
        y (float): top y-coordinate (axes-fraction).
        x_offset (float, optional): horizontal shift applied to legend labels + body; used by the overlay's spanning panel to centre content.
        div_left (float | None, optional): left edge of the divider line above the legend (axes-fraction). Defaults to None (uses 0.11).
        div_right (float | None, optional): right edge of the divider line. Defaults to None (uses 0.89).
    """
    _y = y
    if div_left is None:
        _x_left = 0.11
    else:
        _x_left = div_left
    if div_right is None:
        _x_right = 0.89
    else:
        _x_right = div_right
    ax.plot([_x_left, _x_right], [_y, _y],
            transform=ax.transAxes, color=_TEXT_BLACK,
            linewidth=0.6, clip_on=False)
    _y -= _REPORT_LINE_H
    for _label, _body_lines in _REPORT_LEGEND_ROWS:
        ax.text(_REPORT_LABEL_X + x_offset, _y, _label,
                transform=ax.transAxes, va="top",
                fontsize=_REPORT_FONTSIZE, fontweight="bold",
                color=_TEXT_BLACK, family="monospace")
        for _line in _body_lines:
            ax.text(_REPORT_LEGEND_BODY_X + x_offset, _y, _line,
                    transform=ax.transAxes, va="top",
                    fontsize=_REPORT_FONTSIZE,
                    color=_TEXT_BLACK, family="monospace")
            _y -= _REPORT_LINE_H


def _put_kv(ax: Axes, label: str, value: str, y: float) -> float:
    """Render one bold uppercase header label + regular value on the same row; return the next y."""
    if label:
        ax.text(_REPORT_LABEL_X, y, label,
                transform=ax.transAxes, va="top",
                fontsize=_REPORT_FONTSIZE, fontweight="bold",
                color=_TEXT_BLACK, family="monospace")
    ax.text(_REPORT_VALUE_X, y, value,
            transform=ax.transAxes, va="top",
            fontsize=_REPORT_FONTSIZE,
            color=_TEXT_BLACK, family="monospace")
    return y - _REPORT_LINE_H


def _put_section(ax: Axes, label: str, y: float, *, x_offset: float = 0.0) -> float:
    """Render a bold uppercase section label on its own row; return the next y."""
    ax.text(_REPORT_LABEL_X + x_offset, y, label,
            transform=ax.transAxes, va="top",
            fontsize=_REPORT_FONTSIZE, fontweight="bold",
            color=_TEXT_BLACK, family="monospace")
    return y - _REPORT_LINE_H


def _draw_report_overlay(ax: Axes,
                         envelopes: dict[str, dict[str, Any]]) -> None:
    """Overlay Report panel: borderless table (rows = attributes, columns = envelopes) + legend.

    Each row pairs an attribute label with one value per envelope, all rendered as separate `ax.text()` calls so bold labels coexist with regular values without breaking monospace alignment. The static three-block legend renders below the table.
    """
    ax.axis("off")
    _items = list(envelopes.items())
    _names = [str(_label) for _label, _ in _items]
    _noise_values = [_fmt_noise_floor(_env.get("gate", {}).get("noise_floor_pct"))
                     for _, _env in _items]
    _band_values = [_overlay_band(_env) for _, _env in _items]
    _c_values = [_overlay_c_max(_env) for _, _env in _items]
    _r_values = [_overlay_r_max(_env) for _, _env in _items]
    _w_values = [_overlay_w_max(_env) for _, _env in _items]
    _summaries = [_env.get("gate", {}).get("summary", {}) or {} for _, _env in _items]

    # Centre the table + legend in the spanning panel: estimate the content width
    # (label column + N value columns + a margin), then offset by half the leftover.
    _content_w = _REPORT_LABEL_X + (len(_names) * _OVERLAY_COL_WIDTH) + 0.05
    _x_offset = max(0.0, (1.0 - _content_w) / 2.0 - _REPORT_LABEL_X)

    _y = 0.99
    _y = _draw_table_rows(ax,
                          names=_names,
                          noise_values=_noise_values,
                          band_values=_band_values,
                          c_values=_c_values,
                          r_values=_r_values,
                          w_values=_w_values,
                          summaries=_summaries,
                          y=_y,
                          x_offset=_x_offset)
    # Match the divider line to the table's column span: from the label-column left edge to
    # the right edge of the last value column (after offset).
    _div_left = _REPORT_LABEL_X + _x_offset
    _div_right = (_OVERLAY_COL_START + len(_names) * _OVERLAY_COL_WIDTH) + _x_offset
    _draw_legend_rows(ax, _y,
                      x_offset=_x_offset,
                      div_left=_div_left,
                      div_right=_div_right)
    _segments: list[str] = []
    for _, _env in _items:
        _segments.append(f"{_env.get('host', '?')}: {_fmt_run_date(_env.get('run_id'))}")
    if _segments:
        _subtitle = " vs ".join(_segments)
    else:
        _subtitle = "?"
    ax.set_title(f"Calibration Reports\n{_subtitle}",
                 color=_NEUTRAL_BAR, fontweight="bold")


def _overlay_columns_x(n: int) -> list[float]:
    """Compute the left edge of each value column in the overlay table.

    Uses a fixed per-column width so the table content stays packed near the centre of the panel rather than spreading across the full width (the overlay's Report cell spans two grid columns and would otherwise leave a wide empty gap on the right).

    Args:
        n (int): number of envelope columns.

    Returns:
        list[float]: x positions (axes-fraction); each column is `_OVERLAY_COL_WIDTH` apart, starting at `_OVERLAY_COL_START`.
    """
    _ans: list[float] = []
    _i = 0
    while _i < n:
        _ans.append(_OVERLAY_COL_START + _i * _OVERLAY_COL_WIDTH)
        _i += 1
    return _ans


_OVERLAY_COL_START = 0.12
_OVERLAY_COL_WIDTH = 0.12


def _put_overlay_columns(ax: Axes,
                         names: list[str],
                         y: float,
                         *,
                         x_offset: float = 0.0) -> float:
    """Render the column-header row (uppercase bold envelope names).

    Args:
        ax (Axes): target axis.
        names (list[str]): envelope labels (one per column).
        y (float): top y-coordinate (axes-fraction).
        x_offset (float, optional): horizontal shift applied to every column position; used by the overlay's spanning panel to centre content.

    Returns:
        float: next y-coordinate after the row.
    """
    _xs = _overlay_columns_x(len(names))
    for _x, _name in zip(_xs, names, strict=True):
        ax.text(_x + x_offset, y, _name.upper(),
                transform=ax.transAxes, va="top",
                fontsize=_REPORT_FONTSIZE, fontweight="bold",
                color=_TEXT_BLACK, family="monospace")
    return y - _REPORT_LINE_H


def _put_overlay_row(ax: Axes,
                     label: str,
                     values: list[str],
                     y: float,
                     *,
                     indent: bool,
                     x_offset: float = 0.0) -> float:
    """Render one table row: bold label + one value per envelope column.

    Args:
        ax (Axes): target axis.
        label (str): row label (rendered bold).
        values (list[str]): per-envelope value strings; length must match the column count.
        y (float): top y-coordinate.
        indent (bool): True for nested rows under a section header (uses `_REPORT_ROW_LABEL_X`).
        x_offset (float, optional): horizontal shift applied to label + every column; used by the overlay's spanning panel to centre content.

    Returns:
        float: next y-coordinate after the row.
    """
    if indent:
        _x_label = _REPORT_ROW_LABEL_X
    else:
        _x_label = _REPORT_LABEL_X
    ax.text(_x_label + x_offset, y, label,
            transform=ax.transAxes, va="top",
            fontsize=_REPORT_FONTSIZE, fontweight="bold",
            color=_TEXT_BLACK, family="monospace")
    _xs = _overlay_columns_x(len(values))
    for _x, _val in zip(_xs, values, strict=True):
        ax.text(_x + x_offset, y, _val,
                transform=ax.transAxes, va="top",
                fontsize=_REPORT_FONTSIZE,
                color=_TEXT_BLACK, family="monospace")
    return y - _REPORT_LINE_H


def _overlay_band(env: dict[str, Any]) -> str:
    """Format the precision band for an overlay table cell.

    Args:
        env (dict[str, Any]): one envelope.

    Returns:
        str: `$\\pm$ X.XX $\\mu$s`, or `n/a` when the band is missing.
    """
    _band = (env.get("gate", {}).get("precision_band_us") or {}).get("total_us")
    _ans = "n/a"
    if _band is not None:
        _ans = rf"$\pm$ {float(_band):.2f} $\mu$s"
    return _ans


def _overlay_c_max(env: dict[str, Any]) -> str:
    """Format `c_max` as `$c \\leq N$` for an overlay table cell.

    Args:
        env (dict[str, Any]): one envelope.

    Returns:
        str: mathtext expression, or `n/a` when c_max is missing.
    """
    _c = env.get("gate", {}).get("verifiable_range", {}).get("c_max")
    _ans = "n/a"
    if _c is not None:
        _ans = rf"$c \leq {int(_c)}$"
    return _ans


def _overlay_r_max(env: dict[str, Any]) -> str:
    """Format `r_max` as `$r \\leq N$ req/s` for an overlay table cell.

    Args:
        env (dict[str, Any]): one envelope.

    Returns:
        str: mathtext expression, or `n/a` when r_max is missing.
    """
    _r = env.get("gate", {}).get("verifiable_range", {}).get("r_max_req_s")
    _ans = "n/a"
    if _r is not None:
        _ans = rf"$r \leq {int(_r)}$ req/s"
    return _ans


def _overlay_w_max(env: dict[str, Any]) -> str:
    """Format `w_max` as `$w \\leq N$` for an overlay table cell.

    Args:
        env (dict[str, Any]): one envelope.

    Returns:
        str: mathtext expression, or `n/a` when w_max is missing (single-worker / localhost).
    """
    _w = env.get("gate", {}).get("verifiable_range", {}).get("w_max")
    _ans = "n/a"
    if _w is not None:
        _ans = rf"$w \leq {int(_w)}$"
    return _ans


# ---- Internal helpers ----

def _resolve_floor_us(loopback: dict[str, Any] | None,
                      subtract_floor: bool) -> float:
    """Return the loopback median (us) to subtract from displayed latency, or 0.0 when subtraction is disabled."""
    if not subtract_floor or loopback is None:
        return 0.0
    return float(loopback.get("median_us", 0.0))


def _draw_grouped_barh(ax: Axes,
                       labels: list[str],
                       values: list[float],
                       unit: str,
                       *,
                       color: str,
                       label: str | None,
                       idx: int,
                       total: int,
                       xerr: float = 0.0) -> None:
    """Draw one envelope's row of horizontal bars at offset `idx` of `total`.

    Single-envelope panels pass `idx=0, total=1` and get the original layout. Overlay panels pass `idx=0..N-1, total=N` so each label slot is split into N thinner sub-bars stacked vertically; this avoids value-label overlap when two envelopes share the same row.

    `xerr` (>= 0) renders symmetric +/- caps on every bar (in the same units as `values`); it carries the probe's standard deviation so the precision is visible without spending a separate row on `s^2`.
    """
    _n = len(labels)
    _height = 0.8 / total
    _offset = (idx - (total - 1) / 2.0) * _height
    _ys = [_i + _offset for _i in range(_n)]
    if xerr > 0.0:
        _err_kw: dict[str, Any] = {
            "xerr": xerr,
            "error_kw": {"ecolor": _TEXT_BLACK, "capsize": 3, "elinewidth": 1.0},
        }
    else:
        _err_kw = {}
    _bars = ax.barh(_ys, values, height=_height, color=color, alpha=0.85,
                    label=label, **_err_kw)
    for _bar, _val in zip(_bars, values, strict=True):
        ax.text(_bar.get_width() + xerr,
                _bar.get_y() + _bar.get_height() / 2,
                f"  {_fmt_value(_val)} {unit}",
                va="center", ha="left",
                fontsize=9, color=_TEXT_BLACK)
    if idx == 0:
        ax.set_yticks(list(range(_n)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()


def _fmt_value(value: float) -> str:
    """Format a numeric value with thousands separators and one decimal place when fractional."""
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value)):,}"
    return f"{value:,.1f}"


def _fmt_us(value: float) -> str:
    """Format a microsecond value as `<n> us` (mathtext mu)."""
    return f"{_fmt_value(value)} $\\mu s$"


__all__ = [
    "plot_calibration_summary",
    "plot_envelope_overlay",
    "plot_handler_scaling",
    "plot_jitter",
    "plot_loopback",
    "plot_rate_sweep",
    "plot_timer",
    "plot_workers_scaling",
]
