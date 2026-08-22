# -*- coding: utf-8 -*-
"""Plotting helpers for the CS-01 TAS case study.

Module split:

- `common.py`: shared design-contract primitives + family-private helpers.
- `charter.py`: yoly coefficient charts (theta, sigma, eta, phi).
- `diagrams.py`: queueing topology + per-node heatmaps + architecture bars.
"""

# shared design-contract primitives + project-wide constants
from src.view.common import (
    AxisSpec,
    BodySpec,
    DIM_GLOSSARY_DEFAULT,
    FigureLayout,
    QN_GLOSSARY_DEFAULT,
    attach_axis_spec,
    build_stacked_figure,
    render_footer_legend,
    render_footer_summary,
    render_footer_table,
)
# yoly family
from src.view.charter import (
    plot_selection_surface,
    plot_yoly_arts_behaviour,
    plot_yoly_arts_charts,
    plot_yoly_arts_hist,
    plot_yoly_arts_with_op_points,
    plot_yoly_chart,
    plot_yoly_space,
    plot_yoly_with_op_points,
)
# topology + heatmap + bars + CI family
from src.view.diagrams import (
    plot_arch_bars,
    plot_arch_delta,
    plot_dim_topology,
    plot_method_metric_bars,
    plot_node_ci,
    plot_node_diffmap,
    plot_node_heatmap,
    plot_pareto_front,
    plot_qn_topology,
    plot_verdict_grid,
)

__all__ = [
    # design-contract primitives
    "AxisSpec",
    "BodySpec",
    "FigureLayout",
    "attach_axis_spec",
    "build_stacked_figure",
    "render_footer_legend",
    "render_footer_summary",
    "render_footer_table",
    # public defaults
    "DIM_GLOSSARY_DEFAULT",
    "QN_GLOSSARY_DEFAULT",
    # public plotters (family-prefixed)
    "plot_arch_bars",
    "plot_arch_delta",
    "plot_dim_topology",
    "plot_method_metric_bars",
    "plot_node_ci",
    "plot_node_diffmap",
    "plot_node_heatmap",
    "plot_pareto_front",
    "plot_qn_topology",
    "plot_selection_surface",
    "plot_verdict_grid",
    "plot_yoly_arts_behaviour",
    "plot_yoly_arts_charts",
    "plot_yoly_arts_hist",
    "plot_yoly_arts_with_op_points",
    "plot_yoly_chart",
    "plot_yoly_space",
    "plot_yoly_with_op_points",
]
