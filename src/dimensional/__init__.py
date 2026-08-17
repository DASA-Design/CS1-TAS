# -*- coding: utf-8 -*-
"""TAS dimensional-method adapters around PyDASA.

This subpackage is intentionally thin: PyDASA does the math, these modules provide the case-study glue (config-driven FDU schema, per-artifact engine construction, coefficient derivation from spec, sensitivity reshaping).
"""

from src.dimensional.coefficients import derive_coefs
from src.dimensional.engine import build_engine
from src.dimensional.noise import add_white_noise
from src.dimensional.networks import (sweep_arch,
                                      sweep_artifact,
                                      sweep_artifacts)
from src.dimensional.reshape import (aggregate_arch_coefs,
                                     aggregate_sweep_to_arch,
                                     compute_coefs_delta,
                                     coefs_to_net,
                                     coefs_to_nodes,
                                     compute_net_delta,
                                     load_dim_op_points,
                                     load_dim_op_points_per_node)
from src.dimensional.routing import build_routing
from src.dimensional.schema import build_schema
from src.dimensional.sensitivity import analyse_symbolic

__all__ = [
    "add_white_noise",
    "aggregate_arch_coefs",
    "aggregate_sweep_to_arch",
    "analyse_symbolic",
    "build_engine",
    "build_routing",
    "build_schema",
    "compute_coefs_delta",
    "coefs_to_net",
    "coefs_to_nodes",
    "derive_coefs",
    "compute_net_delta",
    "load_dim_op_points",
    "load_dim_op_points_per_node",
    "sweep_arch",
    "sweep_artifact",
    "sweep_artifacts",
]
