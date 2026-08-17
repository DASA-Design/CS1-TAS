# -*- coding: utf-8 -*-
"""
Module noise.py
===============

Seeded multiplicative white-noise perturbation for the dimensional method. The DASA solution and the yoly sweep treat the M/M/c/K *independent* variables (lambda, mu, epsilon) as noisy observations and recompute the dependent queue metrics from the perturbed inputs; this module supplies the single primitive that applies that noise.

Public API:
    - `add_white_noise(vals, noise, rng)` element-wise multiplicative Gaussian noise on an array of values, with a scalar or per-element noise level.

The helper is deliberately variable-agnostic: callers assemble the array of values they want disturbed and the matching noise spec. Reproducibility is the caller's responsibility via the supplied `rng` (built from `data/config/method/dimensional.json::seed`), so there is no hidden `default_rng`.
"""
# native python modules
from __future__ import annotations

# data types
from typing import Union

# scientific stack
import numpy as np
from numpy.typing import ArrayLike


def add_white_noise(vals: ArrayLike,
                    noise: ArrayLike = 0.05,
                    *,
                    rng: np.random.Generator) -> Union[float, np.ndarray]:
    """*add_white_noise()* apply seeded zero-mean multiplicative Gaussian noise to an array of values.

    Returns `vals * (1 + N(0, sigma))` element-wise, where `sigma` is the noise level. `noise` may be a single float (every element disturbed by the same relative amount) or an array broadcastable to `vals` (per-element amounts). A non-positive `sigma` leaves that element untouched, so `add_white_noise([1, 200, 0.3], [0, 0.05, 0], rng=rng)` perturbs only the index-1 value by about 5 %.

    Args:
        vals (ArrayLike): values to perturb; scalar or any array-like shape.
        noise (ArrayLike): relative noise level (standard deviation of the multiplicative factor); float for a uniform disturbance or an array broadcastable to `vals` for per-element levels. Defaults to 0.05 (5 %).
        rng (np.random.Generator): seeded generator supplying the draws; required so results are reproducible.

    Returns:
        Union[float, np.ndarray]: perturbed values, same shape as `vals`; a scalar input returns a float.
    """
    _vals = np.asarray(vals, dtype=float)
    _sig = np.broadcast_to(np.asarray(noise,
                                      dtype=float),
                           _vals.shape)
    # rng.normal needs scale >= 0; a zero (or clamped negative) sigma yields a deterministic 0 draw -> element passes through
    _sig = np.where(_sig > 0.0, _sig, 0.0)
    _out = _vals * (1.0 + rng.normal(0.0, _sig))
    if _out.ndim == 0:
        return float(_out)
    return _out
