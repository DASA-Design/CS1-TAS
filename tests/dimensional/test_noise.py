# -*- coding: utf-8 -*-
"""
Module test_noise.py
====================

Seeded multiplicative white-noise primitive used by the dimensional method.

    - **TestAddWhiteNoise**: scalar vs per-element levels, selective-element masking, seeded reproducibility, passthrough on non-positive sigma, shape preservation, empirical std.
"""
# scientific stack
import numpy as np

# module under test
from src.dimensional import add_white_noise


class TestAddWhiteNoise:
    """**TestAddWhiteNoise** element-wise `vals * (1 + N(0, sigma))` with scalar or array sigma; zero sigma passes through; draws come from the supplied seeded generator."""

    def test_scalar_noise_disturbs_all(self) -> None:
        """*test_scalar_noise_disturbs_all()* a float noise level perturbs every element away from its input."""
        _vals = np.array([1.0, 200.0, 0.3])
        _out = add_white_noise(_vals, 0.05, rng=np.random.default_rng(0))
        assert _out.shape == _vals.shape
        assert not np.allclose(_out, _vals)

    def test_array_noise_selective(self) -> None:
        """*test_array_noise_selective()* a `[0, 0.05, 0]` level perturbs only the index-1 value and leaves the rest exact."""
        _vals = np.array([1.0, 200.0, 0.3])
        _out = add_white_noise(_vals, [0.0, 0.05, 0.0], rng=np.random.default_rng(1))
        assert _out[0] == _vals[0]
        assert _out[2] == _vals[2]
        assert _out[1] != _vals[1]

    def test_zero_noise_passthrough(self) -> None:
        """*test_zero_noise_passthrough()* a non-positive level returns the input unchanged."""
        _vals = np.array([3.0, 4.0, 5.0])
        _out = add_white_noise(_vals, 0.0, rng=np.random.default_rng(2))
        assert np.array_equal(_out, _vals)

    def test_negative_noise_clamped(self) -> None:
        """*test_negative_noise_clamped()* a negative level is clamped to zero (no perturbation, no error)."""
        _vals = np.array([3.0, 4.0])
        _out = add_white_noise(_vals, -0.1, rng=np.random.default_rng(3))
        assert np.array_equal(_out, _vals)

    def test_seeded_reproducible(self) -> None:
        """*test_seeded_reproducible()* two calls with same-seed generators produce identical output."""
        _vals = np.array([10.0, 20.0, 30.0])
        _a = add_white_noise(_vals, 0.05, rng=np.random.default_rng(42))
        _b = add_white_noise(_vals, 0.05, rng=np.random.default_rng(42))
        assert np.array_equal(_a, _b)

    def test_scalar_input_returns_float(self) -> None:
        """*test_scalar_input_returns_float()* a scalar value returns a Python float, not a 0-d array."""
        _out = add_white_noise(5.0, 0.05, rng=np.random.default_rng(4))
        assert isinstance(_out, float)

    def test_empirical_std(self) -> None:
        """*test_empirical_std()* the multiplicative factor's sample std over a large draw is close to the requested level."""
        _vals = np.ones(50000)
        _out = add_white_noise(_vals, 0.05, rng=np.random.default_rng(5))
        _factor = _out / _vals
        assert abs(float(_factor.std()) - 0.05) < 0.005
