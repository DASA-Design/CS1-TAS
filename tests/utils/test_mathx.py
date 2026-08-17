# -*- coding: utf-8 -*-
"""
Module test_mathx.py
====================

Sanity checks for the math helpers in `src.utils.mathx`.

Each class groups tests by the contract under verification:

    - **TestGFactorial**: numerical correctness of the generalised factorial across integers and halves (which dispatches to the gamma branch).
"""
# native python modules
import math

# testing framework
import pytest

# module under test
from src.utils.mathx import gfactorial


class TestGFactorial:
    """**TestGFactorial** covers `gfactorial()` across integers, zero, and the half-integer case (which dispatches to the gamma branch)."""

    def test_zero(self):
        """*test_zero()* 0! must equal 1 by convention."""
        assert gfactorial(0) == 1

    def test_small_int(self):
        """*test_small_int()* 5! = 120 (standard integer branch)."""
        assert gfactorial(5) == 120

    def test_half(self):
        """*test_half()* gfactorial(0.5) = Γ(1.5) = 0.5 * sqrt(π)."""
        _expected = 0.5 * math.sqrt(math.pi)
        assert gfactorial(0.5) == pytest.approx(_expected)
