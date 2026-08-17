# -*- coding: utf-8 -*-
"""
Module mathx.py
===============

Math helpers shared across analytic, stochastic, and dimensional methods. Kept in a single utility module so mathematical primitives are not inlined into every consumer.

# TODO: extend with other generalised special functions as methods grow.
"""

# python native modules
import math

# data types
from typing import Optional, Union


def gfactorial(x: Union[int, float],
               prec: Optional[int] = None) -> Union[int, float]:
    """*gfactorial()* calculates the factorial of a number, including support for floats less than 1.0.

        - For integers n ≥ 0: Returns n! (n factorial).
        - For floats x: Returns Γ(x+1) (gamma function).

    Args:
        x (Union[int, float]): The number to compute the factorial for.
        prec (Optional[int], optional): precision, or the number of decimal places to round the result to. Defaults to None.

    Raises:
        ValueError: If x is a negative integer.

    Returns:
        Union[int, float]: The factorial of x. Returns an integer for integer inputs ≥ 0, and a float for float inputs or integers < 0.

    Examples:
        >>> gfactorial(5)
        120
        >>> gfactorial(0)
        1
        >>> gfactorial(0.5)  # Equivalent to Γ(1.5) = 0.5 * Γ(0.5) = 0.5 * √π
        0.8862269254527579
        >>> gfactorial(-0.5)  # Equivalent to Γ(0.5) = √π
        1.7724538509055159
    """
    if isinstance(x, int) and x >= 0:
        # Standard factorial for non-negative integers
        _result = math.factorial(x)
    elif isinstance(x, int) and x < 0:
        # Factorial is not defined for negative integers
        raise ValueError("Factorial is not defined for negative integers")
    else:
        # For floats, use the gamma function: Γ(x+1)
        _result = math.gamma(x + 1)

    if prec is not None:
        _result = round(_result, prec)

    return _result
