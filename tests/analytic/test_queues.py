# -*- coding: utf-8 -*-
"""
Module test_queues.py
=====================

Sanity checks for the closed-form queue models in `src.analytic.queues`.

    - **TestMM1** M/M/1 closed-form formulas and guard-rails.
    - **TestMMcK** M/M/c/K formulas, M/M/s/K alias, and capacity-vs-server-count invariants.
    - **TestFactoryErrors** registry-level errors from the `Queue()` factory when the model string is not supported.
"""
# testing framework
import pytest

# module under test
from src.analytic.queues import Queue


class TestMM1:
    """**TestMM1** the M/M/1 closed form at the textbook point `lamb=1, mu=2` -> `rho=0.5, L=1, Lq=0.5, W=1, Wq=0.5`."""

    def test_textbook(self) -> None:
        """*test_textbook()* `rho == 0.5, L == 1.0, Lq == 0.5, W == 1.0, Wq == 0.5` at `lamb=1, mu=2`."""
        _q = Queue("M/M/1", lamb=1.0, mu=2.0)
        _q.calculate_metrics()
        assert _q.rho == pytest.approx(0.5)
        assert _q.avg_len == pytest.approx(1.0)
        assert _q.avg_len_q == pytest.approx(0.5)
        assert _q.avg_wait == pytest.approx(1.0)
        assert _q.avg_wait_q == pytest.approx(0.5)

    def test_unstable_raises(self) -> None:
        """*test_unstable_raises()* `lamb > mu` (rho >= 1) raises `ValueError` matching `"unstable"`."""
        with pytest.raises(ValueError, match="unstable"):
            Queue("M/M/1", lamb=2.0, mu=1.0)

    def test_rejects_finite_capacity(self) -> None:
        """*test_rejects_finite_capacity()* passing `K_max` to an M/M/1 raises `ValueError` matching `"infinite"`."""
        with pytest.raises(ValueError, match="infinite"):
            Queue("M/M/1", lamb=1.0, mu=2.0, K_max=10)


class TestMMcK:
    """**TestMMcK** M/M/c/K closed form and the shape invariants (K >= c, factory aliasing, probability mass)."""

    def test_basic_rho(self) -> None:
        """*test_basic_rho()* `rho == 0.5` and `0 < avg_len < 2` at `c=1, K=2, lamb=1, mu=2`."""
        _q = Queue("M/M/c/K", lamb=1.0, mu=2.0, c_max=1, K_max=2)
        _q.calculate_metrics()
        assert _q.rho == pytest.approx(0.5)
        assert 0 < _q.avg_len < 2

    def test_slash_s_slash_k_alias(self) -> None:
        """*test_slash_s_slash_k_alias()* `Queue("M/M/s/K", ...)` and `Queue("M/M/c/K", ...)` produce instances of the same class."""
        _q_alias = Queue("M/M/s/K", lamb=1.0, mu=2.0, c_max=1, K_max=2)
        _q_canon = Queue("M/M/c/K", lamb=1.0, mu=2.0, c_max=1, K_max=2)
        assert type(_q_alias).__name__ == type(_q_canon).__name__

    def test_rejects_k_below_c(self) -> None:
        """*test_rejects_k_below_c()* `K_max < c_max` raises `ValueError` matching `"K >= c"`."""
        with pytest.raises(ValueError, match="K >= c"):
            Queue("M/M/c/K", lamb=1.0, mu=2.0, c_max=3, K_max=2)

    def test_probs_sum_to_one(self) -> None:
        """*test_probs_sum_to_one()* `sum(P(n) for n in range(K_max + 1)) == 1.0` (state distribution integrates to one over the reachable range)."""
        _q = Queue("M/M/c/K", lamb=1.0, mu=2.0, c_max=2, K_max=5)
        _q.calculate_metrics()
        # M/M/c/K always has finite K; the cast narrows Optional[int] -> int for the type checker
        _k_max = int(_q.K_max or 0)
        _total = sum(_q.calculate_prob_n(n) for n in range(_k_max + 1))
        assert _total == pytest.approx(1.0, abs=1e-9)


class TestFactoryErrors:
    """**TestFactoryErrors** registry-level errors that `Queue()` raises before any class is instantiated."""

    def test_unknown_model(self) -> None:
        """*test_unknown_model()* an unknown model string raises `NotImplementedError` matching `"Unsupported queue model"`."""
        with pytest.raises(NotImplementedError,
                           match="Unsupported queue model"):
            Queue("G/G/1", lamb=1.0, mu=2.0)
