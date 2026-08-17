# -*- coding: utf-8 -*-
"""
Module queues.py
================

Closed-form queue models used by the analytic method for the TAS case study. Field names follow the PyDASA acronym convention:

    - `lamb`: arrival rate lambda (avoids Python's reserved word).
    - `c_max`: number of servers c.
    - `K_max`: system capacity K; `None` means infinite.

Public API:
    - `BasicQueue` abstract dataclass with the shared metric surface.
    - `QueueMM1` / `QueueMMs` / `QueueMM1K` / `QueueMMsK` concrete models.
    - `Queue(model, lamb, mu, c_max, K_max)` factory keyed by model string.

Formulas are drawn from standard queueing theory:

    - Kleinrock, L. (1975), *Queueing Systems, Vol. 1: Theory*.
    - Gross, D. et al. (2008), *Fundamentals of Queueing Theory*, 4th ed.
"""
# native python modules
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# local modules
from src.utils.mathx import gfactorial


@dataclass
class BasicQueue(ABC):
    """**BasicQueue** abstract base class for queueing theory models.

    Attributes:
        lamb (float): arrival rate (lambda).
        mu (float): service rate (mu).
        c_max (int): number of servers (c).
        K_max (Optional[int]): maximum capacity (K), `None` for unbounded.
        rho (float): server utilisation (rho), set by `calculate_metrics()`.
        tau (float): traffic intensity (tau = lamb / mu).
        p_z (float): probability of zero requests in the system, P(0).
        p_n (float): probability of n requests in the system, P(n).
        avg_len (float): L, mean number of requests in the system.
        avg_len_q (float): Lq, mean number of requests in the queue.
        avg_wait (float): W, mean time a request spends in the system.
        avg_wait_q (float): Wq, mean waiting time in the queue.
        lamb_eff (float): effective arrival rate for finite-K models.
    """
    lamb: float = -1.0

    mu: float = -1.0

    c_max: int = 1

    K_max: Optional[int] = None

    rho: float = field(default=0.0, init=False)

    tau: float = field(default=0.0, init=False)

    p_z: float = field(default=0.0, init=False)

    p_n: float = field(default=0.0, init=False)

    avg_len: float = field(default=0.0, init=False)

    avg_len_q: float = field(default=0.0, init=False)

    avg_wait: float = field(default=0.0, init=False)

    avg_wait_q: float = field(default=0.0, init=False)

    # effective arrival rate `lamb * (1 - P(K))`; only set on finite-K models
    lamb_eff: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """*__post_init__()* coerce numeric types and run common plus model-specific validation.

        Raises:
            ValueError: when any input parameter violates the common or model-specific constraints.
        """
        self.c_max = int(self.c_max)
        if self.K_max is not None:
            self.K_max = int(self.K_max)
        self._validate_basic_params()
        self._validate_params()

    def _validate_basic_params(self) -> None:
        """*_validate_basic_params()* check the parameters common to every queueing model.

        Raises:
            ValueError: if arrival rate is negative, service rate is non-positive, or server count is non-positive.
        """
        if self.lamb < 0:
            raise ValueError("Arrival rate must be non-negative.")
        if self.mu <= 0:
            raise ValueError("Service rate must be positive.")
        if self.c_max < 1:
            raise ValueError("Number of servers must be positive.")

    @abstractmethod
    def _validate_params(self) -> None:
        """*_validate_params()* check the parameters specific to the concrete model.

        Raises:
            ValueError: when the concrete model's invariants are violated.
        """
        pass

    @abstractmethod
    def calculate_metrics(self) -> None:
        """*calculate_metrics()* compute the analytical metrics and write them in place.

        Side effects: sets `rho`, `tau`, `p_z`, `avg_len`, `avg_len_q`, `avg_wait`, `avg_wait_q`, and for finite-K models `lamb_eff`.
        """
        pass

    @abstractmethod
    def calculate_prob_zero(self) -> float:
        """*calculate_prob_zero()* return P(0), the probability of an empty system.

        Returns:
            float: probability of having 0 requests in the system.
        """
        pass

    @abstractmethod
    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* return P(n), the probability of n requests in the system.

        Args:
            n (int): number of requests in the system.

        Returns:
            float: probability of having n requests in the system.
        """
        pass

    @abstractmethod
    def is_stable(self) -> bool:
        """*is_stable()* report whether the system admits a steady-state solution.

        Returns:
            bool: True if stable, False otherwise.
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """*get_metrics()* return the summary of computed performance metrics.

        Returns:
            Dict[str, Any]: dictionary with keys `L`, `Lq`, `W`, `Wq`, `rho`.
        """
        return {
            "L": self.avg_len,
            "Lq": self.avg_len_q,
            "W": self.avg_wait,
            "Wq": self.avg_wait_q,
            "rho": self.rho,
        }

    def __str__(self) -> str:
        """*__str__()* return a formatted multiline summary of the queue."""
        _output = [f"{self.__class__.__name__}("]

        _params = [
            f"\tlamb={self.lamb}",
            f"\tmu={self.mu}",
            f"\tc_max={self.c_max}",
        ]
        _output.extend(_params)
        if self.K_max is not None:
            _output.append(f"\tK_max={self.K_max}")

        if self.is_stable():
            _status_word = "STABLE"
        else:
            _status_word = "UNSTABLE"
        _status = f"\tStatus: {_status_word}"
        _output.append(_status)

        _metrics = self.get_metrics()
        for _key, _value in _metrics.items():
            if isinstance(_value, float):
                _output.append(f"\t{_key}={_value:.6f}")
            else:
                _output.append(f"\t{_key}={_value}")
        _output.append(")")
        return ",\n".join(_output)

    def __repr__(self) -> str:
        """*__repr__()* return the same formatted summary as `__str__`."""
        return self.__str__()


@dataclass
class QueueMM1(BasicQueue):
    """**QueueMM1** M/M/1 queueing model (1 server, infinite capacity).

    Raises:
        ValueError: if server count is not 1, capacity is not infinite, or the system is unstable (`lamb >= mu`).
    """

    def _validate_params(self) -> None:
        """*_validate_params()* check the M/M/1 invariants.

        Raises:
            ValueError: if server count is not 1, capacity is not infinite, or the system is unstable.
        """
        if self.c_max != 1:
            _msg = f"M/M/1 requires exactly 1 server. c={self.c_max}"
            raise ValueError(_msg)
        if self.K_max is not None:
            _msg = f"M/M/1 assumes infinite capacity. K={self.K_max}"
            raise ValueError(_msg)
        if not self.is_stable():
            _msg = "System is unstable (lamb >= mu). "
            _msg += f"lamb={self.lamb}, mu={self.mu}"
            raise ValueError(_msg)

    def is_stable(self) -> bool:
        """*is_stable()* return True when `lamb / mu < 1`."""
        return self.lamb / self.mu < 1.0

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* compute rho, L, Lq, W, Wq for the M/M/1 model.

        Side effects: sets `rho`, `tau`, `p_z`, `avg_len`, `avg_len_q`, `avg_wait`, `avg_wait_q` on the instance.
        """
        # utilisation rho = lamb / mu (single-server form); traffic intensity tau coincides at c=1
        self.rho = self.lamb / self.mu
        self.tau = self.lamb / self.mu
        self.p_z = self.calculate_prob_zero()
        # closed-form M/M/1: L = rho / (1 - rho), Lq = rho^2 / (1 - rho)
        self.avg_len = self.rho / (1 - self.rho)
        self.avg_len_q = self.rho ** 2 / (1 - self.rho)
        # Little's Law (W = L / lamb, Wq = Lq / lamb); lamb=0 short-circuits to avoid 0 / 0
        if self.lamb > 0:
            self.avg_wait = self.avg_len / self.lamb
            self.avg_wait_q = self.avg_len_q / self.lamb
        else:
            self.avg_wait = 0.0
            self.avg_wait_q = 0.0

    def calculate_prob_zero(self) -> float:
        """*calculate_prob_zero()* return P(0) for the M/M/1 model.

        Returns:
            float: probability of 0 requests in the system, `1 - rho`.
        """
        return 1.0 - self.rho

    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* return P(n) for the M/M/1 model.

        Args:
            n (int): number of requests in the system.

        Returns:
            float: probability of n requests; `-1.0` when n is negative.
        """
        _p_n = 0.0
        if n < 0:
            _p_n = -1.0
        elif n >= 0:
            _p_n = (1 - self.rho) * (self.rho ** n)
        self.p_n = _p_n
        return _p_n


@dataclass
class QueueMMs(BasicQueue):
    """**QueueMMs** M/M/s queueing model (multi-server, infinite capacity).

    Raises:
        ValueError: if server count is less than 1, capacity is not infinite, or the system is unstable (`lamb >= c * mu`).
    """

    def _validate_params(self) -> None:
        """*_validate_params()* check the M/M/s invariants.

        Raises:
            ValueError: if server count is less than 1, capacity is not infinite, or the system is unstable.
        """
        if self.c_max < 1:
            _msg = f"M/M/s requires at least one server. c={self.c_max}"
            raise ValueError(_msg)
        if self.K_max is not None:
            _msg = f"M/M/s assumes infinite capacity. K={self.K_max}"
            raise ValueError(_msg)
        if not self.is_stable():
            _msg = "System is unstable (lamb >= c * mu). "
            _msg += f"lamb={self.lamb}, c={self.c_max}, mu={self.mu}"
            raise ValueError(_msg)

    def is_stable(self) -> bool:
        """*is_stable()* return True when `lamb / (c * mu) < 1`."""
        return self.lamb / (self.c_max * self.mu) < 1.0

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* compute rho, L, Lq, W, Wq via the Erlang-C formulas.

        Side effects: sets `rho`, `tau`, `p_z`, `avg_len`, `avg_len_q`, `avg_wait`, `avg_wait_q` on the instance.
        """
        # rho = lamb / (c * mu); tau = lamb / mu (offered load before sharing across c servers)
        self.rho = self.lamb / (self.c_max * self.mu)
        self.tau = self.lamb / self.mu
        self.p_z = self.calculate_prob_zero()
        # Erlang-C: Lq = P0 * tau^c * rho / (c! * (1 - rho)^2)
        _num = self.p_z * (self.tau ** self.c_max) * self.rho
        _den = gfactorial(self.c_max) * ((1 - self.rho) ** 2)
        self.avg_len_q = _num / _den
        # L = Lq + tau (mean in service equals offered load tau)
        self.avg_len = self.avg_len_q + self.tau
        # Wq via Little's Law; lamb=0 short-circuits to avoid 0 / 0
        if self.lamb > 0:
            self.avg_wait_q = self.avg_len_q / self.lamb
        else:
            self.avg_wait_q = 0.0
        # W = Wq + 1/mu (queue wait plus service time)
        self.avg_wait = self.avg_wait_q + self.mu ** -1

    def calculate_prob_zero(self) -> float:
        """*calculate_prob_zero()* return P(0) for the M/M/s model via the Erlang-C denominator.

        Returns:
            float: probability of 0 requests in the system.
        """
        # mass under the c saturation point
        _p_under_c = sum((self.tau ** i) / gfactorial(i)
                         for i in range(self.c_max))
        # tail mass at and above c
        _num = self.tau ** self.c_max
        _den = gfactorial(self.c_max) * (1 - self.rho)
        _p_over_c = _num / _den
        return (_p_under_c + _p_over_c) ** -1

    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* return P(n) for the M/M/s model.

        Args:
            n (int): number of requests in the system.

        Returns:
            float: probability of n requests; `-1.0` when n is negative.
        """
        if n < 0:
            self.p_n = -1.0
            return -1.0
        _num = self.tau ** n
        if n <= self.c_max:
            _den = gfactorial(n)
        else:
            # servers saturated; scale by c-to-excess power
            _den = gfactorial(self.c_max) * self.c_max ** (n - self.c_max)
        _p_n = (_num / _den) * self.p_z
        self.p_n = _p_n
        return _p_n


@dataclass
class QueueMM1K(BasicQueue):
    """**QueueMM1K** M/M/1/K queueing model (one server, finite capacity K).

    Narrows the base class's `K_max: Optional[int]` to plain `int` so downstream math does not need null-guards.

    Raises:
        ValueError: if server count is not 1 or capacity is not positive.
    """

    K_max: int = 0  # type: ignore[assignment]

    def _validate_params(self) -> None:
        """*_validate_params()* check the M/M/1/K invariants.

        Raises:
            ValueError: if server count is not 1 or capacity is not positive.
        """
        if self.c_max != 1:
            _msg = f"M/M/1/K requires exactly 1 server. c={self.c_max}"
            raise ValueError(_msg)
        if self.K_max < 1:
            _msg = "M/M/1/K requires a positive finite capacity. "
            _msg += f"K={self.K_max}"
            raise ValueError(_msg)

    def is_stable(self) -> bool:
        """*is_stable()* return True; finite-K queues always admit a steady state."""
        return True

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* compute metrics for M/M/1/K, handling both `rho<1` and `rho=1`.

        Side effects: sets `rho`, `tau`, `lamb_eff`, `avg_len`, `avg_len_q`, `avg_wait`, `avg_wait_q` on the instance.
        """
        # rho = lamb / mu (single-server); tau coincides at c=1
        self.rho = self.lamb / self.mu
        self.tau = self.lamb / self.mu
        # block probability P(K) drives the loss / effective arrival rate
        _p_kmax = self.calculate_prob_n(self.K_max)
        # lamb_eff = lamb * (1 - P(K)) is what actually enters the system
        self.lamb_eff = self.lamb * (1 - _p_kmax)
        if self.rho < 1.0:
            # truncated-geometric closed form (rho < 1 regime)
            # mean in service = rho / (1 - rho) (geometric expectation)
            _in_server = self.rho / (1 - self.rho)
            # finite-truncation excess at state K
            _num = (self.K_max + 1) * self.rho ** (self.K_max + 1)
            _den = 1 - self.rho ** (self.K_max + 1)
            _in_queue = _num / _den
            self.avg_len = _in_server - _in_queue
            # equivalent to: L - rho * (1 - P(K))
            self.avg_len_q = self.avg_len - self.lamb_eff / self.mu
        else:
            # rho = 1 saturation: uniform state distribution => L = K/2
            self.avg_len = self.K_max / 2.0
            # Lq closed form at saturation: K(K-1) / (2K+1)
            _num = self.K_max * (self.K_max - 1)
            _den = 2 * self.K_max + 1
            self.avg_len_q = _num / _den
        # Little's Law on the effective rate: W = L / lamb_eff, Wq = Lq / lamb_eff
        if self.lamb_eff > 0:
            self.avg_wait = self.avg_len / self.lamb_eff
            self.avg_wait_q = self.avg_len_q / self.lamb_eff

    def calculate_prob_zero(self) -> float:
        """*calculate_prob_zero()* return P(0) for M/M/1/K, handling both `rho<1` and `rho=1`.

        Returns:
            float: probability of 0 requests in the system.
        """
        # saturation regime: uniform mass over K+1 states
        if self.rho == 1.0:
            return 1.0 / (self.K_max + 1)
        # truncated-geometric: P(0) = (1 - rho) / (1 - rho^(K+1))
        _num = 1 - self.rho
        _den = 1 - self.rho ** (self.K_max + 1)
        return _num / _den

    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* return P(n) for M/M/1/K.

        Args:
            n (int): number of requests in the system.

        Returns:
            float: probability of n requests in the system.
        """
        # saturation: uniform over K+1 states
        if self.rho == 1.0:
            _p_n = 1.0 / (self.K_max + 1)
        else:
            # truncated-geometric: P(n) = (1 - rho) * rho^n / (1 - rho^(K+1))
            _num = (1 - self.rho) * (self.rho ** n)
            _den = 1 - self.rho ** (self.K_max + 1)
            _p_n = _num / _den
        self.p_n = _p_n
        return _p_n


@dataclass
class QueueMMsK(BasicQueue):
    """**QueueMMsK** M/M/c/K queueing model (finite capacity K, c servers).

    Raises:
        ValueError: if server count is less than 1, or capacity is less than the server count.
    """

    K_max: int = 0  # type: ignore[assignment]

    def _validate_params(self) -> None:
        """*_validate_params()* check the M/M/c/K invariants.

        Raises:
            ValueError: if server count is less than 1 or capacity is less than the server count.
        """
        if self.c_max < 1:
            _msg = f"M/M/c/K requires at least one server. c={self.c_max}"
            raise ValueError(_msg)
        if self.K_max < self.c_max:
            _msg = "M/M/c/K requires capacity K >= c. "
            _msg += f"K={self.K_max}, c={self.c_max}"
            raise ValueError(_msg)

    def is_stable(self) -> bool:
        """*is_stable()* return True; finite-K queues always admit a steady state."""
        return True

    def calculate_metrics(self) -> None:
        """*calculate_metrics()* compute metrics for M/M/c/K via truncated state sums.

        Side effects: sets `rho`, `tau`, `p_z`, `lamb_eff`, `avg_len`, `avg_len_q`, `avg_wait`, `avg_wait_q` on the instance.
        """
        # rho = lamb / (c * mu); tau = lamb / mu (offered load)
        self.rho = self.lamb / (self.c_max * self.mu)
        self.tau = self.lamb / self.mu
        self.p_z = self.calculate_prob_zero()
        # block probability P(K) drives the loss / effective arrival rate
        _p_kmax = self.calculate_prob_n(self.K_max)
        # lamb_eff = lamb * (1 - P(K)) is what actually enters the system
        self.lamb_eff = self.lamb * (1 - _p_kmax)
        _k = self.K_max
        _c = self.c_max
        # truncated state-space sums: L = sum_{i=0..K} i * P(i), Lq counts only i >= c
        self.avg_len = sum(i * self.calculate_prob_n(i)
                           for i in range(_k + 1))
        self.avg_len_q = sum((i - _c) * self.calculate_prob_n(i)
                             for i in range(_c, _k + 1))
        # Little's Law on the effective rate
        if self.lamb_eff > 0:
            self.avg_wait = self.avg_len / self.lamb_eff
            self.avg_wait_q = self.avg_len_q / self.lamb_eff

    def calculate_prob_zero(self) -> float:
        """*calculate_prob_zero()* return P(0) for M/M/c/K from the truncated denominator.

        Returns:
            float: probability of 0 requests in the system.
        """
        _tau = self.tau
        _c = self.c_max
        _k = self.K_max
        # mass at states 0..c-1 (no saturation) plus states c..K (saturated tail)
        _sum_under_c = sum((_tau ** i) / gfactorial(i) for i in range(_c))
        _sum_over_c = sum((_tau ** i) / (gfactorial(_c) * (_c ** (i - _c)))
                          for i in range(_c, _k + 1))
        return (_sum_under_c + _sum_over_c) ** -1

    def calculate_prob_n(self, n: int) -> float:
        """*calculate_prob_n()* return P(n) for M/M/c/K.

        Args:
            n (int): number of requests in the system.

        Returns:
            float: probability of n requests; `0.0` when n falls outside `[0, K_max]`.
        """
        if n < 0 or n > self.K_max:
            return 0.0
        _num = self.tau ** n
        if n < self.c_max:
            _den = gfactorial(n)
        else:
            # servers saturated; scale by c-to-excess power
            _den = gfactorial(self.c_max) * self.c_max ** (n - self.c_max)
        _p_n = (_num / _den) * self.p_z
        self.p_n = _p_n
        return _p_n


# registry: model string -> {class, c_rule (single|multi), K_rule (infinite|finite)}; "M/M/s/K" is an alias for "M/M/c/K"
_QUEUE_MODELS: Dict[str, Dict[str, Any]] = {
    "M/M/1": {
        "class": QueueMM1,
        "c_rule": "single",
        "K_rule": "infinite",
    },
    "M/M/s": {
        "class": QueueMMs,
        "c_rule": "multi",
        "K_rule": "infinite",
    },
    "M/M/1/K": {
        "class": QueueMM1K,
        "c_rule": "single",
        "K_rule": "finite",
    },
    "M/M/c/K": {
        "class": QueueMMsK,
        "c_rule": "multi",
        "K_rule": "finite",
    },
    # alias for M/M/c/K
    "M/M/s/K": {
        "class": QueueMMsK,
        "c_rule": "multi",
        "K_rule": "finite",
    },
}


def Queue(model: str,
          lamb: float,
          mu: float,
          c_max: int = 1,
          K_max: Optional[int] = None) -> BasicQueue:
    """*Queue()* factory that returns a concrete queue model by string key.

    NOTE: parameter names follow the PyDASA acronym convention (`lamb` for lambda, `c_max` for server count, `K_max` for capacity).

    Args:
        model (str): queue model key. Supported: `"M/M/1"`, `"M/M/s"`, `"M/M/1/K"`, `"M/M/c/K"`, plus `"M/M/s/K"` as an alias for the `M/M/c/K` class.
        lamb (float): arrival rate (lambda).
        mu (float): service rate (mu).
        c_max (int): number of servers (c). Defaults to 1.
        K_max (Optional[int]): maximum system capacity (K). Defaults to `None` (infinite).

    Raises:
        NotImplementedError: if `model` is not in the supported registry.
        ValueError: when the model's parameter combination is invalid (e.g. M/M/1 with finite K, or M/M/c/K with K < c).

    Returns:
        BasicQueue: instance of a concrete queue model built on the abstract `BasicQueue`.
    """
    if model not in _QUEUE_MODELS:
        _msg = f"Unsupported queue model: {model}. "
        _msg += f"Supported models: {list(_QUEUE_MODELS.keys())}"
        raise NotImplementedError(_msg)
    _spec = _QUEUE_MODELS[model]
    _cls = _spec["class"]
    _c_rule = _spec["c_rule"]
    _K_rule = _spec["K_rule"]
    # validate c_max against the model's c_rule
    _c_single_ok = _c_rule == "single" and c_max == 1
    _c_multi_ok = _c_rule == "multi" and c_max >= 1
    if not (_c_single_ok or _c_multi_ok):
        _msg = f"{model} requires {_c_rule}-server shape. c={c_max}"
        raise ValueError(_msg)
    # validate K_max against the model's K_rule
    _k_inf_ok = _K_rule == "infinite" and K_max is None
    _k_fin_ok = _K_rule == "finite" and K_max is not None
    if not (_k_inf_ok or _k_fin_ok):
        _msg = f"{model} requires {_K_rule} capacity. K={K_max}"
        raise ValueError(_msg)
    # finite multi-server models additionally require K >= c
    _is_finite_multi = _K_rule == "finite" and _c_rule == "multi"
    if _is_finite_multi and K_max is not None and K_max < c_max:
        _msg = f"{model} requires capacity K >= c. K={K_max}, c={c_max}"
        raise ValueError(_msg)
    return _cls(lamb, mu, c_max, K_max)
