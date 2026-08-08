"""
The sliding-window meter converting submitted items into per-pool work rates.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from math import isfinite
from typing import Protocol

from spatial_ray.scaling.metrics import work_rate_metric


class WorkEstimator(Protocol):
    """A request-to-work mapping supplied by a pluggable pool component."""

    def __call__(self, item: object) -> float:
        """Estimate the work a pool will perform for one submitted item.

        Args:
            item: Item that will eventually enter the pool.

        Returns:
            Nonnegative work in the pool's declared unit.
        """
        ...


class WorkloadMeter:
    """Measure recent request and stage-specific work arrival rates."""

    def __init__(
        self,
        estimators: Mapping[str, WorkEstimator],
        *,
        window_s: float = 30.0,
        min_window_s: float = 1.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if min_window_s <= 0.0 or min_window_s > window_s:
            raise ValueError(f"min_window_s must be in (0, window_s], got {min_window_s}")
        self._estimators = dict(estimators)
        self._window_s = window_s
        self._min_window_s = min_window_s
        self._time_fn = time_fn
        self._events: deque[tuple[float, dict[str, float]]] = deque()
        self._lock = threading.Lock()

    def record(self, item: object) -> dict[str, float]:
        """Record one submitted item for every configured pool.

        Args:
            item: Item whose stage work each estimator predicts.

        Returns:
            Validated estimated work keyed by pool name.
        """
        work = {name: float(estimator(item)) for name, estimator in self._estimators.items()}
        invalid = {
            name: value for name, value in work.items() if not isfinite(value) or value < 0.0
        }
        if invalid:
            raise ValueError(f"work estimates must be nonnegative, got {invalid}")
        now = self._time_fn()
        with self._lock:
            self._events.append((now, work))
            self._prune(now)
        return work

    def snapshot(self) -> dict[str, float]:
        """Return recent work and request rates for each configured pool.

        Returns:
            Metric name to rate over the active sliding window.
        """
        now = self._time_fn()
        with self._lock:
            self._prune(now)
            events = tuple(self._events)
        rates: dict[str, float] = {}
        if events:
            elapsed_s = max(self._min_window_s, min(self._window_s, now - events[0][0]))
        else:
            elapsed_s = self._window_s
        for name in self._estimators:
            rates[work_rate_metric(name)] = sum(event[1][name] for event in events) / elapsed_s
        return rates

    def _prune(self, now: float) -> None:
        # discard events outside the inclusive sliding window
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
