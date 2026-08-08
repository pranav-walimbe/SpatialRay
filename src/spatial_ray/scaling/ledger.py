"""
Runtime-neutral weighted pending-work ledger contracts and in-memory implementation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

_PENDING = "pending"
_EXECUTING = "executing"
_FINISHED = "finished"
_CANCELED = "canceled"


@dataclass(frozen=True)
class PendingWorkSnapshot:
    """Capture exact pending and executing work at one ledger sequence."""

    sequence: int
    captured_at_s: float
    pending: dict[str, float]
    executing: dict[str, float]


@runtime_checkable
class PendingWorkLedger(Protocol):
    """Own authoritative per-request work estimates and lifecycle transitions."""

    def register(self, request_id: str, work_by_pool: Mapping[str, float]) -> None:
        """Register one request's estimated work before pipeline dispatch.

        Args:
            request_id: Stable identity shared by every pool lifecycle event.
            work_by_pool: Nonnegative estimated work keyed by pool name.
        """
        ...

    def start(self, request_id: str, pool: str) -> None:
        """Move one pool's request work from pending to executing.

        Args:
            request_id: Identity of the registered request.
            pool: Pool beginning execution.
        """
        ...

    def finish(self, request_id: str, pool: str) -> None:
        """Remove one pool's request work after execution finishes.

        Args:
            request_id: Identity of the registered request.
            pool: Pool completing execution.
        """
        ...

    def cancel(self, request_id: str) -> None:
        """Remove all pending and executing work for one request.

        Args:
            request_id: Identity of the request being canceled.
        """
        ...

    def snapshot(self) -> PendingWorkSnapshot:
        """Read current aggregate work without exposing mutable ledger state.

        Returns:
            A timestamped copy of pending and executing work by pool.
        """
        ...


@dataclass
class _StageEntry:
    weight: float
    state: str = _PENDING


class InMemoryPendingWorkLedger:
    """Store exact weighted lifecycle state in one thread-safe process."""

    def __init__(
        self,
        *,
        terminal_retention_s: float = 300.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isfinite(terminal_retention_s) or terminal_retention_s < 0.0:
            raise ValueError("terminal_retention_s must be finite and nonnegative")
        self._time_fn = time_fn
        self._terminal_retention_s = terminal_retention_s
        self._requests: dict[str, dict[str, _StageEntry]] = {}
        self._terminal_at: dict[str, float] = {}
        self._pending: dict[str, float] = {}
        self._executing: dict[str, float] = {}
        self._sequence = 0
        self._lock = threading.Lock()

    def register(self, request_id: str, work_by_pool: Mapping[str, float]) -> None:
        """Register one request's estimated work before pipeline dispatch.

        Args:
            request_id: Stable identity shared by every pool lifecycle event.
            work_by_pool: Nonnegative estimated work keyed by pool name.
        """
        work = _validated_work(request_id, work_by_pool)
        with self._lock:
            self._prune(self._time_fn())
            existing = self._requests.get(request_id)
            if existing is not None:
                registered = {pool: entry.weight for pool, entry in existing.items()}
                if registered != work:
                    raise ValueError(f"request {request_id!r} was registered with different work")
                return
            self._requests[request_id] = {
                pool: _StageEntry(weight=weight) for pool, weight in work.items()
            }
            for pool, weight in work.items():
                self._add(self._pending, pool, weight)
            self._sequence += 1

    def start(self, request_id: str, pool: str) -> None:
        """Move one pool's request work from pending to executing.

        Args:
            request_id: Identity of the registered request.
            pool: Pool beginning execution.
        """
        with self._lock:
            entry = self._entry(request_id, pool)
            if entry.state == _PENDING:
                self._add(self._pending, pool, -entry.weight)
                self._add(self._executing, pool, entry.weight)
                entry.state = _EXECUTING
                self._sequence += 1
            elif entry.state not in (_EXECUTING, _FINISHED, _CANCELED):
                raise RuntimeError(f"unknown lifecycle state {entry.state!r}")

    def finish(self, request_id: str, pool: str) -> None:
        """Remove one pool's request work after execution finishes.

        Args:
            request_id: Identity of the registered request.
            pool: Pool completing execution.
        """
        with self._lock:
            entry = self._entry(request_id, pool)
            if entry.state == _PENDING:
                raise ValueError(f"request {request_id!r} has not started in pool {pool!r}")
            if entry.state == _EXECUTING:
                self._add(self._executing, pool, -entry.weight)
                entry.state = _FINISHED
                self._sequence += 1
                self._mark_terminal(request_id)
            elif entry.state not in (_FINISHED, _CANCELED):
                raise RuntimeError(f"unknown lifecycle state {entry.state!r}")

    def cancel(self, request_id: str) -> None:
        """Remove all pending and executing work for one request.

        Args:
            request_id: Identity of the request being canceled.
        """
        with self._lock:
            stages = self._requests.get(request_id)
            if stages is None:
                raise KeyError(f"unknown request {request_id!r}")
            changed = False
            for pool, entry in stages.items():
                if entry.state == _PENDING:
                    self._add(self._pending, pool, -entry.weight)
                elif entry.state == _EXECUTING:
                    self._add(self._executing, pool, -entry.weight)
                else:
                    continue
                entry.state = _CANCELED
                changed = True
            if changed:
                self._sequence += 1
                self._mark_terminal(request_id)

    def snapshot(self) -> PendingWorkSnapshot:
        """Read current aggregate work without exposing mutable ledger state.

        Returns:
            A timestamped copy of pending and executing work by pool.
        """
        with self._lock:
            now = self._time_fn()
            self._prune(now)
            return PendingWorkSnapshot(
                sequence=self._sequence,
                captured_at_s=now,
                pending=dict(self._pending),
                executing=dict(self._executing),
            )

    def _entry(self, request_id: str, pool: str) -> _StageEntry:
        # resolve one registered stage while keeping public transition methods compact
        stages = self._requests.get(request_id)
        if stages is None:
            raise KeyError(f"unknown request {request_id!r}")
        entry = stages.get(pool)
        if entry is None:
            raise KeyError(f"request {request_id!r} has no work for pool {pool!r}")
        return entry

    def _add(self, totals: dict[str, float], pool: str, delta: float) -> None:
        # adjust one aggregate and normalize numerical zero after balanced transitions
        value = totals.get(pool, 0.0) + delta
        totals[pool] = 0.0 if abs(value) < 1e-12 else value

    def _mark_terminal(self, request_id: str) -> None:
        # retain a terminal request briefly so duplicate lifecycle events remain idempotent
        stages = self._requests[request_id]
        if all(entry.state in (_FINISHED, _CANCELED) for entry in stages.values()):
            self._terminal_at[request_id] = self._time_fn()

    def _prune(self, now: float) -> None:
        # bound terminal history without changing current aggregate work
        expired = [
            request_id
            for request_id, terminal_at in self._terminal_at.items()
            if now - terminal_at >= self._terminal_retention_s
        ]
        for request_id in expired:
            del self._terminal_at[request_id]
            del self._requests[request_id]


def _validated_work(request_id: str, work_by_pool: Mapping[str, float]) -> dict[str, float]:
    # copy and validate registration before it can mutate authoritative ledger state
    if not request_id:
        raise ValueError("request_id must not be empty")
    if not work_by_pool:
        raise ValueError("work_by_pool must not be empty")
    work = {pool: float(weight) for pool, weight in work_by_pool.items()}
    if any(not pool for pool in work):
        raise ValueError("pool names must not be empty")
    invalid = {
        pool: weight for pool, weight in work.items() if not isfinite(weight) or weight < 0.0
    }
    if invalid:
        raise ValueError(f"work estimates must be finite and nonnegative, got {invalid}")
    return work
