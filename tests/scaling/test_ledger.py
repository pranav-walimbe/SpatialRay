"""
Tests exact weighted lifecycle accounting in the runtime-neutral pending-work ledger.
"""

from __future__ import annotations

import pytest

from spatial_ray.scaling import InMemoryPendingWorkLedger, PendingWorkLedger


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def test_registers_exact_pending_work():
    """Registration sums heterogeneous request weights independently for each pool."""
    ledger = InMemoryPendingWorkLedger()
    ledger.register("small", {"decode": 100.0, "inference": 2.0})
    ledger.register("large", {"decode": 900.0, "inference": 12.0})
    snapshot = ledger.snapshot()
    assert snapshot.pending == {"decode": 1000.0, "inference": 14.0}
    assert snapshot.executing == {}


def test_moves_work_through_the_lifecycle():
    """Starting and finishing move only the selected pool's exact request weight."""
    ledger = InMemoryPendingWorkLedger()
    ledger.register("request", {"decode": 100.0, "transform": 250.0})
    ledger.start("request", "decode")
    running = ledger.snapshot()
    assert running.pending == {"decode": 0.0, "transform": 250.0}
    assert running.executing == {"decode": 100.0}
    ledger.finish("request", "decode")
    finished = ledger.snapshot()
    assert finished.pending == {"decode": 0.0, "transform": 250.0}
    assert finished.executing == {"decode": 0.0}


def test_duplicate_events_are_idempotent():
    """Retries do not register, start, finish, or cancel the same work twice."""
    ledger = InMemoryPendingWorkLedger()
    work = {"decode": 100.0, "inference": 2.0}
    ledger.register("request", work)
    ledger.register("request", work)
    ledger.start("request", "decode")
    ledger.start("request", "decode")
    ledger.finish("request", "decode")
    ledger.finish("request", "decode")
    ledger.cancel("request")
    ledger.cancel("request")
    snapshot = ledger.snapshot()
    assert snapshot.pending == {"decode": 0.0, "inference": 0.0}
    assert snapshot.executing == {"decode": 0.0}
    assert snapshot.sequence == 4


def test_cancel_removes_pending_and_executing_work():
    """Cancellation clears unfinished work without reviving a completed stage."""
    ledger = InMemoryPendingWorkLedger()
    ledger.register("request", {"decode": 100.0, "transform": 250.0, "inference": 2.0})
    ledger.start("request", "decode")
    ledger.finish("request", "decode")
    ledger.start("request", "transform")
    ledger.cancel("request")
    snapshot = ledger.snapshot()
    assert snapshot.pending == {"decode": 0.0, "transform": 0.0, "inference": 0.0}
    assert snapshot.executing == {"decode": 0.0, "transform": 0.0}


def test_rejects_conflicting_registration_and_invalid_transitions():
    """Conflicting work and completion before execution fail without corrupting totals."""
    ledger = InMemoryPendingWorkLedger()
    ledger.register("request", {"decode": 100.0})
    with pytest.raises(ValueError, match="different work"):
        ledger.register("request", {"decode": 200.0})
    with pytest.raises(ValueError, match="has not started"):
        ledger.finish("request", "decode")
    with pytest.raises(KeyError, match="unknown request"):
        ledger.start("missing", "decode")
    assert ledger.snapshot().pending == {"decode": 100.0}


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_work(weight):
    """Invalid estimates cannot enter authoritative aggregate state."""
    ledger = InMemoryPendingWorkLedger()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ledger.register("request", {"decode": weight})


def test_snapshot_is_timestamped_and_detached():
    """Snapshots carry the ledger sequence and cannot mutate later aggregate state."""
    clock = _Clock()
    ledger = InMemoryPendingWorkLedger(time_fn=clock)
    ledger.register("request", {"decode": 100.0})
    snapshot = ledger.snapshot()
    snapshot.pending["decode"] = 999.0
    clock.now = 12.0
    current = ledger.snapshot()
    assert current.captured_at_s == 12.0
    assert current.pending == {"decode": 100.0}
    assert isinstance(ledger, PendingWorkLedger)


def test_terminal_history_expires_after_the_idempotency_window():
    """Finished identities can be reused only after their bounded retention window expires."""
    clock = _Clock()
    ledger = InMemoryPendingWorkLedger(terminal_retention_s=5.0, time_fn=clock)
    ledger.register("request", {"decode": 100.0})
    ledger.start("request", "decode")
    ledger.finish("request", "decode")
    clock.now = 14.0
    with pytest.raises(ValueError, match="different work"):
        ledger.register("request", {"decode": 200.0})
    clock.now = 15.0
    ledger.register("request", {"decode": 200.0})
    assert ledger.snapshot().pending == {"decode": 200.0}
