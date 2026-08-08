"""
The Ray-hosted owner of exact weighted pending-work state.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from spatial_ray.scaling.ledger import InMemoryPendingWorkLedger, PendingWorkSnapshot


class RayPendingWorkLedger:
    """Expose the runtime-neutral pending-work ledger through a Ray actor."""

    def __init__(self, terminal_retention_s: float = 300.0) -> None:
        self._ledger = InMemoryPendingWorkLedger(
            terminal_retention_s=terminal_retention_s,
            time_fn=time.time,
        )

    def register(self, request_id: str, work_by_pool: Mapping[str, float]) -> None:
        """Register estimated work before a request enters the pipeline.

        Args:
            request_id: Stable request identity shared by every pool.
            work_by_pool: Nonnegative estimated work keyed by pool name.
        """
        self._ledger.register(request_id, work_by_pool)

    def start(self, request_id: str, pool: str) -> None:
        """Move one pool's request work from pending to executing.

        Args:
            request_id: Identity of the registered request.
            pool: Pool beginning execution.
        """
        self._ledger.start(request_id, pool)

    def finish(self, request_id: str, pool: str) -> None:
        """Remove one pool's request work after successful execution.

        Args:
            request_id: Identity of the registered request.
            pool: Pool completing execution.
        """
        self._ledger.finish(request_id, pool)

    def cancel(self, request_id: str) -> None:
        """Remove all remaining work for a failed request.

        Args:
            request_id: Identity of the request being canceled.
        """
        self._ledger.cancel(request_id)

    def snapshot(self) -> PendingWorkSnapshot:
        """Return the latest exact aggregate work state.

        Returns:
            A timestamped snapshot of pending and executing work by pool.
        """
        return self._ledger.snapshot()
