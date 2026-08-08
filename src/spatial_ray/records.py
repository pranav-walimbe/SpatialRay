"""
Request identity and payload records passed through a serving pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True)
class RequestRecord(Generic[PayloadT]):
    """Carry stable request identity and its payload through a pipeline."""

    request_id: str
    payload: PayloadT

    def __post_init__(self) -> None:
        # reject an identity that the pending-work ledger cannot address
        if not self.request_id:
            raise ValueError("request_id must not be empty")
