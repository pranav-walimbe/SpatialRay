"""
Tests request records keep transport identity separate from scaling state.
"""

from __future__ import annotations

import pytest

from spatial_ray import RequestRecord


def test_request_record_carries_only_identity_and_payload():
    """A request record remains independent of pool work estimates."""
    payload = object()
    record = RequestRecord(request_id="request-1", payload=payload)
    assert record.request_id == "request-1"
    assert record.payload is payload


def test_request_record_requires_identity():
    """An empty identity cannot address lifecycle state in the ledger."""
    with pytest.raises(ValueError, match="must not be empty"):
        RequestRecord(request_id="", payload=object())
