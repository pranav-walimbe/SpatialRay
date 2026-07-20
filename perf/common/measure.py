"""
Cross-platform peak-RSS reader shared by every stage process.
"""

from __future__ import annotations

import resource
import sys

# ru_maxrss is bytes on macOS but kibibytes on Linux, so normalize to bytes
_RU_MAXRSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024


def peak_rss_bytes() -> int:
    """Return the current process peak resident set size in bytes.

    Returns:
        Peak RSS high-water mark normalized to bytes across platforms.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RU_MAXRSS_TO_BYTES
