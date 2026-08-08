"""
The stable metric names shared by workload meters and scaling adapters.
"""

WORK_RATE_PREFIX = "spatialray_work_rate_"
PENDING_WORK_PREFIX = "spatialray_pending_work_"
EXECUTING_WORK_PREFIX = "spatialray_executing_work_"
LEDGER_SNAPSHOT_AGE = "spatialray_ledger_snapshot_age_s"


def work_rate_metric(pool: str) -> str:
    """Return the custom work-rate metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{WORK_RATE_PREFIX}{pool}"


def pending_work_metric(pool: str) -> str:
    """Return the exact pending-work metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{PENDING_WORK_PREFIX}{pool}"


def executing_work_metric(pool: str) -> str:
    """Return the exact executing-work metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{EXECUTING_WORK_PREFIX}{pool}"
