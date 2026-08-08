"""
The stable metric names shared by workload meters and scaling adapters.
"""

WORK_RATE_PREFIX = "spatialray_work_rate_"
REQUEST_RATE_PREFIX = "spatialray_request_rate_"
MEAN_WORK_PREFIX = "spatialray_mean_work_"


def work_rate_metric(pool: str) -> str:
    """Return the custom work-rate metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{WORK_RATE_PREFIX}{pool}"


def request_rate_metric(pool: str) -> str:
    """Return the custom request-rate metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{REQUEST_RATE_PREFIX}{pool}"


def mean_work_metric(pool: str) -> str:
    """Return the retained mean-work metric name for a pool.

    Args:
        pool: Pool whose metric name to build.

    Returns:
        The metric key reported through the runtime autoscaling context.
    """
    return f"{MEAN_WORK_PREFIX}{pool}"
