"""
Ray resource helpers used to place disaggregated pool replicas.
"""

_NODE_RESOURCE_SUFFIX = "_node"


def node_resource(pool: str) -> str:
    """Return the custom Ray resource pinning a replica to a pool's node group.

    Args:
        pool: Pool name such as decode, transform, or inference.

    Returns:
        The custom Ray resource key shared by the pool and its node group.
    """
    return f"{pool}{_NODE_RESOURCE_SUFFIX}"
