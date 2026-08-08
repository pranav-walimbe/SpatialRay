# SpatialRay

A decoupled inference framework for geospatial ML

```text
 Requests
    |
    v
+---------+      +-------------+      +----------------+      +----------------+
| Ingress | ---> | Decode Pool | ---> | Transform Pool | ---> | Inference Pool | ---> Results
+----+----+      +------+------+      +-------+--------+      +-------+--------+
     |                  |                     |                       |
     | register work    | start / finish      | start / finish        | start / finish
     v                  v                     v                       v
+--------------------------------------------------------------------------------+
|                         Weighted Pending-Work Ledger                           |
+--------------------------------------+-----------------------------------------+
                                       |
                              exact work snapshots
                                       v
                         +---------------------------+
                         | Workload Autoscaling      |
                         | Independent pool scaling  |
                         +---------------------------+
```
