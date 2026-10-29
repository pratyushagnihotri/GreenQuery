from typing import Dict

from greenquery_server.adapters.base import AVAILABLE, EXPERIMENTAL, PLANNED, RunSpec, SlurmJob, Step, SystemAdapter
from greenquery_server.adapters.sembench import SEMBENCH_SYSTEMS
from greenquery_server.adapters.stretto import StrettoAdapter

SYSTEMS: Dict[str, SystemAdapter] = {
    adapter.key: adapter
    for adapter in [
        StrettoAdapter("stretto"),
        StrettoAdapter("abacus"),
        StrettoAdapter("lotus"),
        *SEMBENCH_SYSTEMS,
    ]
}

__all__ = ["AVAILABLE", "EXPERIMENTAL", "PLANNED", "RunSpec", "SlurmJob", "Step", "SystemAdapter", "SYSTEMS"]
