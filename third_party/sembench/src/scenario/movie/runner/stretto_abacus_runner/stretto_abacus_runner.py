"""SemBench runner for Abacus inside Stretto (same plans, operators and data as StrettoRunner;
only the optimizer differs). Installed by integrations/sembench/install.sh."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["GREENQUERY_OPTIMIZER"] = "abacus"  # read by reasondb's Config when a query runs
sys.path.append(str(Path(__file__).parent.parent))

from stretto_runner.stretto_runner import StrettoRunner  # noqa: E402


class StrettoAbacusRunner(StrettoRunner):
    def get_system_name(self) -> str:
        return "stretto_abacus"
