"""Test script for measurement infrastructure."""

import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.wrapper import MeasurementWrapper
from evaluation.measurement.config import MeasurementConfig


def test_measurement():
    """Test the measurement infrastructure."""
    
    config = MeasurementConfig(
        demo_name="test",
        device="cpu",
        output_dir="evaluation/results/test"
    )
    
    with MeasurementWrapper("test", config.output_dir, config.to_dict()) as wrapper:
        # Simulate query execution
        with wrapper.measure_query("test_query", "test") as tracker:
            time.sleep(0.1)  # Simulate work
            tracker.set_result(["result1", "result2"])
        
        print("✅ Test measurement completed")


if __name__ == "__main__":
    test_measurement()

# Made with Bob
