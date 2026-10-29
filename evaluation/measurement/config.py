"""Configuration management for measurements."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MeasurementConfig:
    """Configuration for measurement runs."""
    
    # Demo configuration
    demo_name: str
    device: str = "cpu"
    
    # Guarantee configuration
    precision_guarantee: float = 0.8
    recall_guarantee: float = 0.8
    
    # Output configuration
    output_dir: str = "evaluation/results"
    save_json: bool = True
    save_csv: bool = True
    
    # Measurement configuration
    num_iterations: int = 3
    warmup_iterations: int = 0
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    planner_model_name: Optional[str] = None
    model_provider: str = "openai"
    model_type: str = "api"
    api_key: Optional[str] = None
    backend_profile: str = "api"
    energy_measurement: str = "none"
    co2_g_per_kwh: Optional[float] = None
    hardware_provider: Optional[str] = None
    cloudlab_node_type: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_count: Optional[int] = None
    idle_gpu_power_w: Optional[float] = None
    direct_qa_batch_size: Optional[int] = None
    reasondb_database_name: Optional[str] = None

    # Dataset/run context
    dataset_name: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_rows: Optional[int] = None
    dataset_sha256: Optional[str] = None
    cache_state: str = "not_controlled"
    model_cache_state: str = "not_controlled"
    local_cache_policy: str = "default"
    planner_cache_state: str = "not_controlled"
    online_cache_policy: str = "default"
    query_set: str = "baseline"
    operator_policy: str = "default"
    baseline_type: str = "traditional_planner"
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "demo_name": self.demo_name,
            "device": self.device,
            "precision_guarantee": self.precision_guarantee,
            "recall_guarantee": self.recall_guarantee,
            "model_name": self.model_name,
            "planner_model_name": self.planner_model_name,
            "model_provider": self.model_provider,
            "model_type": self.model_type,
            "backend_profile": self.backend_profile,
            "energy_measurement": self.energy_measurement,
            "co2_g_per_kwh": self.co2_g_per_kwh,
            "hardware_provider": self.hardware_provider,
            "cloudlab_node_type": self.cloudlab_node_type,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "idle_gpu_power_w": self.idle_gpu_power_w,
            "direct_qa_batch_size": self.direct_qa_batch_size,
            "reasondb_database_name": self.reasondb_database_name,
            "dataset_name": self.dataset_name,
            "dataset_source": self.dataset_source,
            "dataset_rows": self.dataset_rows,
            "dataset_sha256": self.dataset_sha256,
            "cache_state": self.cache_state,
            "model_cache_state": self.model_cache_state,
            "local_cache_policy": self.local_cache_policy,
            "planner_cache_state": self.planner_cache_state,
            "online_cache_policy": self.online_cache_policy,
            "query_set": self.query_set,
            "operator_policy": self.operator_policy,
            "baseline_type": self.baseline_type,
        }

# Made with Bob
