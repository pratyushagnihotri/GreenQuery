"""Metric collection classes for evaluation."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import time


@dataclass
class RuntimeMetrics:
    """Runtime measurement metrics."""
    start_time: float
    end_time: float
    duration_seconds: float
    query_name: str
    
    @classmethod
    def measure(cls, query_name: str):
        """Context manager for runtime measurement."""
        start = time.time()
        yield
        end = time.time()
        return cls(
            start_time=start,
            end_time=end,
            duration_seconds=end - start,
            query_name=query_name
        )


@dataclass
class APIMetrics:
    """API usage metrics."""
    total_calls: int
    models_used: list[str]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    cached_calls: int = 0
    providers: Optional[list[str]] = None
    metric_source: str = "measured"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnergyMetrics:
    """Energy consumption metrics."""
    total_kwh: float
    per_call_kwh: float
    total_gco2eq: float
    per_call_gco2eq: float
    metric_source: str = "not_available"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["co2_g"] = self.total_gco2eq
        return data


@dataclass
class CostMetrics:
    """Cost estimation metrics."""
    total_usd: float
    prompt_cost_usd: float
    completion_cost_usd: float
    metric_source: str = "measured"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryResult:
    """Query execution result."""
    query_name: str
    query_type: str
    runtime: RuntimeMetrics
    api: APIMetrics
    energy: EnergyMetrics
    cost: CostMetrics
    result_summary: Dict[str, Any]
    dataset_name: Optional[str] = None
    dataset_source: Optional[str] = None
    dataset_rows: Optional[int] = None
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    backend_profile: Optional[str] = None
    energy_measurement: Optional[str] = None
    hardware_provider: Optional[str] = None
    cloudlab_node_type: Optional[str] = None
    gpu_model: Optional[str] = None
    gpu_count: Optional[int] = None
    cache_state: Optional[str] = None
    model_cache_state: Optional[str] = None
    planner_cache_state: Optional[str] = None
    query_set: Optional[str] = None
    operator_policy: Optional[str] = None
    baseline_type: Optional[str] = None
    physical_operators: Optional[list[str]] = None
    operator_plan: Optional[list[str]] = None
    reasondb_cost: Optional[Dict[str, Any]] = None
    api_events: Optional[list[Dict[str, Any]]] = None
    local_inference: Optional[Dict[str, Any]] = None
    operator_metrics: Optional[list[Dict[str, Any]]] = None
    execution_plan: Optional[Dict[str, Any]] = None
    metric_sources: Optional[Dict[str, str]] = None
    status: str = "success"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        local_inference = self.local_inference or {}
        has_local_tokens = (
            int(local_inference.get("execution_input_tokens") or 0) > 0
            or int(local_inference.get("execution_output_tokens") or 0) > 0
        )
        token_in = (
            local_inference.get("execution_input_tokens")
            if has_local_tokens
            else self.api.total_prompt_tokens
        )
        token_out = (
            local_inference.get("execution_output_tokens")
            if has_local_tokens
            else self.api.total_completion_tokens
        )
        planner_token_in = local_inference.get("planner_input_tokens", 0)
        planner_token_out = local_inference.get("planner_output_tokens", 0)
        return {
            "query_name": self.query_name,
            "query_type": self.query_type,
            "query_operation_type": self.query_type,
            "dataset_name": self.dataset_name,
            "dataset_source": self.dataset_source,
            "dataset_rows": self.dataset_rows,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "backend_profile": self.backend_profile,
            "energy_measurement": self.energy_measurement,
            "hardware_provider": self.hardware_provider,
            "cloudlab_node_type": self.cloudlab_node_type,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "cache_state": self.cache_state,
            "model_cache_state": self.model_cache_state,
            "planner_cache_state": self.planner_cache_state,
            "query_set": self.query_set,
            "operator_policy": self.operator_policy,
            "baseline_type": self.baseline_type,
            "execution_time_seconds": self.runtime.duration_seconds,
            "execution_time": self.runtime.duration_seconds,
            "compilation_or_planning_time": (
                self.reasondb_cost.get("tuning_cost", {}).get("runtime")
                if self.reasondb_cost
                else None
            ),
            "token_in": token_in,
            "token_out": token_out,
            "planner_token_in": planner_token_in,
            "planner_token_out": planner_token_out,
            "total_tokens": (
                int(token_in or 0)
                + int(token_out or 0)
                + int(planner_token_in or 0)
                + int(planner_token_out or 0)
            ),
            "api_cost_usd": self.cost.total_usd,
            "energy_kwh": self.energy.total_kwh,
            "co2_g": self.energy.total_gco2eq,
            "result_rows": self.result_summary.get("rows_returned"),
            "api_calls": self.api.to_dict(),
            "energy": self.energy.to_dict(),
            "cost": self.cost.to_dict(),
            "result_summary": self.result_summary,
            "physical_operators": self.physical_operators or [],
            "operator_plan": self.operator_plan or [],
            "reasondb_cost": self.reasondb_cost,
            "api_events": self.api_events or [],
            "local_inference": local_inference,
            "operator_metrics": self.operator_metrics or [],
            "execution_plan": self.execution_plan or {},
            "metric_sources": self.metric_sources or {},
            "status": self.status,
            "error": self.error
        }

# Made with Bob
