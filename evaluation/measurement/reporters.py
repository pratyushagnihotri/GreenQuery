"""Result reporting and formatting."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


class JSONReporter:
    """Report results in JSON format."""

    @staticmethod
    def save(data: Dict[str, Any], output_path: Path):
        """Save data to JSON file."""
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ JSON saved to: {output_path}")


class CSVReporter:
    """Report results in CSV format."""

    @staticmethod
    def save_summary(measurements: List[Dict[str, Any]], output_path: Path):
        """Save measurement summary to CSV."""
        if not measurements:
            return

        rows = []
        for measurement in measurements:
            config = measurement.get("config", {})
            for query in measurement.get("queries", []):
                api_calls = query.get("api_calls", {})
                energy = query.get("energy", {})
                cost = query.get("cost", {})
                result_summary = query.get("result_summary", {})
                metric_sources = query.get("metric_sources", {})
                rows.append(
                    {
                        "measurement_id": measurement["measurement_id"],
                        "demo_name": measurement["demo_name"],
                        "timestamp": measurement["timestamp"],
                        "dataset_name": query.get("dataset_name") or config.get("dataset_name"),
                        "dataset_source": query.get("dataset_source") or config.get("dataset_source"),
                        "dataset_rows": query.get("dataset_rows") or config.get("dataset_rows"),
                        "model_provider": query.get("model_provider") or config.get("model_provider"),
                        "model_name": query.get("model_name") or config.get("model_name"),
                        "model_type": query.get("model_type") or config.get("model_type"),
                        "backend_profile": query.get("backend_profile") or config.get("backend_profile"),
                        "energy_measurement": query.get("energy_measurement") or config.get("energy_measurement"),
                        "cache_state": query.get("cache_state") or config.get("cache_state"),
                        "query_set": query.get("query_set") or config.get("query_set"),
                        "operator_policy": query.get("operator_policy") or config.get("operator_policy"),
                        "baseline_type": query.get("baseline_type") or config.get("baseline_type"),
                        "query_name": query["query_name"],
                        "query_operation_type": query.get("query_operation_type", query["query_type"]),
                        "execution_time_seconds": query.get("execution_time_seconds"),
                        "compilation_or_planning_time": query.get("compilation_or_planning_time"),
                        "result_rows": query.get("result_rows", result_summary.get("rows_returned")),
                        "api_calls": api_calls.get("total_calls"),
                        "cached_calls": api_calls.get("cached_calls", 0),
                        "token_in": query.get("token_in", api_calls.get("total_prompt_tokens")),
                        "token_out": query.get("token_out", api_calls.get("total_completion_tokens")),
                        "total_tokens": query.get("total_tokens", api_calls.get("total_tokens")),
                        "energy_kwh": query.get("energy_kwh", energy.get("total_kwh")),
                        "co2_g": query.get(
                            "co2_g", energy.get("co2_g", energy.get("total_gco2eq"))
                        ),
                        "api_cost_usd": query.get("api_cost_usd", cost.get("total_usd")),
                        "api_source": metric_sources.get("api_calls", api_calls.get("metric_source")),
                        "energy_source": metric_sources.get("energy", energy.get("metric_source")),
                        "co2_source": metric_sources.get("co2", energy.get("metric_source")),
                        "physical_operators": ",".join(query.get("physical_operators", [])),
                    }
                )

        if rows:
            fieldnames = rows[0].keys()
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"✅ CSV saved to: {output_path}")


class MarkdownReporter:
    """Report results in Markdown format."""

    @staticmethod
    def create_summary(measurement: Dict[str, Any]) -> str:
        """Create markdown summary of measurement."""
        md = f"# Measurement Report: {measurement['demo_name']}\n\n"
        md += f"**Measurement ID:** {measurement['measurement_id']}\n"
        md += f"**Timestamp:** {measurement['timestamp']}\n"
        md += f"**Metric schema:** {measurement.get('metric_schema_version', 'legacy')}\n\n"

        md += "## Environment\n\n"
        for key, value in measurement["environment"].items():
            md += f"- **{key}:** {value}\n"

        md += "\n## Query Results\n\n"
        md += "| Query | Baseline | Backend | Operator Policy | Dataset | Model | Type | Time (s) | Rows | API Calls | Cached | Tokens In | Tokens Out | Energy (kWh) | CO2 (g) | Cost ($) |\n"
        md += "|-------|----------|---------|-----------------|---------|-------|------|----------|------|-----------|--------|-----------|------------|--------------|--------|----------|\n"

        config = measurement.get("config", {})
        for query in measurement["queries"]:
            api_calls = query.get("api_calls", {})
            energy = query.get("energy", {})
            cost = query.get("cost", {})
            result_summary = query.get("result_summary", {})
            md += f"| {query['query_name']} "
            md += f"| {query.get('baseline_type') or config.get('baseline_type', '')} "
            md += f"| {query.get('backend_profile') or config.get('backend_profile', '')} "
            md += f"| {query.get('operator_policy') or config.get('operator_policy', '')} "
            md += f"| {query.get('dataset_name') or config.get('dataset_name', '')} "
            md += f"| {query.get('model_name') or config.get('model_name', '')} "
            md += f"| {query.get('query_operation_type', query['query_type'])} "
            md += f"| {query.get('execution_time_seconds', 0):.2f} "
            md += f"| {query.get('result_rows', result_summary.get('rows_returned', ''))} "
            md += f"| {api_calls.get('total_calls', '')} "
            md += f"| {api_calls.get('cached_calls', 0)} "
            md += f"| {query.get('token_in', api_calls.get('total_prompt_tokens', ''))} "
            md += f"| {query.get('token_out', api_calls.get('total_completion_tokens', ''))} "
            md += f"| {query.get('energy_kwh', energy.get('total_kwh', 0)):.6f} "
            md += f"| {query.get('co2_g', energy.get('co2_g', energy.get('total_gco2eq', 0))):.3f} "
            md += f"| {query.get('api_cost_usd', cost.get('total_usd', 0)):.6f} |\n"

        md += "\n## Metric Sources\n\n"
        md += "Metrics marked `not_available` were not emitted by the underlying tool for this run and should not be treated as final evidence.\n\n"
        for query in measurement["queries"]:
            sources = query.get("metric_sources", {})
            if sources:
                source_text = ", ".join(f"{key}={value}" for key, value in sources.items())
                md += f"- **{query['query_name']}:** {source_text}\n"

        md += "\n## Physical Operators\n\n"
        for query in measurement["queries"]:
            operators = query.get("physical_operators", [])
            operator_text = ", ".join(operators) if operators else "not_available"
            baseline_type = query.get("baseline_type") or config.get("baseline_type", "unknown")
            md += f"- **{query['query_name']} ({baseline_type}):** {operator_text}\n"

        md += "\n## Totals\n\n"
        totals = measurement["totals"]
        md += f"- **Total Queries:** {totals['total_queries']}\n"
        md += f"- **Total API Calls:** {totals['total_api_calls']}\n"
        md += f"- **Total Cached Calls:** {totals.get('total_cached_calls', 0)}\n"
        md += f"- **Total Tokens:** {totals['total_tokens']}\n"
        md += f"- **Total Energy:** {totals['total_energy_kwh']:.6f} kWh\n"
        md += f"- **Total Carbon:** {totals.get('total_co2_g', totals['total_carbon_gco2eq']):.3f} gCO2eq\n"
        md += f"- **Total Cost:** ${totals['total_cost_usd']:.6f}\n"

        return md
