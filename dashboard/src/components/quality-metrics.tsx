import { qualityMetrics, type BenchmarkRecord } from "@/data/benchmarks";

const format = (value: number | string | boolean) => typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : typeof value === "boolean" ? (value ? "Correct" : "Not correct") : value;

export function QualityMetrics({ record }: { record: BenchmarkRecord }) {
  const metrics = qualityMetrics(record);
  const isLegacy = record.quality_metric === "exact_numeric_field_accuracy";
  return <>
    <div className="quality-contract"><span>Evaluation contract</span><strong>{record.quality_metric?.replaceAll("_", " ") ?? "Not recorded"}</strong></div>
    {isLegacy && <p className="quality-caveat">Historical scorer output. Not comparable answer-quality evidence.</p>}
    <div className="quality-grid">{metrics.map((metric) => <div className="stat" key={metric.label}><span>{metric.label}</span><strong>{format(metric.value)}</strong><small className={metric.provenance === "derived" ? "metric-provenance derived" : "metric-provenance"}>{metric.provenance}{metric.direction === "informational" ? "" : ` · ${metric.direction}`}</small></div>)}</div>
  </>;
}
