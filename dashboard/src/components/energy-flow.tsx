import { ArrowDown } from "lucide-react";
import type { BenchmarkRecord } from "@/data/benchmarks";

// Query Lab "where the energy goes": the recorded plan as a flow whose boxes are sized by recorded phase
// energy. GreenQuery records energy per phase (planner, execution), not per operator: the execution energy is
// attributed to an operator only when it is the single LLM operator of the plan (relational operators such as
// Project or Limit run on the CPU in milliseconds); otherwise the LLM operators share one execution bracket.

const LLM_OPERATOR = /Backend|Qa|LLM|Sem/i;
const num = (value: unknown) => (typeof value === "number" && !Number.isNaN(value) ? value : null);
const wh = (value: number) => (value < 0.01 ? value.toFixed(4) : value < 1 ? value.toFixed(3) : value.toFixed(2));
const shortOp = (operator: string) => operator.split("-")[0].replace(/([a-z])([A-Z])/g, "$1 $2");

export function EnergyFlow({ record }: { record: BenchmarkRecord }) {
  const planner = num(record.planner_energy_wh), execution = num(record.execution_energy_wh), total = num(record.energy_wh);
  if (planner == null || execution == null || !total) return null;
  const rest = Math.max(total - planner - execution, 0);
  const estimated = record.deployment === "Online API";
  const share = (value: number) => value / total;
  const steps = record.execution_plan?.physical?.steps ?? [];
  const operators = steps.length ? steps.map((step) => step.operator ?? step.type ?? step.id) : record.physical_operators ?? [];
  const llmOps = operators.filter((op) => LLM_OPERATOR.test(op));
  const plannerCalls = num(record.planner_requests) ?? num((record.planning as { total_calls?: number } | undefined)?.total_calls);
  const execCalls = num(record.local_requests) ?? num((record.execution as { total_calls?: number } | undefined)?.total_calls);
  // Box widths follow the energy share, with a floor so small phases stay readable.
  const width = (value: number) => `${Math.max(share(value) * 100, 34)}%`;

  return (
    <section className="ef" aria-label="Where the energy goes in the plan">
      <div className="ef-head">
        <div><span className="eyebrow">WHERE THE ENERGY GOES</span>
          <h4>{wh(total)} Wh for this run{estimated ? " (EcoLogits estimate)" : " (NVML, measured)"}</h4></div>
        <div className="ef-bar" role="img" aria-label={`Planner ${Math.round(share(planner) * 100)} %, execution ${Math.round(share(execution) * 100)} %, outside both phases ${Math.round(share(rest) * 100)} %`}>
          <i className="planner" style={{ width: `${share(planner) * 100}%` }} />
          <i className="execution" style={{ width: `${share(execution) * 100}%` }} />
          <i className="rest" style={{ width: `${share(rest) * 100}%` }} />
        </div>
      </div>
      <div className="ef-flow">
        <div className="ef-box planner" style={{ width: width(planner) }}>
          <span>Planner (LLM)</span><strong>{wh(planner)} Wh · {Math.round(share(planner) * 100)} %</strong>
          <small>{plannerCalls != null ? `${plannerCalls.toLocaleString()} LLM calls` : "calls not recorded"}</small>
        </div>
        <ArrowDown size={15} className="ef-arrow" />
        <div className="ef-exec" style={{ width: width(execution) }}>
          <div className="ef-exec-label">Execution · {wh(execution)} Wh · {Math.round(share(execution) * 100)} %{execCalls != null ? ` · ${execCalls.toLocaleString()} LLM calls` : ""}</div>
          <div className="ef-ops">
            {operators.map((op, index) => {
              const llm = LLM_OPERATOR.test(op);
              return (
                <div key={`${op}-${index}`} className={`ef-op${llm ? " llm" : ""}`} title={op}>
                  <strong>{shortOp(op)}</strong>
                  <small>{llm ? (llmOps.length === 1 ? `≈ all ${wh(execution)} Wh` : "shares the execution energy") : "relational, ≈ 0 Wh"}</small>
                </div>
              );
            })}
          </div>
        </div>
        {rest > 0 && <>
          <ArrowDown size={15} className="ef-arrow" />
          <div className="ef-box rest" style={{ width: width(rest) }}>
            <span>Outside both phases</span><strong>{wh(rest)} Wh · {Math.round(share(rest) * 100)} %</strong>
            <small>startup, data loading, idle</small>
          </div>
        </>}
      </div>
      <p className="ef-note">Box size = share of the run’s energy. Energy is recorded per phase, not per operator; the execution energy
        is attributed to an operator only when it is the plan’s single LLM operator.</p>
    </section>
  );
}
