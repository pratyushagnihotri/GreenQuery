import { CircleDashed, Database, FileCheck2, GitBranch, Route } from "lucide-react";
import { useMemo, useState } from "react";
import { QueryPlan as RecordedQueryPlan } from "@/components/query-plan";
import type { BenchmarkRecord, StructuredExecutionPlan } from "@/data/benchmarks";

type StageId = "logical" | "physical" | "mapping" | "evidence" | "optimizer";

const stageInfo: Array<{ id: StageId; label: string; description: string; icon: typeof Route }> = [
  { id: "logical", label: "Logical intent", description: "What the query asks the system to do.", icon: Route },
  { id: "physical", label: "Selected execution", description: "What the recorded benchmark ran.", icon: Database },
  { id: "mapping", label: "Implementation map", description: "How logical steps map to physical operators.", icon: GitBranch },
  { id: "evidence", label: "Run evidence", description: "Full-query measurements for this selected run.", icon: FileCheck2 },
  { id: "optimizer", label: "Optimizer history", description: "Recorded candidate exploration or rewrites.", icon: CircleDashed },
];

function stagePlan(plan: StructuredExecutionPlan, stage: StageId) {
  if (stage === "logical") return { schema_version: plan.schema_version, logical: plan.logical };
  if (stage === "physical") return { schema_version: plan.schema_version, physical: plan.physical };
  return plan;
}

export function StagedQueryPlan({ operators, query, executionPlan, record }: { operators?: string[]; query: string; executionPlan?: StructuredExecutionPlan; record?: BenchmarkRecord }) {
  const structured = Boolean(executionPlan?.logical?.steps.length || executionPlan?.physical?.steps.length);
  const optimizerRecorded = (executionPlan?.optimization?.candidate_steps?.length ?? 0) > 0;
  const available = useMemo(() => new Set<StageId>(structured ? ["logical", "physical", "mapping", "evidence"] : ["physical", "evidence"]), [structured]);
  const [stage, setStage] = useState<StageId>(structured ? "mapping" : "physical");
  const visiblePlan = executionPlan ? stagePlan(executionPlan, stage) : undefined;
  const current = stageInfo.find((item) => item.id === stage) ?? stageInfo[2];

  if (!structured) return <RecordedQueryPlan operators={operators} query={query} executionPlan={executionPlan} record={record} />;

  return <div className="staged-query-plan">
    <nav className="plan-stage-rail" aria-label="Recorded plan stages">
      <span className="plan-stage-rail-title">PLAN VIEWS</span>
      {stageInfo.map(({ id, label, icon: Icon }) => {
        const recorded = available.has(id) && (id !== "optimizer" || optimizerRecorded);
        return <button key={id} type="button" className={stage === id ? "plan-stage active" : "plan-stage"} disabled={!recorded} onClick={() => setStage(id)} aria-current={stage === id ? "step" : undefined} title={recorded ? label : `${label}: not recorded`}><Icon size={17} /><span>{label}</span>{!recorded && <small>Not recorded</small>}</button>;
      })}
    </nav>
    <section className="plan-stage-content">
      <header className="plan-stage-heading"><div><span className="eyebrow">PLAN STAGE</span><h3>{current.label}</h3><p>{current.description}</p></div><span>{stage === "optimizer" ? "Archive evidence required" : "Recorded data only"}</span></header>
      {stage === "optimizer" && !optimizerRecorded ? <div className="plan-stage-missing"><CircleDashed size={20} /><strong>Optimizer history not recorded</strong><p>The Movie archive contains the selected logical and physical plan, but no candidate steps or optimizer snapshots.</p></div> : <RecordedQueryPlan operators={operators} query={query} executionPlan={visiblePlan} record={record} />}
    </section>
  </div>;
}
