import { linkHorizontal, linkVertical } from "d3-shape";
import { select } from "d3-selection";
import { zoom, zoomIdentity, type ZoomBehavior } from "d3-zoom";
import { CircleDashed, Database, Eye, GitBranch, Minus, Plus, RotateCcw, Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { primaryQuality, queryText, type BenchmarkRecord, type PlanEdge, type PlanStep, type StructuredExecutionPlan } from "@/data/benchmarks";

const meaning = (operator: string, query: string) => {
  if (/ImageQaFilter/i.test(operator)) return "Vision check: " + queryText(query);
  if (/TextQa/i.test(operator)) return "Text answer extraction: " + queryText(query);
  if (/Project/i.test(operator)) return "Shape the recorded result columns";
  if (/GroupBy|Aggregate/i.test(operator)) return "Group and aggregate recorded values";
  if (/Join/i.test(operator)) return "Combine recorded input relations";
  if (/Limit/i.test(operator)) return "Restrict the recorded result set";
  return operator;
};

function A2ExecutionNarrative({ operators }: { operators: string[] }) {
  const selected = operators.find((operator) => /ImageQaFilter/i.test(operator)) ?? "ImageQaFilter";
  const visionModel = selected.match(/ImageQABackend-(.+?)-cr/i)?.[1] ?? "recorded vision backend";
  return <div className="execution-narrative" aria-label="Artwork A2 semantic execution narrative">
    <div className="narrative-node input"><Database size={17} /><div><span>DATASET / INPUT</span><strong>Artwork benchmark input</strong><small>Recorded dataset boundary</small></div></div>
    <div className="narrative-arrow" />
    <div className="narrative-node caesura"><Sparkles size={17} /><div><span>CAESURA SEMANTIC OPERATION</span><strong>Filter artworks that show more than two people</strong><small>Planner workload is shown beside the graph</small></div></div>
    <div className="narrative-arrow" />
    <div className="choice-stage"><div className="narrative-node choice"><GitBranch size={17} /><div><span>PHYSICAL IMPLEMENTATION CHOICE</span><strong>Selected implementation is recorded</strong><small>Candidate search details were not persisted</small></div></div><div className="unrecorded-branch"><CircleDashed size={15} /> Alternative paths not recorded</div></div>
    <div className="narrative-arrow" />
    <div className="narrative-node stretto"><Eye size={17} /><div><span>STRETTO EXECUTION</span><strong>Vision check: does the artwork show more than two people?</strong><small>Selected: ImageQaFilter · {visionModel}</small></div></div>
    <div className="narrative-arrow" />
    <div className="narrative-node output"><Database size={17} /><div><span>MEASURED RESULT</span><strong>Result and quality are recorded beside the graph</strong><small>Full-query measurements only; no per-operator allocation</small></div></div>
    <p className="operator-provenance">Recorded operator sequence: {operators.join(" → ")}</p>
  </div>;
}

type StepPresentation = {
  badge: string;
  title: string;
  summary: string;
  input: string;
  output: string;
  implementation: string;
  facts: Array<{ label: string; value: string }>;
};

const stepExplanation = (step: Partial<PlanStep>) => step.explanation ?? step.expression ?? "No step explanation was recorded.";
const valueText = (value: unknown, suffix = "") => typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) + suffix : "Not recorded";

type GraphKind = "input" | "logical" | "physical" | "result";
type GraphSelection = { kind: GraphKind; id: string };
type GraphNode = { id: string; kind: GraphKind; x: number; y: number; width: number; height: number; step?: PlanStep; presentation: StepPresentation };
type GraphLink = { id: string; from: string; to: string; kind: "data" | "implementation"; label?: string };
type GraphModel = { width: number; height: number; nodes: GraphNode[]; links: GraphLink[] };
type Point = { x: number; y: number };

const nodeId = (kind: GraphKind, id: string) => kind + ":" + id;

const titleCase = (value: string) => value
  .replace(/([a-z])([A-Z])/g, "$1 $2")
  .replace(/[_-]+/g, " ")
  .replace(/\s+/g, " ")
  .trim()
  .replace(/\b\w/g, (letter) => letter.toUpperCase());

const fieldLabel = (value: string) => {
  const field = value.split(".").pop()?.replace(/[{}[\]'"]/g, "") ?? value;
  const compact = field.toLowerCase();
  if (compact.endsWith("id") && compact.length > 2) return titleCase(compact.slice(0, -2)) + " ID";
  if (compact.endsWith("text") && compact.length > 4) return titleCase(compact.slice(0, -4)) + " text";
  const words = field.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/[_-]+/g, " ").trim().toLowerCase();
  return words.replace(/\bid\b/g, "ID");
};

const cleanRelation = (value: string) => {
  const withoutUuid = value.replace(/_[0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12}/gi, "");
  return titleCase(withoutUuid).replace(/\bReviews\b/i, "reviews");
};

const extractFields = (value?: string) => [...(value ?? "").matchAll(/[A-Za-z_][\w-]*\.([A-Za-z_][\w-]*)/g)].map((match) => fieldLabel(match[1]));
const readableExpression = (value: string) => value
  .replace(/[{}]/g, "")
  .replace(/[A-Za-z_][\w-]*_[0-9a-f]{8}(?:_[0-9a-f]{4}){3}_[0-9a-f]{12}/gi, (relation) => cleanRelation(relation))
  .replace(/\b([A-Za-z_][\w-]*)\.([A-Za-z_][\w-]*)\b/g, (_, relation, field) => (/^reviews(?:_|$)/i.test(relation) ? "" : cleanRelation(relation) + " ") + fieldLabel(field))
  .replace(/^Filter the data by:\s*/i, "")
  .replace(/^Project the data by:\s*/i, "")
  .replace(/^Keep\s*/i, "")
  .trim();
const lowerFirst = (value: string) => value ? value.charAt(0).toLowerCase() + value.slice(1) : value;

function operatorKind(step: Partial<PlanStep>) {
  const value = step.operator ?? step.type ?? "";
  if (/TextQa|ImageQa|QaBackend|vision|model/i.test(value)) return "MODEL-BACKED";
  if (/Filter/i.test(value)) return "FILTER";
  if (/Project|Rename/i.test(value)) return "PROJECT";
  if (/Join/i.test(value)) return "JOIN";
  if (/GroupBy/i.test(value)) return "GROUP";
  if (/Aggregate/i.test(value)) return "AGGREGATE";
  if (/Limit/i.test(value)) return "LIMIT";
  return "OPERATOR";
}

function operatorTitle(step: Partial<PlanStep>) {
  const value = step.operator ?? step.type ?? "Recorded step";
  if (/TextQa|ImageQa|QaBackend|vision|model/i.test(value)) return "Model-backed " + (/ImageQa|vision/i.test(value) ? "vision check" : "text filter");
  if (/LogicalFilter/i.test(value)) return "Logical filter";
  if (/LogicalProject/i.test(value)) return "Logical projection";
  if (/LogicalLimit/i.test(value)) return "Result limit";
  if (/LogicalJoin/i.test(value)) return "Logical join";
  if (/LogicalGroup/i.test(value)) return "Logical grouping";
  if (/GroupBy/i.test(value)) return "Group results";
  if (/Aggregate/i.test(value)) return "Aggregate results";
  if (/Project/i.test(value)) return "Projection";
  if (/Rename/i.test(value)) return "Rename fields";
  if (/Join/i.test(value)) return "Join relations";
  if (/Limit/i.test(value)) return "Result limit";
  return titleCase(value);
}

function relationAliases(plan: StructuredExecutionPlan, record?: BenchmarkRecord) {
  const aliases = new Map<string, string>();
  const steps = [...(plan.logical?.steps ?? []), ...(plan.physical?.steps ?? [])];
  const defaultInput = record?.dataset === "Movie" ? "Movie reviews" : (record?.dataset ? record.dataset + " records" : "Recorded input");
  for (const step of steps) {
    for (const input of step.inputs ?? []) if (!aliases.has(input)) aliases.set(input, input === "reviews" ? defaultInput : cleanRelation(input));
  }
  const ordered = plan.logical?.steps ?? plan.physical?.steps ?? [];
  for (const step of ordered) {
    const input = step.inputs?.[0];
    const inputLabel = input ? aliases.get(input) ?? cleanRelation(input) : defaultInput;
    const kind = operatorTitle(step);
    let output = inputLabel;
    if (/filter|check/i.test(kind)) output = "Filtered " + inputLabel.toLowerCase();
    else if (/project|rename/i.test(kind)) output = extractFields(step.expression).join(" and ") || "Selected fields";
    else if (/limit/i.test(kind)) output = "Limited result set";
    else if (/join/i.test(kind)) output = "Joined relations";
    else if (/group|aggregate/i.test(kind)) output = "Summarized results";
    if (step.output) aliases.set(step.output, output);
  }
  return { aliases, defaultInput };
}

function readableValue(key: string, value: unknown) {
  if (key === "context" && typeof value === "string") return fieldLabel(value);
  if (key === "keep_columns" && typeof value === "string") return extractFields(value).join(", ") || "Recorded columns";
  if (typeof value === "string") return value.replace(/[{}]/g, "");
  return String(value);
}

function presentStep(step: PlanStep, record: BenchmarkRecord | undefined, aliases: Map<string, string>, defaultInput: string): StepPresentation {
  const title = operatorTitle(step);
  const badge = operatorKind(step);
  const input = step.inputs?.map((item) => aliases.get(item) ?? (item === "reviews" ? defaultInput : cleanRelation(item))).join(" and ") || "Recorded input";
  const output = step.output ? aliases.get(step.output) ?? cleanRelation(step.output) : "Recorded output";
  const expression = step.expression ?? (typeof step.operator_config?.__expression__ === "string" ? step.operator_config.__expression__ : "");
  const fields = extractFields(expression);
  const config = step.operator_config ?? {};
  const model = step.operator?.match(/(?:Backend|backend)-(.+)$/)?.[1] ?? "";
  const question = typeof config.question_template === "string" ? config.question_template : "";
  let summary = stepExplanation(step);
  if (/Filter/i.test(title) && expression) summary = "Filter " + input.toLowerCase() + " where " + lowerFirst(readableExpression(expression).replace(/\.$/, "")) + ".";
  if (/Project|Projection/i.test(title) && fields.length) summary = "Keep " + lowerFirst(fields.join(" and ")) + " from each matching record.";
  if (/Limit/i.test(title)) summary = "Keep the first " + (config.num_remaining ?? expression.match(/\d+/)?.[0] ?? "recorded") + " matching results.";
  if (/Model-backed/i.test(title)) summary = question ? "Ask " + (model || "the recorded model") + ': "' + question + '" and keep the recorded answer.' : "Apply the recorded model-backed predicate.";
  if (/Group|Aggregate/i.test(title)) summary = "Summarize matching records using the recorded grouping or aggregate operation.";
  if (/Join/i.test(title)) summary = "Combine the recorded input relations using the selected join implementation.";
  const facts = Object.entries(config)
    .filter(([key]) => key !== "__expression__")
    .slice(0, 8)
    .map(([key, value]) => ({ label: titleCase(key), value: readableValue(key, value) }));
  if (record?.model && /Model-backed/i.test(title) && !facts.some((fact) => fact.label === "Model")) facts.unshift({ label: "Model", value: record.model });
  return { badge, title, summary, input, output, implementation: title, facts };
}

function topologicalSteps(steps: PlanStep[], edges: PlanEdge[]) {
  const incoming = new Map(steps.map((step) => [step.id, 0]));
  for (const edge of edges) if (incoming.has(edge.to)) incoming.set(edge.to, (incoming.get(edge.to) ?? 0) + 1);
  const result: PlanStep[] = [];
  const queue = steps.filter((step) => incoming.get(step.id) === 0);
  while (queue.length) {
    const current = queue.shift();
    if (!current) break;
    result.push(current);
    for (const edge of edges.filter((item) => item.from === current.id)) {
      const count = (incoming.get(edge.to) ?? 0) - 1;
      incoming.set(edge.to, count);
      if (count === 0) {
        const next = steps.find((step) => step.id === edge.to);
        if (next) queue.push(next);
      }
    }
  }
  return result.length === steps.length ? result : steps;
}

function buildGraph(plan: StructuredExecutionPlan, record: BenchmarkRecord | undefined, structured: boolean): GraphModel {
  const logicalSteps = topologicalSteps(plan.logical?.steps ?? [], plan.logical?.edges ?? []);
  const physicalSteps = topologicalSteps(plan.physical?.steps ?? [], plan.physical?.edges ?? []);
  const nodeWidth = structured ? 264 : 360;
  const top = 112;
  const rowGap = 36;
  const maxRows = Math.max(logicalSteps.length, physicalSteps.length, 1);
  const height = Math.max(640, top + maxRows * 128 + rowGap * Math.max(maxRows - 1, 0) + 160);
  const width = structured ? 930 : 760;
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const logicalById = new Map(logicalSteps.map((step) => [step.id, step]));
  const { aliases, defaultInput } = relationAliases(plan, record);
  const inputPresentation: StepPresentation = { badge: "INPUT", title: defaultInput, summary: "Recorded dataset relation entering the plan.", input: "Benchmark dataset", output: defaultInput, implementation: "Dataset input", facts: [] };
  const resultPresentation: StepPresentation = { badge: "RESULT", title: "Measured result", summary: record?.query ? queryText(record.query) : "Recorded full-query output", input: "Final execution step", output: "Full-query result", implementation: "Recorded query result", facts: [] };
  nodes.push({ id: "input", kind: "input", x: (width - 260) / 2, y: 22, width: 260, height: 86, presentation: inputPresentation });
  logicalSteps.forEach((step, index) => nodes.push({ id: nodeId("logical", step.id), kind: "logical", x: structured ? 28 : (width - nodeWidth) / 2, y: top + index * (128 + rowGap), width: nodeWidth, height: 128, step, presentation: presentStep(step, record, aliases, defaultInput) }));
  physicalSteps.forEach((step, index) => {
    const logicalStep = step.logical_step_id ? logicalById.get(step.logical_step_id) : undefined;
    const presentation = presentStep(step, record, aliases, defaultInput);
    if (stepExplanation(step) === "No step explanation was recorded." && logicalStep) presentation.summary = "Implement the logical step: " + lowerFirst(presentStep(logicalStep, record, aliases, defaultInput).summary);
    nodes.push({ id: nodeId("physical", step.id), kind: "physical", x: structured ? width - nodeWidth - 28 : (width - nodeWidth) / 2, y: top + index * (128 + rowGap), width: nodeWidth, height: 128, step, presentation });
  });
  const resultY = top + maxRows * (128 + rowGap) - rowGap + 58;
  nodes.push({ id: "result", kind: "result", x: (width - 300) / 2, y: resultY, width: 300, height: 90, presentation: resultPresentation });
  const addDataLinks = (steps: PlanStep[], kind: "logical" | "physical") => {
    const edges = plan[kind]?.edges ?? [];
    for (const edge of edges) links.push({ id: kind + "-" + edge.from + "-" + edge.to, from: nodeId(kind, edge.from), to: nodeId(kind, edge.to), kind: "data", label: edge.kind === "data_flow" ? "Recorded data flow" : edge.kind });
    const roots = new Set(steps.map((step) => step.id));
    for (const edge of edges) roots.delete(edge.to);
    for (const root of roots) links.push({ id: kind + "-input-" + root, from: "input", to: nodeId(kind, root), kind: "data" });
    const sinks = new Set(steps.map((step) => step.id));
    for (const edge of edges) sinks.delete(edge.from);
    for (const sink of sinks) links.push({ id: kind + "-" + sink + "-result", from: nodeId(kind, sink), to: "result", kind: "data" });
  };
  addDataLinks(logicalSteps, "logical");
  addDataLinks(physicalSteps, "physical");
  for (const edge of plan.edges ?? []) links.push({ id: "implementation-" + edge.from + "-" + edge.to, from: nodeId("logical", edge.from), to: nodeId("physical", edge.to), kind: "implementation", label: "Recorded implementation link" });
  return { width, height, nodes, links };
}

const wrap = (text: string, max = 35) => {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    if ((line + " " + word).trim().length > max && line) {
      lines.push(line);
      line = word;
    } else line = (line + " " + word).trim();
  }
  if (line) lines.push(line);
  return lines.slice(0, 4);
};

function GraphNodeView({ node, selected, onSelect }: { node: GraphNode; selected: boolean; onSelect: () => void }) {
  const lines = wrap(node.presentation.summary);
  return <g className={"d3-plan-node " + node.kind + (selected ? " selected" : "")} transform={"translate(" + node.x + "," + node.y + ")"} role="button" tabIndex={0} aria-label={node.presentation.title + ": " + node.presentation.summary} aria-pressed={selected} onClick={onSelect} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(); } }}>
    <rect width={node.width} height={node.height} rx={node.kind === "input" || node.kind === "result" ? 14 : 7} />
    <text className="d3-node-kind" x="16" y="20">{node.kind === "input" ? "DATASET INPUT" : node.kind === "result" ? "MEASURED RESULT" : node.kind === "logical" ? "LOGICAL PLAN" : "PHYSICAL EXECUTION"}</text>
    <rect className="d3-node-badge-bg" x={node.width - 88} y="10" width="72" height="18" rx="9" />
    <text className="d3-node-badge" x={node.width - 52} y="22">{node.presentation.badge}</text>
    <text className="d3-node-title" x="16" y="48">{wrap(node.presentation.title, 38)[0]}</text>
    {lines.map((line, index) => <text className="d3-node-summary" key={line + index} x="16" y={69 + index * 14}>{line}</text>)}
  </g>;
}

function Inspector({ graph, selection, record, structured }: { graph: GraphModel; selection: GraphSelection; record?: BenchmarkRecord; structured: boolean }) {
  const node = graph.nodes.find((item) => item.id === selection.id);
  const linkedLogical = node?.kind === "physical" && node.step?.logical_step_id ? graph.nodes.find((item) => item.id === nodeId("logical", node.step?.logical_step_id ?? "")) : undefined;
  const linkedPhysical = node?.kind === "logical" ? graph.nodes.find((item) => item.kind === "physical" && item.step?.logical_step_id === node.step?.id) : undefined;
  const quality = record ? primaryQuality(record) : null;
  const phaseRecorded = typeof record?.planner_energy_wh === "number" && typeof record?.execution_energy_wh === "number";
  const title = node?.presentation.title ?? "Selected node";
  const implementation = node?.kind === "physical" ? (linkedLogical?.presentation.title ? "Selected physical implementation of " + linkedLogical.presentation.title : "Recorded physical operator") : node?.kind === "logical" ? (linkedPhysical?.presentation.title ? "Implemented by " + linkedPhysical.presentation.title : "No physical counterpart recorded") : node?.presentation.implementation ?? "Recorded plan boundary";
  return <aside className="d3-plan-inspector" aria-live="polite">
    <div className="d3-inspector-heading"><div><span className="eyebrow">SELECTED NODE</span><h4>{title}</h4></div><span>{node?.presentation.badge ?? node?.kind ?? "result"}</span></div>
    {!node && <p className="d3-inspector-missing">Select a graph node to inspect its recorded details.</p>}
    {node?.kind === "input" && <><section className="d3-inspector-section"><span className="d3-inspector-label">Purpose</span><p>{node.presentation.summary}</p></section><dl className="d3-readable-facts"><div><dt>Dataset</dt><dd>{record?.dataset ?? "Not recorded"}</dd></div><div><dt>Relation</dt><dd>{node.presentation.output}</dd></div></dl></>}
    {node?.kind === "result" && <><section className="d3-inspector-section"><span className="d3-inspector-label">Purpose</span><p>Full-query measurements for the selected run. Energy and latency are not allocated to individual operators.</p></section><div className="d3-evidence-grid"><div><span>Runtime</span><strong>{valueText(record?.runtime_seconds, " s")}</strong></div><div><span>Energy</span><strong>{valueText(record?.energy_wh, " Wh")}</strong></div><div><span>CO2</span><strong>{valueText(record?.co2_g, " g")}</strong></div><div><span>Tokens</span><strong>{valueText(record?.total_tokens)}</strong></div><div><span>Quality</span><strong>{quality ? quality.label + ": " + valueText(quality.value) : "Not recorded"}</strong></div><div><span>Phase evidence</span><strong>{phaseRecorded ? "Recorded" : "Not recorded"}</strong></div></div>{phaseRecorded && <p className="d3-inspector-note">Planner {valueText(record?.planner_energy_wh, " Wh")} · Execution {valueText(record?.execution_energy_wh, " Wh")}</p>}</>}
    {node?.step && <><section className="d3-inspector-section"><span className="d3-inspector-label">Purpose</span><p>{node.presentation.summary}</p></section><dl className="d3-readable-facts"><div><dt>Input</dt><dd>{node.presentation.input}</dd></div><div><dt>Output</dt><dd>{node.presentation.output}</dd></div><div><dt>Implementation</dt><dd>{implementation}</dd></div>{node.presentation.facts.map((fact) => <div key={fact.label}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}</dl>{!structured && <p className="d3-inspector-missing">Structured plan not recorded. This is the recorded physical operator sequence.</p>}<details className="d3-technical-details"><summary>Recorded technical details</summary><dl><div><dt>Exact operator</dt><dd>{node.step.operator ?? node.step.type ?? "Not recorded"}</dd></div><div><dt>Internal step ID</dt><dd>{node.step.id}</dd></div>{node.step.logical_step_id && <div><dt>Logical step ID</dt><dd>{node.step.logical_step_id}</dd></div>}</dl><pre>{JSON.stringify(node.step, null, 2)}</pre></details></>}
  </aside>;
}

function ExecutionGraph({ plan, record, structured }: { plan: StructuredExecutionPlan; record?: BenchmarkRecord; structured: boolean }) {
  const graph = useMemo(() => buildGraph(plan, record, structured), [plan, record, structured]);
  const [selection, setSelection] = useState<GraphSelection>({ kind: "result", id: "result" });
  const svgRef = useRef<SVGSVGElement>(null);
  const viewportRef = useRef<SVGGElement>(null);
  const zoomRef = useRef<ZoomBehavior<SVGSVGElement, unknown> | null>(null);
  useEffect(() => {
    if (!svgRef.current || !viewportRef.current) return;
    const svg = select<SVGSVGElement, unknown>(svgRef.current);
    const viewport = select<SVGGElement, unknown>(viewportRef.current);
    const behavior = zoom<SVGSVGElement, unknown>().scaleExtent([0.55, 2.4]).on("zoom", (event) => viewport.attr("transform", event.transform.toString()));
    zoomRef.current = behavior;
    svg.call(behavior);
    return () => { svg.on(".zoom", null); zoomRef.current = null; };
  }, [graph.width, graph.height]);
  const applyZoom = (factor: number) => { if (svgRef.current && zoomRef.current) select<SVGSVGElement, unknown>(svgRef.current).call(zoomRef.current.scaleBy, factor); };
  const resetView = () => { if (svgRef.current && zoomRef.current) select<SVGSVGElement, unknown>(svgRef.current).call(zoomRef.current.transform, zoomIdentity); };
  const fitView = () => { if (!svgRef.current || !zoomRef.current) return; const bounds = svgRef.current.getBoundingClientRect(); const scale = Math.min((bounds.width - 24) / graph.width, (bounds.height - 24) / graph.height, 1); const x = (bounds.width - graph.width * scale) / 2; const y = (bounds.height - graph.height * scale) / 2; select<SVGSVGElement, unknown>(svgRef.current).call(zoomRef.current.transform, zoomIdentity.translate(x, y).scale(scale)); };
  const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
  return <div className="d3-plan-shell" aria-label="Recorded logical and physical execution plan">
    <div className="d3-plan-header"><div><span className="eyebrow">RECORDED EXECUTION PLAN</span><h3>Query plan graph</h3><p>{structured ? "Recorded logical intent mapped to the selected physical execution." : "Recorded physical operator sequence."}</p>{!structured && typeof record?.legacy_plan_status === "string" && <p className="d3-inspector-missing">{record.legacy_plan_status}</p>}</div><span>schema v{plan.schema_version ?? "1"} · {structured ? (plan.edges ?? []).length + " implementation links" : "summary-only plan"}</span></div>
    <div className="d3-plan-workspace">
      <section className="d3-plan-canvas" aria-label="Execution graph canvas"><div className="d3-plan-toolbar"><span>Solid = data flow · Dashed = selected implementation · Select a node for details</span><div><button type="button" aria-label="Zoom out" title="Zoom out" onClick={() => applyZoom(0.8)}><Minus size={15} /></button><button type="button" aria-label="Zoom in" title="Zoom in" onClick={() => applyZoom(1.25)}><Plus size={15} /></button><button type="button" onClick={fitView}>Fit plan</button><button type="button" aria-label="Reset view" title="Reset view" onClick={resetView}><RotateCcw size={15} /></button></div></div><svg ref={svgRef} className="d3-plan-svg" viewBox={"0 0 " + graph.width + " " + graph.height} role="img" aria-label="Interactive execution plan graph"><defs><marker id="d3-plan-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" /></marker></defs><g ref={viewportRef}>{graph.links.map((link) => { const source = nodeMap.get(link.from); const target = nodeMap.get(link.to); if (!source || !target) return null; const sameLane = Math.abs(source.x - target.x) < 10; const sourcePoint = { x: source.x + source.width / 2, y: source.y + source.height }; const targetPoint = { x: target.x + target.width / 2, y: target.y }; const path = sameLane ? linkVertical<{ source: Point; target: Point }, Point>().x((point) => point.x).y((point) => point.y)({ source: sourcePoint, target: targetPoint }) : linkHorizontal<{ source: Point; target: Point }, Point>().x((point) => point.x).y((point) => point.y)({ source: { x: source.x + (source.x < target.x ? source.width : 0), y: source.y + source.height / 2 }, target: { x: target.x + (source.x < target.x ? 0 : target.width), y: target.y + target.height / 2 } }); return <g className={"d3-plan-link " + (link.kind === "implementation" ? "implementation" : "data")} key={link.id}><path d={path ?? undefined} markerEnd="url(#d3-plan-arrow)" /><title>{link.label ?? (link.kind === "implementation" ? "selected implementation" : "data flow")}</title></g>; })}{graph.nodes.map((node) => <GraphNodeView key={node.id} node={node} selected={selection.id === node.id} onSelect={() => setSelection({ kind: node.kind, id: node.id })} />)}</g></svg></section>
      <Inspector graph={graph} selection={selection} record={record} structured={structured} />
    </div>
    <p className="d3-plan-footnote">Solid arrows show recorded data flow. Dashed arrows show recorded logical-to-physical implementation links. Measurements belong to the full query.</p>
  </div>;
}

export function QueryPlan({ operators, query, executionPlan, record }: { operators?: string[]; query: string; executionPlan?: StructuredExecutionPlan; record?: BenchmarkRecord }) {
  const graphKey = record ? record.dataset + "-" + record.query + "-" + record.model + "-" + record.run : query;
  if (executionPlan?.logical?.steps.length || executionPlan?.physical?.steps.length) return <ExecutionGraph key={graphKey} plan={executionPlan} record={record} structured />;
  if (query === "A2_more_than_two_people" && operators?.length) return <A2ExecutionNarrative operators={operators} />;
  if (!operators?.length) return <div className="empty-plan">No physical operators were recorded for this run.</div>;
  const fallbackPlan: StructuredExecutionPlan = { physical: { steps: [...operators].reverse().map((operator, index) => ({ id: "physical_" + index, operator, explanation: meaning(operator, query) })), edges: operators.slice(1).map((_, index) => ({ from: "physical_" + index, to: "physical_" + (index + 1), kind: "data_flow" })) } };
  return <ExecutionGraph key={graphKey} plan={fallbackPlan} record={record} structured={false} />;
}
