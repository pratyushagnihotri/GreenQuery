"use client";

import { Check, Clock, Crown, Minus, X } from "lucide-react";
import { type CSSProperties, useMemo, useState } from "react";
import snapshot from "@/data/sembench_measured.json";

export type CrossSystemRow = {
  system: string;
  model: string;
  query: string;
  repetition: number;
  status: string;
  valid: boolean;
  reason: string;
  quality_metric: string;
  quality: number | null;
  score: number | null;
  runtime_s: number | null;
  llm_calls: number | null;
  llm_errors: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  gpu_wh: number | null;
  gpu_dynamic_wh: number | null;
  cpu_job_wh: number | null;
  gpu_idle_w: number | null;
  energy_wh: number | null;
  energy_source: "measured" | "api";
  api_cost_usd: number | null;
  cost_source: string;
  run: string;
};

type Basis = "total" | "dynamic";

const SYSTEMS: Array<{ key: string; label: string; note: string }> = [
  { key: "stretto", label: "Stretto", note: "LLM planner + gradient-descent optimizer" },
  { key: "stretto_abacus", label: "Abacus (in Stretto)", note: "Pareto-cascade optimizer, same operators" },
  { key: "lotus", label: "LOTUS", note: "semantic operators, model cascades" },
  { key: "palimpzest", label: "Palimpzest", note: "declarative, cost-based plans" },
  { key: "thalamusdb", label: "ThalamusDB", note: "approximate processing, early stop" },
  { key: "caesura", label: "CAESURA", note: "LLM writes the plan (SQL + Python)" },
];
const QUERIES: Array<{ key: string; label: string; text: string }> = [
  { key: "Q1", label: "Q1", text: "Five clearly positive reviews (any movie)" },
  { key: "Q2", label: "Q2", text: "Five positive reviews of one movie" },
  { key: "Q4", label: "Q4", text: "Share of positive reviews of one movie" },
  { key: "Q5", label: "Q5", text: "Ten review pairs with the same sentiment (256 × 256 candidate pairs)" },
];
const COST_SOURCE: Record<string, string> = {
  provider_billed: "charged by provider",
  tokens_x_price: "tokens × price",
  list_price_no_cache: "list price, upper bound",
  mixed: "mixed sources",
};
const MODEL_COLOURS = ["#9bdc55", "#5bb8f0", "#f0a35b", "#c58af0", "#f06b8a"];

const rows = (snapshot as { rows: CrossSystemRow[] }).rows;
const wh = (value: number | null | undefined) =>
  typeof value === "number" ? `${value < 1 ? value.toFixed(2) : value < 10 ? value.toFixed(1) : value.toFixed(0)} Wh` : "–";
const energyOf = (row: CrossSystemRow, basis: Basis) =>
  basis === "total" ? row.energy_wh : row.gpu_dynamic_wh == null ? null : row.gpu_dynamic_wh + (row.cpu_job_wh ?? 0);

function outcome(row: CrossSystemRow): { tag: string; tone: "ok" | "warn" | "bad" } {
  if (row.reason.includes("timeout")) return { tag: "timeout (1 h)", tone: "bad" };
  if (!row.valid) return { tag: "invalid run", tone: "bad" };
  if (row.status !== "success") return { tag: "engine error", tone: "bad" };
  if ((row.score ?? 0) >= 0.8) return { tag: "correct", tone: "ok" };
  if ((row.score ?? 0) > 0) return { tag: "partly correct", tone: "warn" };
  return { tag: "wrong answer", tone: "bad" };
}

// Median-energy repetition of one configuration (repeated headline configurations have up to 3 runs).
function medianRow(candidates: CrossSystemRow[], basis: Basis): CrossSystemRow | undefined {
  const sorted = [...candidates].sort((a, b) => (energyOf(a, basis) ?? 0) - (energyOf(b, basis) ?? 0));
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function qualityText(row: CrossSystemRow) {
  if (row.quality == null) return "no result";
  return row.quality_metric === "relative_error" ? `rel. error ${row.quality.toFixed(2)}` : `F1 ${row.quality.toFixed(2)}`;
}

function Scatter({ query, model, models, basis, selected, onSelect }: {
  query: string; model: string; models: string[]; basis: Basis; selected: string | null; onSelect: (system: string) => void;
}) {
  // One point per system and model (median repetition); the current model is drawn last, on top.
  const groups = new Map<string, CrossSystemRow[]>();
  rows.filter((row) => row.query === query && energyOf(row, basis) != null && energyOf(row, basis)! > 0)
    .forEach((row) => groups.set(`${row.system}|${row.model}`, [...(groups.get(`${row.system}|${row.model}`) ?? []), row]));
  const points = [...groups.values()].map((runs) => medianRow(runs, basis)!)
    .sort((a, b) => Number(a.model === model) - Number(b.model === model));
  const values = points.map((row) => energyOf(row, basis)!);
  const lo = Math.log10(Math.max(Math.min(...values, 1) / 1.5, 0.01));
  const hi = Math.log10(Math.max(...values, 10) * 1.5);
  const width = 460, height = 250, left = 40, right = 70, top = 12, bottom = 34;
  const x = (value: number) => left + ((Math.log10(value) - lo) / (hi - lo)) * (width - left - right);
  const y = (score: number) => top + (1 - score) * (height - top - bottom);
  // Labels of the current model's points, shifted down in 11px steps so they do not overlap.
  const labelY = new Map<string, number>();
  const placed: Array<{ x: number; y: number }> = [];
  points.filter((row) => row.model === model)
    .sort((a, b) => energyOf(a, basis)! - energyOf(b, basis)!)
    .forEach((row) => {
      const cx = x(energyOf(row, basis)!);
      // Shift colliding labels away from the nearest axis edge: up in the lower half, down in the upper half.
      const lower = (row.score ?? 0) < 0.5;
      let ly = y(row.score ?? 0) - 6;
      while (placed.some((p) => Math.abs(p.x - cx) < 75 && Math.abs(p.y - ly) < 11)) ly += lower ? -11 : 11;
      placed.push({ x: cx, y: ly });
      labelY.set(row.system, ly);
    });
  const ticks = [];
  for (let exp = Math.ceil(lo); exp <= Math.floor(hi); exp += 1) ticks.push(10 ** exp);
  return (
    <svg className="xs-scatter" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Quality versus energy for ${query}`}>
      {[0, 0.5, 1].map((score) => (
        <g key={score}>
          <line x1={left} x2={width - right} y1={y(score)} y2={y(score)} className="xs-grid" />
          <text x={left - 6} y={y(score) + 3} textAnchor="end" className="xs-axis">{score}</text>
        </g>
      ))}
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={top} y2={height - bottom} className="xs-grid" />
          <text x={x(tick)} y={height - bottom + 14} textAnchor="middle" className="xs-axis">{tick >= 1 ? tick : tick.toFixed(1)} Wh</text>
        </g>
      ))}
      <text x={(left + width - right) / 2} y={height - 4} textAnchor="middle" className="xs-axis">energy per query (log scale) — further left is cheaper</text>
      <text x={12} y={(top + height - bottom) / 2} textAnchor="middle" className="xs-axis" transform={`rotate(-90 12 ${(top + height - bottom) / 2})`}>quality</text>
      {points.map((row) => {
        const current = row.model === model;
        const colour = MODEL_COLOURS[models.indexOf(row.model) % MODEL_COLOURS.length];
        const failed = outcome(row).tone === "bad" && (row.score ?? 0) === 0;
        const cx = x(energyOf(row, basis)!), cy = y(row.score ?? 0);
        const label = SYSTEMS.find((item) => item.key === row.system)?.label ?? row.system;
        return (
          <g key={`${row.system}-${row.model}`} className={`xs-point${current ? " current" : ""}${selected === row.system && current ? " selected" : ""}`}
            onClick={() => current && onSelect(row.system)}>
            <circle cx={cx} cy={cy} r={current ? 6 : 3.5} fill={failed ? "none" : colour} stroke={colour} strokeWidth={1.6} />
            <title>{`${label} · ${row.model}\n${qualityText(row)} · ${wh(energyOf(row, basis))} · ${outcome(row).tag}`}</title>
          </g>
        );
      })}
      {points.filter((row) => row.model === model).map((row) => {
        const cx = x(energyOf(row, basis)!), ly = labelY.get(row.system) ?? y(row.score ?? 0) - 6;
        const label = SYSTEMS.find((item) => item.key === row.system)?.label ?? row.system;
        // Labels near the right edge go to the left of their point.
        const flip = cx > width - right - 60;
        return <text key={`l-${row.system}`} x={flip ? cx - 8 : cx + 8} y={ly} textAnchor={flip ? "end" : "start"}
          className="xs-point-label">{label}</text>;
      })}
    </svg>
  );
}

export function CrossSystem() {
  const models = useMemo(() => [...new Set(rows.map((row) => row.model))].sort(), []);
  const [model, setModel] = useState(models.includes("qwen2.5:7b-instruct") ? "qwen2.5:7b-instruct" : models[0]);
  const [basis, setBasis] = useState<Basis>("total");
  const [query, setQuery] = useState("Q5");
  const [system, setSystem] = useState<string | null>("palimpzest");
  const cell = (systemKey: string, queryKey: string) =>
    medianRow(rows.filter((row) => row.system === systemKey && row.query === queryKey && row.model === model), basis);
  // Heatmap scale: log energy across this model's matrix.
  const heat = useMemo(() => {
    const values = rows.filter((row) => row.model === model).map((row) => energyOf(row, basis)).filter((v): v is number => v != null && v > 0);
    const lo = Math.log10(Math.min(...values)), hi = Math.log10(Math.max(...values));
    return { min: Math.min(...values), max: Math.max(...values),
      at: (value: number | null) => (value == null || value <= 0 || hi === lo ? 0 : (Math.log10(value) - lo) / (hi - lo)) };
  }, [model, basis]);
  const detail = system ? cell(system, query) : undefined;
  const meta = snapshot as { generated_at: string; benchmark: string; hardware: string; measurement: string; query_timeout_s: number };

  // Per-query spread among runs that finished with a correct answer vs everything that ran.
  const insights = QUERIES.map((item) => {
    const here = SYSTEMS.map((entry) => cell(entry.key, item.key))
      .filter((row): row is CrossSystemRow => row !== undefined && energyOf(row, basis) != null);
    const correct = here.filter((row) => outcome(row).tone === "ok").sort((a, b) => energyOf(a, basis)! - energyOf(b, basis)!);
    const best = [...here].filter((row) => (row.score ?? 0) > 0)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0) || energyOf(a, basis)! - energyOf(b, basis)!)[0];
    const all = here.map((row) => energyOf(row, basis)!).filter((value) => value > 0);
    return {
      ...item,
      cheapest: correct[0],
      best,
      spread: all.length > 1 ? Math.max(...all) / Math.min(...all) : null,
      wasted: here.filter((row) => outcome(row).tone === "bad").reduce((sum, row) => sum + (energyOf(row, basis) ?? 0), 0),
    };
  });

  return (
    <div className="page-stack xs-page">
      <div className="panel xs-intro">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">CROSS-SYSTEM</span>
            <h3>Same queries, same data, same model, same GPU — how much energy does each system spend for which answer?</h3>
          </div>
          <span>Snapshot {meta.generated_at}</span>
        </div>
        <div className="xs-badges">
          <span title={meta.benchmark}>SemBench Movie · sf 1000</span>
          <span title={meta.hardware}>1 × A100 per run · 4-bit local models</span>
          <span title={meta.measurement}>Energy measured (NVML + perf) for local models</span>
          {model.startsWith("gpt-") && <span className="warn">API model: quality, tokens and calls only — GPU energy is at the provider</span>}
          <span>Scored by SemBench evaluator</span>
          <span>Same logging proxy for every system</span>
          <span className="warn">Median of up to 3 repetitions</span>
          <span className="warn">Timeout {meta.query_timeout_s / 3600} h per query</span>
        </div>
        <div className="xs-controls">
          <label>Model
            <select value={model} onChange={(event) => setModel(event.target.value)}>
              {models.map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
          </label>
          <label>Energy
            <select value={basis} onChange={(event) => setBasis(event.target.value as Basis)}>
              <option value="total">Total (GPU incl. idle + CPU share)</option>
              <option value="dynamic">Above GPU idle (+ CPU share)</option>
            </select>
          </label>
        </div>
      </div>

      <div className="panel xs-matrix-panel">
        <div className="panel-heading"><div><span className="eyebrow">ENERGY HEATMAP · {model}</span><h3>Darker = more energy · outlined = cheapest correct answer per query</h3></div><span>Click a cell for details</span></div>
        <div className="xs-matrix" style={{ gridTemplateColumns: `minmax(150px, 1.3fr) repeat(${QUERIES.length}, minmax(120px, 1fr))` }}>
          <div className="xs-head" />
          {QUERIES.map((item) => (
            <button key={item.key} className={`xs-head xs-query${query === item.key ? " active" : ""}`} onClick={() => setQuery(item.key)} title={item.text}>
              <strong>{item.label}</strong><small>{item.text}</small>
            </button>
          ))}
          {SYSTEMS.map((item) => (
            <div key={item.key} className="xs-row">
              <div className="xs-system"><strong>{item.label}</strong><small>{item.note}</small></div>
              {QUERIES.map((q) => {
                const row = cell(item.key, q.key);
                if (!row) return <div key={q.key} className="xs-cell empty">not run</div>;
                const result = outcome(row);
                const level = heat.at(energyOf(row, basis));
                const cheapest = insights.find((entry) => entry.key === q.key)?.cheapest?.system === item.key;
                const Icon = result.tag.startsWith("timeout") ? Clock : result.tone === "ok" ? Check : result.tone === "warn" ? Minus : X;
                return (
                  <button key={q.key} style={{ "--heat": `${Math.round(6 + level * 80)}%` } as CSSProperties}
                    className={`xs-cell heat ${result.tone}${level > 0.55 ? " hot" : ""}${cheapest ? " cheapest" : ""}${system === item.key && query === q.key ? " active" : ""}`}
                    onClick={() => { setSystem(item.key); setQuery(q.key); }}>
                    {cheapest && <span className="xs-crown" title="cheapest correct answer"><Crown size={12} /></span>}
                    <strong>{wh(energyOf(row, basis))}</strong>
                    <small><Icon size={11} /> {result.tag}{row.score != null && result.tone !== "ok" ? ` · ${row.score.toFixed(2)}` : ""}</small>
                  </button>
                );
              })}
            </div>
          ))}
        </div>
        <div className="xs-heat-legend">
          <span>{wh(heat.min)}</span><i /><span>{wh(heat.max)}</span>
          <span className="xs-heat-key"><Check size={11} /> correct (quality ≥ 0.8)</span>
          <span className="xs-heat-key"><Minus size={11} /> partly correct</span>
          <span className="xs-heat-key"><X size={11} /> wrong or error</span>
          <span className="xs-heat-key"><Clock size={11} /> timeout — the energy was spent anyway</span>
        </div>
      </div>

      <div className="xs-split">
        <div className="panel xs-chart">
          <div className="panel-heading"><div><span className="eyebrow">QUALITY VS ENERGY</span><h3>{QUERIES.find((item) => item.key === query)?.text}</h3></div>
            <span>Large dots: {model}; small dots: other models; hollow: no correct answer</span></div>
          <Scatter query={query} model={model} models={models} basis={basis} selected={system} onSelect={setSystem} />
        </div>
        <div className="panel xs-detail">
          <div className="panel-heading"><div><span className="eyebrow">RUN DETAILS</span><h3>{detail ? `${SYSTEMS.find((item) => item.key === detail.system)?.label} · ${detail.query}` : "Select a cell"}</h3></div>
            {detail && <span className={`xs-source ${detail.energy_source}`}>{detail.energy_source === "api" ? "API · GPU energy not measured" : "measured"}</span>}</div>
          {detail && (
            <dl>
              <dt>Outcome</dt><dd>{outcome(detail).tag}{detail.reason ? ` — ${detail.reason.slice(0, 140)}` : ""}</dd>
              <dt>Quality</dt><dd>{qualityText(detail)}</dd>
              <dt>Runtime</dt><dd>{detail.runtime_s?.toLocaleString()} s</dd>
              <dt>LLM calls</dt><dd>{detail.llm_calls?.toLocaleString()}{detail.llm_errors ? ` (${detail.llm_errors} failed)` : ""}</dd>
              <dt>Tokens in / out</dt><dd>{detail.prompt_tokens?.toLocaleString()} / {detail.completion_tokens?.toLocaleString()}</dd>
              <dt>GPU energy</dt><dd>{wh(detail.gpu_wh)} (above idle {wh(detail.gpu_dynamic_wh)}, idle {detail.gpu_idle_w ?? "–"} W)</dd>
              <dt>CPU energy (job share)</dt><dd>{wh(detail.cpu_job_wh)}</dd>
              <dt>API cost</dt><dd>{detail.api_cost_usd ? `$${detail.api_cost_usd.toFixed(detail.api_cost_usd < 0.1 ? 4 : 2)} (${COST_SOURCE[detail.cost_source] ?? detail.cost_source})` : "none (local model)"}</dd>
              <dt>Raw run</dt><dd className="mono">{detail.run}</dd>
            </dl>
          )}
        </div>
      </div>

      <div className="panel xs-insights">
        <div className="panel-heading"><div><span className="eyebrow">WHAT THE MATRIX SHOWS</span><h3>Per query, {model}</h3></div></div>
        <div className="xs-insight-grid">
          {insights.map((item) => (
            <div key={item.key} className="xs-insight">
              <strong>{item.label}</strong>
              <p>{item.cheapest ? <>Cheapest correct answer: <b>{SYSTEMS.find((s) => s.key === item.cheapest!.system)?.label}</b>, {wh(energyOf(item.cheapest, basis))}.</> : item.best ? <>No fully correct answer. Best: <b>{SYSTEMS.find((s) => s.key === item.best!.system)?.label}</b> ({qualityText(item.best)}), {wh(energyOf(item.best, basis))}.</> : "No system returned a usable answer."}</p>
              <p>{item.spread ? <>Energy differs <b>{item.spread >= 10 ? Math.round(item.spread) : item.spread.toFixed(1)}×</b> between systems.</> : ""}</p>
              <p>{item.wasted > 0 ? <>Spent on wrong answers, errors and timeouts: <b>{wh(item.wasted)}</b>.</> : "No energy spent on failed runs."}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
