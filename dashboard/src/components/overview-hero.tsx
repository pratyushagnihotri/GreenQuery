"use client";

import { ArrowRight } from "lucide-react";
import { useMemo, useState } from "react";
import snapshot from "@/data/sembench_measured.json";
import type { CrossSystemRow } from "@/components/cross-system";

// Engine colours: validated categorical palette (dataviz skill), fixed order; CSS variables --eng-<key> in globals.css.
// Overview hero: the SemBench findings at a glance (headline numbers from the measured snapshot, median of
// repetitions) and quality vs. energy of every engine and query for one model.

export const ENGINES = [
  { key: "stretto", label: "Stretto", light: "#2a78d6", dark: "#3987e5" },
  { key: "stretto_abacus", label: "Abacus", light: "#eb6834", dark: "#d95926" },
  { key: "lotus", label: "LOTUS", light: "#1baf7a", dark: "#199e70" },
  { key: "palimpzest", label: "Palimpzest", light: "#eda100", dark: "#c98500" },
  { key: "thalamusdb", label: "ThalamusDB", light: "#e87ba4", dark: "#d55181" },
  { key: "caesura", label: "CAESURA", light: "#008300", dark: "#008300" },
];
const OPTIMIZERS = ["stretto", "stretto_abacus"];
const QUERIES = [
  { key: "Q1", text: "five positive reviews" },
  { key: "Q2", text: "five positive reviews of one movie" },
  { key: "Q4", text: "share of positive reviews" },
  { key: "Q5", text: "ten same-sentiment pairs (join)" },
];
const W = 640, H = 250, LEFT = 44, RIGHT = 16, TOP = 12, BOTTOM = 36;

const all = (snapshot as { rows: CrossSystemRow[] }).rows;
const baseModel = (model: string) => !model.includes("operators");
const correct = (row: CrossSystemRow) => row.valid && row.status === "success" && (row.score ?? 0) >= 0.8;
const fmt = (value: number) => (value < 1 ? value.toFixed(2) : value < 10 ? value.toFixed(1) : Math.round(value).toString());

// Median-energy repetition per (system, model, query) among measured runs.
const medians = (() => {
  const groups = new Map<string, CrossSystemRow[]>();
  for (const row of all) {
    if (row.energy_source !== "measured" || !row.energy_wh || !baseModel(row.model)) continue;
    const key = `${row.system}|${row.model}|${row.query}`;
    groups.set(key, [...(groups.get(key) ?? []), row]);
  }
  return [...groups.values()].map((runs) => [...runs].sort((a, b) => a.energy_wh! - b.energy_wh!)[Math.floor((runs.length - 1) / 2)]);
})();

const headline = (() => {
  const ok = medians.filter(correct);
  let spread = { value: 0, model: "", query: "" }, beat = { value: 0, model: "", query: "", system: "" };
  const pairs = new Set(medians.map((row) => `${row.model}|${row.query}`));
  for (const pair of pairs) {
    const [model, query] = pair.split("|");
    const here = medians.filter((row) => row.model === model && row.query === query);
    const ran = here.filter((row) => row.valid && (row.status === "success" || row.reason.includes("timeout")));
    if (ran.length > 1) {
      const value = Math.max(...ran.map((r) => r.energy_wh!)) / Math.min(...ran.map((r) => r.energy_wh!));
      if (value > spread.value) spread = { value, model, query };
    }
    const opt = here.filter((row) => OPTIMIZERS.includes(row.system) && correct(row));
    const other = here.filter((row) => !OPTIMIZERS.includes(row.system) && correct(row)).sort((a, b) => a.energy_wh! - b.energy_wh!);
    if (opt.length && other.length) {
      const value = Math.min(...opt.map((r) => r.energy_wh!)) / other[0].energy_wh!;
      if (value > beat.value) beat = { value, model, query, system: other[0].system };
    }
  }
  return {
    runs: all.length,
    engines: new Set(all.map((row) => row.system)).size,
    models: new Set(all.map((row) => row.model).filter(baseModel)).size,
    minCorrect: Math.min(...ok.map((row) => row.energy_wh!)),
    maxCorrect: Math.max(...ok.map((row) => row.energy_wh!)),
    spread, beat,
    apiUsd: all.reduce((sum, row) => sum + (row.api_cost_usd ?? 0), 0),
  };
})();

const engineLabel = (key: string) => ENGINES.find((item) => item.key === key)?.label ?? key;

function Marker({ query, x, y, colour, hollow }: { query: string; x: number; y: number; colour: string; hollow: boolean }) {
  const common = { fill: hollow ? "var(--ov-surface)" : colour, stroke: hollow ? colour : "var(--ov-surface)", strokeWidth: 2 };
  if (query === "Q2") return <rect x={x - 6} y={y - 6} width={12} height={12} rx={2} {...common} />;
  if (query === "Q4") return <rect x={x - 6} y={y - 6} width={12} height={12} rx={2} transform={`rotate(45 ${x} ${y})`} {...common} />;
  if (query === "Q5") return <path d={`M${x} ${y - 7} L${x + 7} ${y + 6} L${x - 7} ${y + 6} Z`} {...common} />;
  return <circle cx={x} cy={y} r={6.5} {...common} />;
}

export function OverviewHero({ onCompare }: { onCompare?: () => void }) {
  const models = useMemo(() => [...new Set(medians.map((row) => row.model))].sort(), []);
  const [model, setModel] = useState(models.includes("qwen2.5:7b-instruct") ? "qwen2.5:7b-instruct" : models[0]);
  const [hover, setHover] = useState<CrossSystemRow | null>(null);
  const points = medians.filter((row) => row.model === model);
  const lo = Math.log10(Math.min(...medians.map((row) => row.energy_wh!)));
  const hi = Math.log10(Math.max(...medians.map((row) => row.energy_wh!)));
  const decades = Array.from({ length: Math.ceil(hi) - Math.floor(lo) + 1 }, (_, i) => Math.floor(lo) + i);
  // Rounded to 0.1 px: server and browser log10 can differ in the last digit (hydration mismatch).
  const round = (value: number) => Math.round(value * 10) / 10;
  const x = (wh: number) => round(LEFT + ((Math.log10(wh) - decades[0]) / (decades[decades.length - 1] - decades[0])) * (W - LEFT - RIGHT));
  const y = (score: number) => round(TOP + (1 - score) * (H - TOP - BOTTOM));

  return (
    <section className="panel ov-hero">
      <span className="eyebrow">SEMBENCH · {headline.engines} ENGINES · {headline.models} MODELS · {headline.runs} RUNS</span>
      <h2>Correct answers cost between {fmt(headline.minCorrect)} and {fmt(headline.maxCorrect)} Wh — the engine decides more than the model</h2>
      <div className="ov-kpis">
        <div><span>Energy spread, same query and model</span><strong>{Math.round(headline.spread.value)}×</strong>
          <small>{headline.spread.query} · {headline.spread.model}</small></div>
        <div><span>Optimizers beaten by</span><strong>up to {Math.round(headline.beat.value)}×</strong>
          <small>{engineLabel(headline.beat.system)} on {headline.beat.query} · {headline.beat.model}</small></div>
        <div><span>Correct answers (median runs)</span><strong>{medians.filter(correct).length} / {medians.length}</strong>
          <small>quality ≥ 0.8, scored by SemBench</small></div>
        <div><span>API cost tracked</span><strong>${headline.apiUsd.toFixed(0)}</strong>
          <small>billed per query (OpenAI, OpenRouter)</small></div>
      </div>

      <div className="ov-chart-head">
        <div><strong>Quality vs. energy</strong><small>every engine and query · one model · median of repetitions · log scale</small></div>
        <label>Model
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            {models.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
      </div>
      <div className="ov-chart">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Quality against energy for ${model}: ${points.length} runs`}>
          <rect x={LEFT} y={y(1)} width={W - LEFT - RIGHT} height={y(0.8) - y(1)} className="ov-band" />
          <text x={W - RIGHT - 6} y={y(1) + 14} textAnchor="end" className="ov-band-label">correct (quality ≥ 0.8)</text>
          {[0, 0.5, 1].map((score) => (
            <g key={score}>
              <line x1={LEFT} x2={W - RIGHT} y1={y(score)} y2={y(score)} className="ov-grid" />
              <text x={LEFT - 8} y={y(score) + 4} textAnchor="end" className="ov-axis">{score.toFixed(1)}</text>
            </g>
          ))}
          {decades.map((d) => (
            <text key={d} x={x(10 ** d)} y={H - BOTTOM + 18} textAnchor="middle" className="ov-axis">{10 ** d >= 1 ? 10 ** d : (10 ** d).toFixed(1)} Wh</text>
          ))}
          <text x={8} y={TOP + 4} className="ov-axis" transform={`rotate(-90 8 ${TOP + 4})`} textAnchor="end">quality</text>
          {points.map((row) => {
            const engine = ENGINES.find((item) => item.key === row.system);
            const cx = x(row.energy_wh!), cy = y(Math.max(0, Math.min(1, row.score ?? 0)));
            return (
              <g key={`${row.system}-${row.query}`} className={`ov-point${hover === row ? " hover" : ""}`}
                onMouseEnter={() => setHover(row)} onMouseLeave={() => setHover(null)}>
                <circle cx={cx} cy={cy} r={14} fill="transparent" />
                <g style={{ color: `var(--eng-${engine?.key ?? "caesura"})` }} className="ov-mark">
                  <Marker query={row.query} x={cx} y={cy} colour="currentColor" hollow={!correct(row)} />
                </g>
                <title>{`${engineLabel(row.system)} · ${row.query}: ${fmt(row.energy_wh!)} Wh, quality ${(row.score ?? 0).toFixed(2)}`}</title>
              </g>
            );
          })}
        </svg>
        <div className="ov-tip" aria-live="polite">{hover
          ? <><b>{engineLabel(hover.system)} · {hover.query}</b> {fmt(hover.energy_wh!)} Wh · quality {(hover.score ?? 0).toFixed(2)} · {hover.llm_calls?.toLocaleString()} LLM calls</>
          : "Hover a point for details. Filled = correct, hollow = not correct."}</div>
      </div>
      <div className="ov-legend">
        {ENGINES.map((item) => (
          <span key={item.key}><i style={{ background: `var(--eng-${item.key})` }} className="ov-swatch" />{item.label}</span>
        ))}
        <span className="ov-sep" />
        {QUERIES.map((item) => (
          <span key={item.key}><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <Marker query={item.key} x={7} y={7} colour="var(--ov-muted)" hollow={false} /></svg>{item.key} {item.text}</span>
        ))}
      </div>
      {onCompare && <button className="secondary-action ov-cta" onClick={onCompare}>Explore every run in Cross-System <ArrowRight size={14} /></button>}
    </section>
  );
}
