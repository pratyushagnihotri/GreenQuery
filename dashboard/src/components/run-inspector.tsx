"use client";

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

// Run inspector (demo mode 2): GPU power over time, idle baseline, when LLM calls were in flight, and the
// first calls of one query window. Recorded runs load a condensed trace bundled with the dashboard
// (public/inspector, evaluation/scripts/export_run_traces.py); live runs ask the GreenQuery service.

export type InspectSource =
  | { kind: "recorded"; run: string; system: string; query: string; repetition: number }
  | { kind: "live"; jobId: string; system: string; query: string };

type Point = { t: number; w: number | null; busy: number; calls: number; tok: number };
type Call = { start: number; dur: number; in: number | null; out: number | null; status: number | null; model: string | null };
type Window = {
  system: string; query: string; repetition: number; duration_s: number; step_s: number; idle_w: number | null;
  calls_total: number; points: Point[]; first_calls: Call[];
};

const W = 680, H = 200, LEFT = 44, RIGHT = 12, TOP = 10, POWER_H = 120, STRIP_TOP = 146, STRIP_H = 10, CALLS_TOP = 164, CALLS_H = 26;

async function load(source: InspectSource): Promise<Window> {
  if (source.kind === "recorded") {
    const file = `${source.run}__${source.system}__${source.query}__rep${source.repetition}.json`;
    const response = await fetch(`/inspector/${encodeURIComponent(file)}`);
    if (!response.ok) throw new Error("No energy trace was recorded for this run.");
    return (await response.json()) as Window;
  }
  const response = await fetch(`${API_BASE}/api/runs/${source.jobId}/trace`);
  if (!response.ok) throw new Error("The trace of this live run is not available.");
  const body = (await response.json()) as { windows: Window[] };
  const match = body.windows.find((w) => w.system === source.system && w.query === source.query) ?? body.windows[0];
  if (!match) throw new Error("The trace of this live run is empty.");
  return match;
}

export function RunInspector({ source }: { source: InspectSource }) {
  const [data, setData] = useState<Window | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    load(source).then((w) => { if (alive) { setData(w); setError(""); } })
      .catch((reason: unknown) => { if (alive) setError(reason instanceof Error ? reason.message : String(reason)); });
    return () => { alive = false; };
  }, [source]);

  if (error) return <p className="ch-note">{error}</p>;
  if (!data) return <p className="ch-note">Loading the energy trace…</p>;

  const points = data.points;
  const maxW = Math.max(100, ...points.map((p) => p.w ?? 0), data.idle_w ?? 0) * 1.1;
  const maxCalls = Math.max(1, ...points.map((p) => p.calls));
  const x = (t: number) => LEFT + (t / Math.max(data.duration_s, 1)) * (W - LEFT - RIGHT);
  const y = (w: number) => TOP + POWER_H - (w / maxW) * POWER_H;
  const barW = Math.max(1, (W - LEFT - RIGHT) / points.length);
  const line = points.filter((p) => p.w != null).map((p) => `${x(p.t + data.step_s / 2).toFixed(1)},${y(p.w!).toFixed(1)}`).join(" ");
  const busyShare = points.length ? points.reduce((sum, p) => sum + p.busy, 0) / points.length : 0;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * data.duration_s));
  const fmtT = (s: number) => (data.duration_s > 180 ? `${Math.round(s / 60)} min` : `${s} s`);

  return (
    <div className="ri">
      <div className="ch-metrics small">
        <div><span>Duration</span><strong>{data.duration_s > 180 ? `${(data.duration_s / 60).toFixed(1)} min` : `${data.duration_s} s`}</strong></div>
        <div><span>LLM calls</span><strong>{data.calls_total.toLocaleString()}</strong></div>
        <div><span>GPU busy with LLM calls</span><strong>{Math.round(busyShare * 100)} % of the time</strong></div>
        <div><span>Idle baseline</span><strong>{data.idle_w != null ? `${Math.round(data.idle_w)} W` : "–"}</strong></div>
      </div>
      <svg className="ri-chart" viewBox={`0 0 ${W} ${H + 24}`} role="img"
        aria-label={`GPU power over ${Math.round(data.duration_s)} seconds with ${data.calls_total} LLM calls`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={LEFT} x2={W - RIGHT} y1={y(maxW * f)} y2={y(maxW * f)} className="ri-grid" />
            <text x={LEFT - 6} y={y(maxW * f) + 3} textAnchor="end" className="ri-axis">{Math.round(maxW * f)} W</text>
          </g>
        ))}
        {data.idle_w != null && (
          <g>
            <line x1={LEFT} x2={W - RIGHT} y1={y(data.idle_w)} y2={y(data.idle_w)} className="ri-idle" />
            <text x={W - RIGHT} y={y(data.idle_w) - 4} textAnchor="end" className="ri-axis">idle {Math.round(data.idle_w)} W</text>
          </g>
        )}
        <polyline points={line} className="ri-power" />
        <text x={LEFT - 6} y={STRIP_TOP + 8} textAnchor="end" className="ri-axis">calls</text>
        {points.map((p, i) => (
          <rect key={`b${i}`} x={LEFT + i * barW} y={STRIP_TOP} width={barW + 0.3} height={STRIP_H} className="ri-busy"
            style={{ opacity: p.busy }} />
        ))}
        {points.map((p, i) => p.calls > 0 && (
          <rect key={`c${i}`} x={LEFT + i * barW} y={CALLS_TOP + CALLS_H - (p.calls / maxCalls) * CALLS_H}
            width={Math.max(barW - 0.5, 0.8)} height={(p.calls / maxCalls) * CALLS_H} className="ri-calls">
            <title>{`${p.calls} calls started at ${p.t} s`}</title>
          </rect>
        ))}
        {ticks.map((t) => <text key={t} x={x(t)} y={H + 18} textAnchor="middle" className="ri-axis">{fmtT(t)}</text>)}
      </svg>
      <p className="ch-note">Line: GPU power from the NVML energy counter. Strip: share of time with at least one
        LLM call in flight. Bars: LLM calls started (max {maxCalls} per {data.step_s} s).</p>
      {data.first_calls.length > 0 && (
        <details className="ri-calls-table">
          <summary>First {data.first_calls.length} of {data.calls_total.toLocaleString()} LLM calls</summary>
          <div className="ri-row ri-head"><span>start</span><span>duration</span><span>tokens in / out</span><span>model</span><span>status</span></div>
          {data.first_calls.map((call, i) => (
            <div key={i} className="ri-row">
              <span>{call.start.toFixed(1)} s</span><span>{call.dur.toFixed(2)} s</span>
              <span>{call.in ?? "–"} / {call.out ?? "–"}</span><span>{call.model ?? "–"}</span><span>{call.status ?? "–"}</span>
            </div>
          ))}
        </details>
      )}
    </div>
  );
}
