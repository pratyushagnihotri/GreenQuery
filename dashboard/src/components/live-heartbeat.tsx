"use client";

import { Check, LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";

// Challenge heartbeat: the stages of a run and the GPU power over time, with the idle baseline and a tick per
// LLM call. Live runs pass the power series of the GreenQuery service; recorded runs replay the condensed
// trace bundled with the dashboard (public/inspector), revealed up to the replay's clock.

export const STAGES = ["Waiting for a GPU", "Starting the model", "Idle baseline", "Running the query", "Scoring"];

// Index of the active stage for a live phase text of the service (STAGES.length = finished).
export function stageOf(phase: string): number {
  if (phase === "finished") return STAGES.length;
  if (phase.includes("scoring")) return 4;
  if (phase.includes("running the query")) return 3;
  if (phase.includes("idle")) return 2;
  if (phase.includes("starting") || phase.includes("downloading")) return 1;
  return 0;
}

export type Beat = { power: Array<[number, number]>; calls: number[]; idleW: number | null };

type TracePoint = { t: number; w: number | null; calls: number };
type Trace = { duration_s: number; step_s: number; idle_w: number | null; points: TracePoint[] };

// Recorded heartbeat: the run's condensed trace, cut at `clock` seconds.
export function useRecordedBeat(file: string | null, clock: number): Beat | null {
  const [trace, setTrace] = useState<{ file: string; data: Trace | null } | null>(null);
  useEffect(() => {
    if (!file) return;
    let alive = true;
    fetch(`/inspector/${encodeURIComponent(file)}`).then((response) => (response.ok ? response.json() : null))
      .then((data: Trace | null) => { if (alive) setTrace({ file, data }); }).catch(() => undefined);
    return () => { alive = false; };
  }, [file]);
  const data = trace?.file === file ? trace.data : null;
  if (!data) return null;
  const shown = data.points.filter((p) => p.t <= clock);
  return {
    power: shown.filter((p) => p.w != null).map((p) => [p.t, p.w!] as [number, number]),
    calls: shown.flatMap((p) => Array.from({ length: Math.min(p.calls, 3) }, () => p.t + data.step_s / 2)),
    idleW: data.idle_w,
  };
}

const W = 640, H = 170, LEFT = 46, RIGHT = 14, TOP = 12, PLOT_H = 112, TICK_TOP = 136, TICK_H = 14;

export function LiveHeartbeat({ title, stage, beat, powerNow, replay }: {
  title: string; stage: number; beat: Beat | null; powerNow: number | null; replay: boolean;
}) {
  const power = beat?.power ?? [];
  const t0 = power.length ? power[0][0] : 0;
  const t1 = Math.max(t0 + 10, power.length ? power[power.length - 1][0] : 10);
  // Scale to the 98th percentile so a single counter glitch does not flatten the line; spikes are clipped.
  const sortedW = power.map(([, w]) => w).sort((a, b) => a - b);
  const maxW = Math.max(100, beat?.idleW ?? 0, sortedW[Math.floor(sortedW.length * 0.98)] ?? 0) * 1.12;
  const x = (t: number) => LEFT + ((t - t0) / (t1 - t0)) * (W - LEFT - RIGHT);
  const y = (w: number) => TOP + PLOT_H - (Math.min(w, maxW) / maxW) * PLOT_H;
  const line = power.map(([t, w]) => `${x(t).toFixed(1)},${y(w).toFixed(1)}`).join(" ");
  const last = power[power.length - 1];
  const ticks = (beat?.calls ?? []).filter((t) => t >= t0 && t <= t1);
  return (
    <div className="hb">
      <div className="hb-head">
        <div><span className="eyebrow">{title}</span>{replay && <small>replay of a measured run</small>}</div>
        <div className="hb-now"><span>GPU power now</span><strong>{powerNow != null ? `${Math.round(powerNow)} W` : "–"}</strong></div>
      </div>
      <ol className="hb-stages">
        {STAGES.map((label, index) => (
          <li key={label} className={index < stage ? "done" : index === stage ? "active" : ""}>
            {index < stage ? <Check size={13} /> : index === stage ? <LoaderCircle size={13} className="hb-spin" /> : <i />}
            {label}
          </li>
        ))}
      </ol>
      <div className="hb-plot">
        {power.length < 2
          ? <p className="hb-wait">{stage < 2 ? "The power trace starts once the model server is up." : "Reading the GPU energy counter…"}</p>
          : (
            <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`GPU power over ${Math.round(t1 - t0)} seconds, ${ticks.length} LLM calls`}>
              {[0.5, 1].filter((f) => beat?.idleW == null || Math.abs(y(maxW * f / 1.12) - y(beat.idleW)) > 14).map((f) => (
                <text key={f} x={LEFT - 8} y={y(maxW * f / 1.12) + 4} textAnchor="end" className="hb-axis">{Math.round(maxW * f / 1.12)} W</text>
              ))}
              {beat?.idleW != null && (
                <g>
                  <line x1={LEFT} x2={W - RIGHT} y1={y(beat.idleW)} y2={y(beat.idleW)} className="hb-idle" />
                  <text x={LEFT - 8} y={y(beat.idleW) + 4} textAnchor="end" className="hb-axis">idle</text>
                </g>
              )}
              {t0 < 0 && t1 > 0 && <line x1={x(0)} x2={x(0)} y1={TOP} y2={TICK_TOP + TICK_H} className="hb-start" />}
              <polyline points={line} className="hb-line" />
              {last && <circle cx={x(last[0])} cy={y(last[1])} r={4.5} className="hb-dot" />}
              {ticks.map((t, i) => <rect key={i} x={x(t) - 1} y={TICK_TOP} width={2} height={TICK_H} rx={1} className="hb-tick" />)}
              <text x={W - RIGHT} y={H - 2} textAnchor="end" className="hb-axis">{`${Math.round(t1 - t0)} s shown · ticks = LLM calls`}</text>
            </svg>
          )}
      </div>
    </div>
  );
}
