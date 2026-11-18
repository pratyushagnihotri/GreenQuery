"use client";

import { Check, CircleStop, Crown, Gauge, Leaf, Lightbulb, Play, Plus, RotateCcw, Smartphone, Trophy, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import snapshot from "@/data/sembench_measured.json";
import type { CrossSystemRow } from "@/components/cross-system";
import { PlayQr } from "@/components/play-qr";
import { RunInspector, type InspectSource } from "@/components/run-inspector";
import { LiveHeartbeat, STAGES, stageOf, useRecordedBeat, type Beat } from "@/components/live-heartbeat";
import {
  addCustomModel, cancelRun, fetchBoard, fetchCatalog, fetchLive, MODEL_SOURCES, postBoardEntry, resetBoard, startRun,
  type BoardEntry, type LiveProgress, type LiveRecord, type ModelSource,
} from "@/lib/api";

// GreenQuery Challenge (demo mode 1): answer a SemBench query correctly with the least measured energy and
// beat the built-in optimizers. Recorded mode replays measured runs from the bundled snapshot (works offline);
// live mode runs the query on the DGX through the GreenQuery service with measured energy. With the service
// reachable the leaderboard is shared (server), otherwise it lives in this browser.

const QUALITY_TARGET = 0.8;
// Combinations whose recorded runs took longer than this are replay-only at the booth.
const LIVE_MAX_SECONDS = 600;
const TASKS = [
  { key: "Q1", query: "Q1_positive_reviews", title: "Five clearly positive reviews", detail: "Any movie, out of 1,000 reviews." },
  { key: "Q2", query: "Q2_positive_reviews_for_taken_3", title: "Five positive reviews of Taken 3", detail: "Only the 120 reviews of one movie qualify." },
  { key: "Q4", query: "Q4_positive_reviews_for_taken_3_ratio", title: "Share of positive reviews of Taken 3", detail: "One number; quality = 1 − relative error." },
  { key: "Q5", query: "Q3_reviews_same_sentiment", title: "Ten review pairs with the same sentiment", detail: "Ant-Man reviews: 256 × 256 candidate pairs." },
];
// key = system name in the measured data; service = system key in the GreenQuery service.
const SYSTEMS = [
  { key: "thalamusdb", service: "thalamusdb", label: "ThalamusDB" },
  { key: "palimpzest", service: "palimpzest", label: "Palimpzest" },
  { key: "lotus", service: "lotus", label: "LOTUS" },
  { key: "stretto", service: "sembench-stretto", label: "Stretto" },
  { key: "stretto_abacus", service: "sembench-abacus", label: "Abacus (in Stretto)" },
  { key: "caesura", service: "caesura", label: "CAESURA" },
];
const MODELS = [
  { key: "qwen2.5:3b-instruct", label: "Qwen 2.5 3B" },
  { key: "qwen2.5:7b-instruct", label: "Qwen 2.5 7B" },
  { key: "qwen2.5:14b-instruct", label: "Qwen 2.5 14B" },
  { key: "qwen2.5:32b-instruct", label: "Qwen 2.5 32B" },
  { key: "llama3.2:1b", label: "Llama 3.2 1B" },
  { key: "llama3.2:3b", label: "Llama 3.2 3B" },
  { key: "llama3.1:8b", label: "Llama 3.1 8B" },
  { key: "gemma3:4b", label: "Gemma 3 4B" },
  { key: "gemma3:12b", label: "Gemma 3 12B" },
  { key: "gemma3:27b", label: "Gemma 3 27B" },
];
// Everyday equivalents of an energy value (Challenge "what is a watt-hour?").
const LED_BULB_W = 10;
const PHONE_CHARGE_WH = 15;
const GRID_G_PER_KWH = 344; // same fixed grid factor as the CO2 values elsewhere in GreenQuery
const BOARD_KEY = "greenquery.challenge.board.v2";
const BOARD_EVENT = "greenquery-challenge-board";

type Mode = "recorded" | "live";
type RunResult = {
  mode: Mode; correct: boolean; label: string; energyWh: number; calls: number; runtimeS: number;
  quality: number | null; gpuDynamicWh: number; gpuIdleWh: number; idleW: number | null; cpuWh: number;
  promptTokens: number; completionTokens: number; reference: string; inspect: InspectSource;
};

const rows = ((snapshot as { rows: CrossSystemRow[] }).rows).filter((row) => row.energy_source === "measured"
  && row.energy_wh != null && !row.model.includes("operators"));
const systemLabel = (key: string) => SYSTEMS.find((item) => item.key === key)?.label ?? key;
const modelLabel = (key: string) => MODELS.find((item) => item.key === key)?.label ?? key;
const fmtDuration = (seconds: number) => (seconds < 90 ? `${Math.max(1, Math.round(seconds))} s`
  : seconds < 5400 ? `${Math.round(seconds / 60)} min` : `${(seconds / 3600).toFixed(1)} h`);
const fmtShare = (share: number) => (share < 0.01 ? "< 1 %" : share < 1 ? `${Math.round(share * 100)} %` : `${share.toFixed(1)}×`);
const wh = (value: number) => (value < 1 ? value.toFixed(2) : value < 10 ? value.toFixed(1) : value.toFixed(0));

function verdict(timedOut: boolean, failed: boolean, quality: number | null): [boolean, string] {
  if (timedOut) return [false, "Timed out"];
  if (failed) return [false, "The system failed before answering"];
  if ((quality ?? 0) >= QUALITY_TARGET) return [true, "Correct"];
  return [false, "Answer below the quality target"];
}

function fromRow(row: CrossSystemRow): RunResult {
  const [correct, label] = verdict(row.reason.includes("timeout"), row.status !== "success", row.score);
  return {
    mode: "recorded", correct, label, energyWh: row.energy_wh ?? 0, calls: row.llm_calls ?? 0, runtimeS: row.runtime_s ?? 0,
    quality: row.quality == null ? null : row.score, gpuDynamicWh: row.gpu_dynamic_wh ?? 0,
    gpuIdleWh: (row.gpu_wh ?? 0) - (row.gpu_dynamic_wh ?? 0), idleW: row.gpu_idle_w, cpuWh: row.cpu_job_wh ?? 0,
    promptTokens: row.prompt_tokens ?? 0, completionTokens: row.completion_tokens ?? 0,
    reference: `recorded run ${row.run}, repetition ${row.repetition}`,
    inspect: { kind: "recorded", run: row.run, system: row.system, query: row.query, repetition: row.repetition },
  };
}

function fromLive(record: LiveRecord, jobId: string, system: string, query: string): RunResult {
  const m = record.measurement;
  const [correct, label] = verdict(m.timed_out, record.quality.status !== "evaluated", record.quality.score);
  return {
    mode: "live", correct, label, energyWh: record.energy_kwh * 1000, calls: m.llm_calls ?? 0,
    runtimeS: record.runtime_seconds ?? 0, quality: record.quality.score, gpuDynamicWh: m.gpu_dynamic_wh,
    gpuIdleWh: m.gpu_wh - m.gpu_dynamic_wh, idleW: m.gpu_idle_w, cpuWh: m.cpu_job_wh,
    promptTokens: m.prompt_tokens ?? 0, completionTokens: m.completion_tokens ?? 0, reference: `live run ${jobId}`,
    inspect: { kind: "live", jobId, system, query },
  };
}

// Median-energy recorded run for a configuration (the optimizers' reference value).
function reference(system: string, model: string, query: string): CrossSystemRow | undefined {
  const found = rows.filter((r) => r.system === system && r.model === model && r.query === query)
    .sort((a, b) => (a.energy_wh ?? 0) - (b.energy_wh ?? 0));
  return found[Math.floor((found.length - 1) / 2)];
}

// Offline leaderboard in this browser's localStorage, read through useSyncExternalStore.
let memoryBoard = "[]"; // fallback when storage is unavailable (private mode)
function boardSnapshot(): string {
  try {
    return window.localStorage.getItem(BOARD_KEY) ?? "[]";
  } catch {
    return memoryBoard;
  }
}

function subscribeBoard(onChange: () => void) {
  window.addEventListener("storage", onChange);
  window.addEventListener(BOARD_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(BOARD_EVENT, onChange);
  };
}

function parseBoard(raw: string): BoardEntry[] {
  try {
    return JSON.parse(raw) as BoardEntry[];
  } catch {
    return [];
  }
}

function writeLocalBoard(entries: BoardEntry[]) {
  memoryBoard = JSON.stringify(entries);
  try {
    window.localStorage.setItem(BOARD_KEY, memoryBoard);
  } catch {
    // Storage unavailable: memoryBoard keeps the board for this session.
  }
  window.dispatchEvent(new Event(BOARD_EVENT));
}

export function Challenge({ onShowResults }: { onShowResults?: () => void }) {
  const [task, setTask] = useState("Q1");
  const [system, setSystem] = useState("lotus");
  const [model, setModel] = useState("qwen2.5:7b-instruct");
  const [mode, setMode] = useState<Mode>("recorded");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);
  const [replay, setReplay] = useState<{ row: CrossSystemRow; progress: number } | null>(null);
  const [live, setLive] = useState<{ jobId: string; progress: LiveProgress | null } | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [online, setOnline] = useState(false);
  const [serverBoard, setServerBoard] = useState<BoardEntry[]>([]);
  const [lastEntry, setLastEntry] = useState<number | null>(null);
  const [inspecting, setInspecting] = useState(false);
  // "Bring your own model": local models from the Ollama library or Hugging Face (GGUF), added from here
  // (live only, no recorded runs yet). The Challenge ranks measured energy, so API models are added on
  // the Run Benchmark page instead.
  const [customModels, setCustomModels] = useState<{ key: string; label: string }[]>([]);
  const [newModel, setNewModel] = useState("");
  const [newSource, setNewSource] = useState<ModelSource>("ollama");
  const [adding, setAdding] = useState(false);
  const models = useMemo(() => [...MODELS, ...customModels.filter((c) => !MODELS.some((m) => m.key === c.key))],
    [customModels]);
  const isCustom = customModels.some((item) => item.key === model) && !MODELS.some((item) => item.key === model);
  const localRaw = useSyncExternalStore(subscribeBoard, boardSnapshot, () => "[]");
  const localBoard = useMemo(() => parseBoard(localRaw), [localRaw]);
  const timer = useRef<number | null>(null);

  const stopTimer = () => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = null;
  };
  useEffect(() => stopTimer, []);

  const loadBoard = useCallback((query: string) => {
    fetchBoard(query).then((entries) => { setServerBoard(entries); setOnline(true); }).catch(() => setOnline(false));
  }, []);
  useEffect(() => {
    let alive = true;
    fetchCatalog().then((catalog) => {
      if (!alive) return;
      setOnline(true);
      loadBoard(task);
      setCustomModels(catalog.models.filter((m) => m.custom).map((m) => ({ key: m.api_model, label: `${m.api_model} (added)` })));
    }).catch(() => { if (alive) setOnline(false); });
    return () => { alive = false; };
  }, [task, loadBoard]);

  const recorded = useMemo(() => rows.filter((r) => r.system === system && r.model === model && r.query === task),
    [system, model, task]);
  const recordedSeconds = useMemo(() => {
    const times = recorded.map((r) => r.runtime_s ?? 0).sort((a, b) => a - b);
    return times.length ? times[Math.floor((times.length - 1) / 2)] : null;
  }, [recorded]);
  const liveBlocked = !online ? "The GreenQuery service is not reachable — recorded mode only."
    : recordedSeconds != null && recordedSeconds > LIVE_MAX_SECONDS
      ? `Measured runs of this combination took ${Math.round(recordedSeconds / 60)} min — replay it instead.` : "";
  const effectiveMode: Mode = mode === "live" && !liveBlocked ? "live" : "recorded";

  const optimizers = useMemo(() => ["stretto", "stretto_abacus"].map((key) => {
    const row = reference(key, model, task);
    return row ? { name: `${systemLabel(key)} optimizer`, result: fromRow(row) } : null;
  }).filter((item): item is { name: string; result: RunResult } => item !== null), [model, task]);

  const addModel = async () => {
    setAdding(true);
    setError("");
    try {
      const { model: added } = await addCustomModel(newSource, newModel.trim());
      setCustomModels((items) => items.some((item) => item.key === added.api_model) ? items
        : [...items, { key: added.api_model, label: `${added.api_model} (added)` }]);
      setModel(added.api_model);
      setMode("live");
      setResult(null);
      setNewModel("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setAdding(false);
    }
  };

  const record = async (outcome: RunResult, jobId: string | null) => {
    setResult(outcome);
    setRunning(false);
    if (!outcome.correct) return;
    const entry: BoardEntry = { name: name.trim(), query: task, system, model, energy_wh: outcome.energyWh,
      quality: outcome.quality ?? 0, mode: outcome.mode, job_id: jobId, created_at: Date.now() / 1000 };
    if (online) {
      try {
        const { entry: saved } = await postBoardEntry(entry);
        setLastEntry(saved.id ?? null);
        loadBoard(task);
        return;
      } catch {
        // Fall back to the local board below.
      }
    }
    writeLocalBoard([...parseBoard(boardSnapshot()), entry]);
    setLastEntry(entry.created_at ?? null);
  };

  const runRecorded = () => {
    const row = recorded[Math.floor(Math.random() * recorded.length)];
    const seconds = Math.min(Math.max((row.runtime_s ?? 10) / 40, 2.5), 8);
    const started = Date.now();
    setReplay({ row, progress: 0 });
    timer.current = window.setInterval(() => {
      const p = Math.min((Date.now() - started) / (seconds * 1000), 1);
      setReplay({ row, progress: p });
      if (p >= 1) {
        stopTimer();
        void record(fromRow(row), null);
      }
    }, 50);
  };

  const runLive = async () => {
    const service = SYSTEMS.find((item) => item.key === system)!.service;
    const query = TASKS.find((item) => item.key === task)!.query;
    try {
      const { job } = await startRun({ system: service, workload: "movie", model: `ollama/${model}`, query,
        repetitions: 1, cache_policy: "disabled", operator_models: null });
      setLive({ jobId: job.id, progress: null });
      timer.current = window.setInterval(() => {
        fetchLive(job.id).then(({ progress }) => {
          setLive({ jobId: job.id, progress });
          if (!progress.done) return;
          stopTimer();
          if (progress.record) void record(fromLive(progress.record, job.id, system, task), job.id);
          else {
            setRunning(false);
            setError(progress.phase === "stopped" ? "Run stopped." : `The run failed: ${progress.error ?? progress.phase}`);
          }
        }).catch(() => undefined);
      }, 2000);
    } catch (reason) {
      setRunning(false);
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  const run = () => {
    if (!name.trim()) {
      setError("Enter a name for the leaderboard");
      return;
    }
    if (!recorded.length && effectiveMode === "recorded") return;
    stopTimer();
    setError("");
    setResult(null);
    setInspecting(false);
    setReplay(null);
    setLive(null);
    setRunning(true);
    if (effectiveMode === "live") void runLive();
    else runRecorded();
  };

  const stop = () => {
    if (live) void cancelRun(live.jobId).catch(() => undefined);
  };

  const meter = result ? { energy: result.energyWh, calls: result.calls }
    : replay ? { energy: (replay.row.energy_wh ?? 0) * replay.progress, calls: Math.round((replay.row.llm_calls ?? 0) * replay.progress) }
      : live?.progress ? { energy: live.progress.energy_wh, calls: live.progress.llm_calls } : { energy: 0, calls: 0 };
  const phase = running ? (live ? (live.progress?.phase ?? "submitting to the GPU cluster") : "replaying a measured run") : "";

  const ranking = useMemo(() => {
    const entries = online ? serverBoard : localBoard.filter((entry) => entry.query === task);
    const humans = entries.map((entry) => ({
      key: `h-${entry.id ?? entry.created_at}`, name: entry.name, wh: entry.energy_wh, builtIn: false,
      detail: `${systemLabel(entry.system)} · ${modelLabel(entry.model)}${entry.mode === "live" ? " · live" : ""}`,
      mine: lastEntry != null && (entry.id === lastEntry || entry.created_at === lastEntry),
    }));
    const builtIn = optimizers.filter((o) => o.result.correct).map((o) => ({
      key: `o-${o.name}`, name: o.name, wh: o.result.energyWh, builtIn: true, mine: false, detail: modelLabel(model),
    }));
    return [...humans, ...builtIn].sort((a, b) => a.wh - b.wh).slice(0, 12);
  }, [online, serverBoard, localBoard, task, optimizers, model, lastEntry]);

  const best = optimizers.filter((o) => o.result.correct).sort((a, b) => a.result.energyWh - b.result.energyWh)[0];
  const lock = running;
  const beatOptimizer = !!(result?.correct && best && result.energyWh < best.result.energyWh);

  // Race lanes: you, the (best) optimizer, and the most energy-hungry other engine for this model and task.
  // Reference lanes advance with the run's clock (replayed seconds, or live seconds since the query started)
  // and complete once your run has finished.
  const baseline = useMemo(() => SYSTEMS.map((item) => item.key)
    .filter((key) => key !== system && key !== "stretto" && key !== "stretto_abacus")
    .map((key) => reference(key, model, task))
    .filter((row): row is CrossSystemRow => row !== undefined)
    .sort((a, b) => (b.energy_wh ?? 0) - (a.energy_wh ?? 0))[0], [system, model, task]);
  const optimizerLane = best ?? [...optimizers].sort((a, b) => a.result.energyWh - b.result.energyWh)[0];
  const clock = result ? Infinity : replay ? replay.progress * (replay.row.runtime_s ?? 0)
    : live?.progress ? live.progress.elapsed_s : 0;
  const refLane = (kind: string, title: string, sub: string, ref: RunResult) => {
    const share = Math.min(1, clock / Math.max(ref.runtimeS, 0.1));
    return { key: title, title, sub, kind, wh: ref.energyWh * share, calls: Math.round(ref.calls * share),
      final: ref.energyWh, finished: share >= 1, correct: ref.correct };
  };
  const yourFinal = result?.energyWh ?? (replay ? replay.row.energy_wh ?? 0 : meter.energy);
  const lanes = [
    { key: "you", title: name.trim() || "You", sub: `${systemLabel(system)} · ${modelLabel(model)}`, kind: "you",
      wh: meter.energy, calls: meter.calls, final: yourFinal, finished: !!result, correct: !!result?.correct },
    ...(optimizerLane ? [refLane("optimizer", optimizerLane.name, "built-in optimizer", optimizerLane.result)] : []),
    ...(baseline ? [refLane("baseline", systemLabel(baseline.system), "most energy for this task", fromRow(baseline))] : []),
  ];
  const raceScale = Math.max(0.01, ...lanes.map((lane) => lane.final));

  // Heartbeat: stages and GPU power of the live run, or of the replayed measured run.
  const replayFile = replay ? `${replay.row.run}__${replay.row.system}__${replay.row.query}__rep${replay.row.repetition}.json` : null;
  const recordedBeat = useRecordedBeat(replayFile, replay ? (result ? Infinity : replay.progress * (replay.row.runtime_s ?? 0)) : 0);
  const liveBeat: Beat | null = live?.progress?.power ? {
    power: live.progress.power, calls: live.progress.call_starts ?? [], idleW: live.progress.idle_w ?? null } : null;
  const beat = live ? liveBeat : recordedBeat;
  const stage = result || (live?.progress?.done) ? STAGES.length
    : live ? stageOf(live.progress?.phase ?? "") : replay ? 3 : 0;
  const powerNow = live ? live.progress?.power_w ?? null
    : recordedBeat?.power.length ? recordedBeat.power[recordedBeat.power.length - 1][1] : null;
  const podium = ranking.slice(0, 3);

  return (
    <div className="page-stack ch-page">
      <div className="panel ch-intro">
        <PlayQr />
        <span className="eyebrow">GREENQUERY CHALLENGE</span>
        <h3>Answer the query correctly with the least energy — and beat the optimizer</h3>
        <p>Every run is a real execution of an unmodified engine on one A100, with energy measured from the GPU and CPU.
          Quality is scored by SemBench; only answers with quality ≥ {QUALITY_TARGET} are ranked.</p>
        <div className="ch-tasks">
          {TASKS.map((item) => (
            <button key={item.key} className={`ch-task${task === item.key ? " active" : ""}`} disabled={lock}
              onClick={() => { setTask(item.key); setResult(null); }}>
              <strong>{item.key} · {item.title}</strong><small>{item.detail}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="panel ch-config">
        <div className="panel-heading">
          <span className="eyebrow">YOUR CHOICE</span>
          <div className="ch-mode" role="group" aria-label="Run mode">
            <button className={effectiveMode === "recorded" ? "active" : ""} disabled={lock} onClick={() => setMode("recorded")}>Recorded</button>
            <button className={effectiveMode === "live" ? "active" : ""} disabled={lock || !!liveBlocked}
              title={liveBlocked || "Run on the GPU now"} onClick={() => setMode("live")}>Live on the GPU</button>
          </div>
        </div>
        <div className="ch-controls">
          <label>Name on the leaderboard
            <input value={name} maxLength={24} placeholder="Ada" disabled={lock}
              onChange={(event) => { setName(event.target.value); setError(""); }} />
          </label>
          <label>System
            <select value={system} disabled={lock} onChange={(event) => { setSystem(event.target.value); setResult(null); }}>
              {SYSTEMS.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
            </select>
          </label>
          <label>Model
            <select value={model} disabled={lock} onChange={(event) => { setModel(event.target.value); setResult(null); }}>
              {models.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
            </select>
          </label>
          {running && live
            ? <button className="secondary-action ch-run" onClick={stop}><CircleStop size={14} /> Stop</button>
            : <button className="primary-action ch-run" disabled={lock || (effectiveMode === "recorded" && !recorded.length)} onClick={run}>
                <Play size={14} /> {running ? "Running…" : effectiveMode === "live" ? "Run live" : "Run"}</button>}
        </div>
        {online && (
          <div className="ch-add">
            <label>Bring your own model
              <select value={newSource} disabled={lock || adding} onChange={(event) => setNewSource(event.target.value as ModelSource)}>
                {MODEL_SOURCES.filter((item) => item.local).map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
              </select>
            </label>
            <label>Model
              <input value={newModel} maxLength={160} placeholder={MODEL_SOURCES.find((item) => item.key === newSource)?.placeholder}
                disabled={lock || adding} onChange={(event) => setNewModel(event.target.value)}
                onKeyDown={(event) => { if (event.key === "Enter" && newModel.trim()) void addModel(); }} />
            </label>
            <button className="secondary-action" disabled={lock || adding || !newModel.trim()} onClick={() => void addModel()}>
              <Plus size={14} /> {adding ? "Checking…" : "Add model"}</button>
          </div>
        )}
        {error && <p className="ch-error">{error}</p>}
        <p className="ch-note">{effectiveMode === "live"
          ? `Live: runs on a free A100 of the DGX with measured energy${recordedSeconds ? ` (measured runs took about ${Math.max(1, Math.round(recordedSeconds / 60))} min)` : ""}.`
          : `Recorded: replays one of ${recorded.length} measured ${recorded.length === 1 ? "run" : "runs"} of this configuration from the DGX, accelerated.`}
          {mode === "live" && liveBlocked ? ` ${liveBlocked}` : ""}
          {isCustom ? " Added model: the first run downloads it (not measured), then measures as usual." : ""}</p>
      </div>

      <div className="panel ch-meter">
        <span className="eyebrow">{running ? phase.toUpperCase() : result ? "FINISHED" : "ENERGY METER"}</span>
        <div className="ch-metrics">
          <div><span>Energy{live && !result ? " (GPU, so far)" : ""}</span><strong>{wh(meter.energy)} Wh</strong></div>
          <div><span>LLM calls</span><strong>{meter.calls.toLocaleString()}</strong></div>
          <div><span>Runtime</span><strong>{result ? `${Math.round(result.runtimeS).toLocaleString()} s`
            : live?.progress ? `${Math.round(live.progress.elapsed_s)} s` : running ? "…" : "–"}</strong></div>
          <div><span>Quality</span><strong className={result ? (result.correct ? "ok" : "bad") : ""}>
            {result && result.quality != null ? result.quality.toFixed(2) : "–"}</strong></div>
        </div>
        {(live || replay) && (
          <LiveHeartbeat title={live ? `Live on a DGX A100 · ${systemLabel(system)} · ${modelLabel(model)}`
            : `${systemLabel(replay!.row.system)} · ${modelLabel(replay!.row.model)}`}
            stage={stage} beat={beat} powerNow={powerNow} replay={!live} />
        )}
        <div className={`ch-race${live && running ? " live" : ""}`}>
          {lanes.map((lane) => (
            <div key={lane.key} className={`ch-lane ${lane.kind}`}>
              <div className="ch-lane-name"><strong>{lane.title}</strong><small>{lane.sub}</small></div>
              <div className="ch-track">
                <i style={{ width: `${Math.max(lane.wh > 0 ? 0.6 : 0, (lane.wh / raceScale) * 100)}%` }} />
                {lane.finished && (
                  <span className={`ch-flag ${lane.correct ? "ok" : "bad"}`} title={lane.correct ? "correct answer" : "wrong answer"}>
                    {lane.correct ? <Check size={12} /> : <X size={12} />}</span>)}
              </div>
              <div className="ch-lane-val"><strong>{wh(lane.wh)} Wh</strong><small>{lane.calls.toLocaleString()} calls</small></div>
            </div>
          ))}
        </div>
        {result && (
          <div className={`ch-verdict ${result.correct ? "ok" : "bad"}`}>
            {beatOptimizer && (
              <div className="ch-confetti" aria-hidden="true">
                {Array.from({ length: 18 }, (_, i) => <i key={i} style={{ left: `${(i * 37) % 100}%`, animationDelay: `${(i % 6) * 60}ms` }} />)}
              </div>)}
            <Gauge size={18} />
            <div>
              <strong>{result.label} · {wh(result.energyWh)} Wh{result.mode === "live" ? " · measured live" : ""}</strong>
              <p>{result.correct && best
                ? result.energyWh < best.result.energyWh
                  ? `You beat the ${best.name} (${wh(best.result.energyWh)} Wh) by ${Math.max(1, Math.round(best.result.energyWh / Math.max(result.energyWh, 0.001)))}×.`
                  : `The ${best.name} needed ${wh(best.result.energyWh)} Wh — try another system or model.`
                : result.correct ? optimizers.length ? "No optimizer answered this one correctly with this model."
                  : "No optimizer reference for this model yet — your run sets the first mark."
                  : `The energy was spent anyway. ${task === "Q5" ? "Hint: some systems stop as soon as ten pairs are found." : "Try another system or model."}`}</p>
            </div>
          </div>
        )}
        {result && (
          <div className="ch-equiv">
            <div><Lightbulb size={16} /><span>an LED bulb ({LED_BULB_W} W) for</span>
              <strong>{fmtDuration((result.energyWh / LED_BULB_W) * 3600)}</strong></div>
            <div><Smartphone size={16} /><span>of a phone charge</span>
              <strong>{fmtShare(result.energyWh / PHONE_CHARGE_WH)}</strong></div>
            <div><Leaf size={16} /><span>CO₂ at {GRID_G_PER_KWH} g/kWh</span>
              <strong>{((result.energyWh * GRID_G_PER_KWH) / 1000).toFixed(result.energyWh < 3 ? 2 : 1)} g</strong></div>
            {beatOptimizer && best && (
              <div className="win"><Trophy size={16} /><span>saved vs. the optimizer</span>
                <strong>{wh(best.result.energyWh - result.energyWh)} Wh</strong></div>)}
          </div>
        )}
        {result && (
          <div className="ch-breakdown">
            <span className="eyebrow">HOW IT WAS MEASURED</span>
            <div className="ch-metrics small">
              <div><span>GPU above idle</span><strong>{wh(result.gpuDynamicWh)} Wh</strong></div>
              <div><span>GPU idle{result.idleW ? ` (${Math.round(result.idleW)} W)` : ""}</span><strong>{wh(result.gpuIdleWh)} Wh</strong></div>
              <div><span>CPU (job share)</span><strong>{wh(result.cpuWh)} Wh</strong></div>
              <div><span>Tokens in / out</span><strong>{result.promptTokens.toLocaleString()} / {result.completionTokens.toLocaleString()}</strong></div>
            </div>
            <p className="ch-note">NVML energy counter + RAPL CPU energy × job CPU share; every LLM call logged by a proxy.
              Source: <code>{result.reference}</code>.</p>
            <div className="ch-actions">
              <button className="secondary-action" onClick={() => setInspecting((open) => !open)}>
                {inspecting ? "Hide run inspector" : "Open run inspector"}</button>
              {onShowResults && <button className="secondary-action" onClick={onShowResults}>Compare all systems</button>}
            </div>
            {inspecting && <RunInspector source={result.inspect} />}
          </div>
        )}
      </div>

      <div className="panel ch-board">
        <div className="panel-heading">
          <div><span className="eyebrow">LEADERBOARD · {task} · {online ? "SHARED" : "THIS BROWSER"}</span>
            <h3><Trophy size={15} /> Least energy for a correct answer</h3></div>
          <button className="ch-reset" title="Clear the leaderboard"
            onClick={() => {
              if (!window.confirm("Clear the leaderboard?")) return;
              if (online) void resetBoard().then(() => loadBoard(task)).catch(() => undefined);
              writeLocalBoard([]);
            }}>
            <RotateCcw size={12} /> reset
          </button>
        </div>
        {ranking.length === 0 && <p className="ch-note">No correct answers yet — be the first.</p>}
        {podium.length > 0 && (
          <div className="ch-podium">
            {[1, 0, 2].map((place) => podium[place] && (
              <div key={podium[place].key} className={`ch-step p${place + 1}${podium[place].mine ? " mine" : ""}${podium[place].builtIn ? " builtin" : ""}`}>
                {place === 0 ? <Crown size={18} /> : <span>{place + 1}</span>}
                <strong>{podium[place].name}</strong>
                <em>{wh(podium[place].wh)} Wh</em>
                <small>{podium[place].builtIn ? "built-in optimizer" : podium[place].detail}</small>
                <div className="ch-step-block" />
              </div>
            ))}
          </div>
        )}
        {ranking.slice(3).map((entry, index) => (
          <div key={entry.key} className={`ch-rank${entry.builtIn ? " builtin" : ""}${entry.mine ? " mine" : ""}`}>
            <span>{index + 4}</span>
            <strong>{entry.name}</strong>
            <small>{entry.builtIn ? `built-in optimizer · ${entry.detail}` : entry.detail}</small>
            <em>{wh(entry.wh)} Wh</em>
          </div>
        ))}
      </div>
    </div>
  );
}
