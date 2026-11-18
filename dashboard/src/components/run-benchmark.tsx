"use client";

import { Check, CircleStop, LayoutGrid, List, LoaderCircle, Play, Plus, RefreshCw, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE,
  ApiError,
  addCustomModel,
  cancelRun,
  estimateRun,
  fetchCatalog,
  fetchJobs,
  fetchLive,
  MODEL_SOURCES,
  startRun,
  subscribeToJob,
  type Catalog,
  type CostEstimate,
  type Job,
  type LiveProgress,
  type ModelInfo,
  type ModelSource,
  type RunRequest,
} from "@/lib/api";
import { queryLabel } from "@/data/benchmarks";
import { RunInspector } from "@/components/run-inspector";
import { STAGES, stageOf } from "@/components/live-heartbeat";

const PROVIDER_LABEL: Record<ModelInfo["provider"], string> = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  openai_compatible: "Local server (OpenAI-compatible)",
};
const ACTIVE = new Set(["queued", "running"]);
const money = (value: number | null | undefined) => (typeof value === "number" ? `$${value.toFixed(value < 0.1 ? 4 : 2)}` : "Unknown");
const time = (seconds: number) => new Date(seconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

const CARD_COUNT = 6;

// Job card: phase, stage bar, energy so far, LLM calls and a GPU power sparkline (measured DGX jobs report
// live progress from their output folder; other jobs show the service's step text).
function JobCard({ job, progress, title, subtitle, selected, onSelect, onInspect, stop }: {
  job: Job; progress: LiveProgress | undefined; title: string; subtitle: string; selected: boolean;
  onSelect: () => void; onInspect: () => void; stop: React.ReactNode;
}) {
  const active = ACTIVE.has(job.status);
  const measured = !!progress && (progress.power?.length || progress.llm_calls || progress.record);
  const stage = job.status === "succeeded" ? STAGES.length : active && progress ? stageOf(progress.phase) : 0;
  const share = job.status === "succeeded" ? 1 : active ? (measured ? (stage + 0.5) / STAGES.length : 0.15) : 1;
  const energy = progress?.record ? progress.record.energy_kwh * 1000 : progress?.energy_wh ?? 0;
  const calls = progress?.record?.measurement.llm_calls ?? progress?.llm_calls ?? 0;
  const runtime = progress?.record?.runtime_seconds ?? progress?.elapsed_s ?? 0;
  const power = (progress?.power ?? []).slice(-60);
  const maxW = Math.max(100, ...power.map(([, w]) => w));
  const spark = power.map(([, w], i) => `${((i / Math.max(power.length - 1, 1)) * 100).toFixed(1)},${(28 - (w / maxW) * 26).toFixed(1)}`).join(" ");
  const score = progress?.record?.quality.score;
  const status = job.status !== "succeeded" ? job.status
    : score == null ? "finished" : `${score >= 0.8 ? "correct" : "below target"} · ${score.toFixed(2)}`;
  return (
    <div role="button" tabIndex={0} className={`run-card ${job.status}${selected ? " active" : ""}`} onClick={onSelect}
      onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onSelect(); } }}>
      <div className="run-card-head">
        <div><strong>{title}</strong><small>{subtitle}</small></div>
        <span className={`run-card-status ${job.status}`}>
          {active ? <LoaderCircle size={13} className="hb-spin" /> : job.status === "succeeded" ? <Check size={13} /> : <X size={13} />}
          {status}</span>
      </div>
      <div className="run-card-bar"><i style={{ width: `${Math.round(share * 100)}%` }} className={active && !measured ? "indeterminate" : ""} /></div>
      <small className="run-card-phase">{active ? (measured ? progress!.phase : job.step ?? "queued") : job.status === "succeeded"
        ? `${job.record_count} record${job.record_count === 1 ? "" : "s"} · ${time(job.updated_at)}` : job.error ?? job.step ?? ""}</small>
      <div className="run-card-metrics">
        <div><span>{active ? "Energy so far" : "Energy"}</span><strong>{measured ? `${energy < 1 ? energy.toFixed(2) : energy.toFixed(1)} Wh` : "–"}</strong></div>
        <div><span>LLM calls</span><strong>{measured ? calls.toLocaleString() : "–"}</strong></div>
        <div><span>Runtime</span><strong>{measured && runtime ? `${Math.round(runtime)} s` : "–"}</strong></div>
      </div>
      {power.length > 1 && (
        <svg className="run-card-spark" viewBox="0 0 100 30" preserveAspectRatio="none" aria-label="GPU power, last minute">
          <polyline points={spark} vectorEffect="non-scaling-stroke" />
        </svg>
      )}
      <div className="run-card-actions" onClick={(event) => event.stopPropagation()}>
        {job.status === "succeeded" && measured && <button className="secondary-action" onClick={onInspect}>Inspect</button>}
        {stop}
      </div>
    </div>
  );
}

function energyNote(model: ModelInfo | undefined) {
  if (!model) return "";
  if (model.energy === "ecologits") return "Energy and CO₂ estimated by EcoLogits for this model.";
  if (model.energy === "ecologits_proxy") return `EcoLogits has no profile for this model yet; energy is estimated with ${model.ecologits_model} as a labelled proxy.`;
  if (model.deployment === "Online API") return "Online API: inference runs at the provider, so energy is not measured; tokens, cost and quality are recorded.";
  return "No energy meter for this server: energy and CO₂ will be recorded as not measured.";
}

export function RunBenchmark({ onRecords, onShowResults }: { onRecords: () => void; onShowResults?: () => void }) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [offline, setOffline] = useState<string | null>(null);
  const [form, setForm] = useState<RunRequest>({ system: "stretto", workload: "movie", model: "", query: null, repetitions: 1, cache_policy: "disabled", operator_models: null });
  const [estimate, setEstimate] = useState<CostEstimate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<string | null>(null);
  const [lastStarted, setLastStarted] = useState<string | null>(null);
  const [stopping, setStopping] = useState<Set<string>>(new Set());
  const [log, setLog] = useState<string[]>([]);
  const [progress, setProgress] = useState<Record<string, LiveProgress>>({});
  const [layout, setLayout] = useState<"cards" | "table">("cards");
  const [inspect, setInspect] = useState<Job | null>(null);
  // "Bring your own model": any Ollama, Hugging Face (GGUF), OpenAI or OpenRouter model.
  const [addSource, setAddSource] = useState<ModelSource>("ollama");
  const [addName, setAddName] = useState("");
  const [addPrices, setAddPrices] = useState({ price_in: "", price_out: "" });
  const [adding, setAdding] = useState(false);
  const [addNote, setAddNote] = useState<string | null>(null);
  const finished = useRef(new Set<string>());
  const logEnd = useRef<HTMLDivElement>(null);

  const loadCatalog = () =>
    fetchCatalog()
      .then((data) => {
        setCatalog(data);
        setOffline(null);
        setForm((current) => ({ ...current, model: current.model || data.models.find((model) => model.credential_available)?.key || data.models[0]?.key || "" }));
      })
      .catch((reason: unknown) => setOffline(reason instanceof Error ? reason.message : String(reason)));

  useEffect(() => {
    void loadCatalog();
  }, []);

  const addModel = async () => {
    setAdding(true);
    setAddNote(null);
    try {
      const entered = addPrices.price_in !== "" && addPrices.price_out !== "";
      const prices = addSource === "openai" || (addSource === "openrouter" && entered)
        ? { price_in: Number(addPrices.price_in) || undefined, price_out: Number(addPrices.price_out) || undefined } : undefined;
      const { model: added, added: isNew } = await addCustomModel(addSource, addName.trim(), prices);
      await loadCatalog();
      setForm((current) => ({ ...current, model: added.key }));
      setAddName("");
      setAddNote(`${added.display_name} ${isNew ? "added" : "is already available"}.`
        + (added.credential_available ? "" : ` Set ${added.credential_env} on the server to run it.`)
        + (added.deployment === "Online API" ? "" : " The first run downloads it on the DGX."));
    } catch (reason) {
      setAddNote(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setAdding(false);
    }
  };

  // Poll the job list; when a job newly succeeds, ask the parent to merge its records.
  useEffect(() => {
    if (!catalog) return;
    let alive = true;
    const tick = () =>
      fetchJobs()
        .then((list) => {
          if (!alive) return;
          setJobs(list);
          let changed = false;
          for (const job of list) {
            if (job.status === "succeeded" && !finished.current.has(job.id)) {
              finished.current.add(job.id);
              changed = true;
            }
          }
          if (changed) onRecords();
        })
        .catch(() => undefined);
    void tick();
    const timer = window.setInterval(tick, 4000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [catalog, onRecords]);

  // Live progress of the jobs shown as cards: active ones every poll, finished ones once.
  const cardJobs = useMemo(() => jobs.slice(0, CARD_COUNT), [jobs]);
  useEffect(() => {
    let alive = true;
    const tick = () => {
      for (const job of cardJobs) {
        const known = progress[job.id];
        if (!ACTIVE.has(job.status) && known && (known.done || job.status !== "succeeded")) continue;
        fetchLive(job.id).then(({ progress: next }) => {
          if (alive) setProgress((current) => ({ ...current, [job.id]: next }));
        }).catch(() => undefined);
      }
    };
    tick();
    const timer = window.setInterval(tick, 3000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- progress is read as a cache, not a trigger
  }, [cardJobs]);

  // Live log for the selected job.
  useEffect(() => {
    if (!selectedJob) return;
    setLog([]);
    return subscribeToJob(selectedJob, (event) => {
      if (event.type === "log") setLog((lines) => [...lines.slice(-800), event.line]);
      if (event.type === "status" && event.job) setJobs((list) => list.map((job) => (job.id === event.job?.id ? event.job : job)));
    });
  }, [selectedJob]);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ block: "nearest" });
  }, [log]);

  const system = catalog?.systems.find((item) => item.key === form.system);
  const workload = catalog?.workloads.find((item) => item.key === form.workload);
  const model = catalog?.models.find((item) => item.key === form.model);
  // Only models whose provider the selected system supports.
  const systemModels = useMemo(
    () => (catalog?.models ?? []).filter((item) => !system || system.providers.includes(item.provider)),
    [catalog, system],
  );
  const modelGroups = useMemo(() => {
    const groups = new Map<string, ModelInfo[]>();
    for (const item of systemModels) groups.set(item.provider, [...(groups.get(item.provider) ?? []), item]);
    return [...groups.entries()];
  }, [systemModels]);
  const multiModel = Boolean(form.operator_models);
  const runnable = Boolean(
    system && system.status !== "planned" && workload && model?.credential_available
    && (!multiModel || (form.operator_models?.length ?? 0) >= 2),
  );

  // Cost estimate whenever the form changes.
  useEffect(() => {
    if (!catalog || !system || system.status === "planned" || !workload || !model) return;
    let alive = true;
    const timer = window.setTimeout(() => {
      estimateRun(form)
        .then((value) => alive && setEstimate(value))
        .catch(() => alive && setEstimate(null));
    }, 250);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [catalog, form, system, workload, model]);

  const update = (patch: Partial<RunRequest>) => {
    setError(null);
    setForm((current) => ({ ...current, ...patch }));
  };

  // Keep the form consistent with what the chosen system supports.
  const selectSystem = (key: string) => {
    const next = catalog?.systems.find((item) => item.key === key);
    if (!next || !catalog) return;
    const workloadOk = next.workloads.includes(form.workload);
    const modelOk = catalog.models.some((item) => item.key === form.model && next.providers.includes(item.provider));
    update({
      system: key,
      workload: workloadOk ? form.workload : next.workloads[0],
      query: workloadOk ? form.query : null,
      model: modelOk ? form.model : catalog.models.find((item) => next.providers.includes(item.provider) && item.credential_available)?.key ?? form.model,
      operator_models: next.multi_model ? form.operator_models : null,
    });
  };
  const toggleOperatorModel = (key: string) => {
    const current = form.operator_models ?? [];
    const next = current.includes(key) ? current.filter((item) => item !== key) : [...current, key];
    // Keep registry order (cheapest first, as listed in config/models.yaml).
    const order = (catalog?.models ?? []).map((item) => item.key);
    update({ operator_models: next.sort((a, b) => order.indexOf(a) - order.indexOf(b)) });
  };

  const submit = async (acceptUnknownCost = false) => {
    setSubmitting(true);
    setError(null);
    try {
      const { job } = await startRun({ ...form, accept_unknown_cost: acceptUnknownCost });
      setJobs((list) => [job, ...list]);
      setSelectedJob(job.id);
      setLastStarted(job.id);
    } catch (reason) {
      if (reason instanceof ApiError && reason.status === 409 && !acceptUnknownCost) {
        if (window.confirm("No cost estimate exists for this combination. Start the paid run anyway?")) {
          await submit(true);
          return;
        }
      } else {
        setError(reason instanceof Error ? reason.message : String(reason));
      }
    } finally {
      setSubmitting(false);
    }
  };

  // Stop = cancel the job: the service runs scancel on its SLURM job and stops the local process.
  const stop = async (job: Job) => {
    if (!window.confirm(`Stop run ${job.id}? The job is cancelled on the server; results so far are kept in its log.`)) return;
    setStopping((set) => new Set(set).add(job.id));
    try {
      const { job: updated } = await cancelRun(job.id);
      setJobs((list) => list.map((item) => (item.id === updated.id ? updated : item)));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
      setStopping((set) => { const next = new Set(set); next.delete(job.id); return next; });
    }
  };
  const stopButton = (job: Job, label = "Stop") => (
    <button className="secondary-action run-stop" disabled={stopping.has(job.id)}
      onClick={(event) => { event.stopPropagation(); void stop(job); }}>
      <CircleStop size={14} /> {stopping.has(job.id) ? "Stopping…" : label}
    </button>
  );

  if (offline) {
    return (
      <div className="page-stack run-page">
        <div className="panel run-offline">
          <span className="eyebrow">SNAPSHOT MODE</span>
          <h3>Live runs need the GreenQuery service on the GPU server</h3>
          <p>It is not reachable right now, so new runs cannot start. Every view with archived and measured results works without it — the cross-system comparison shows all measured runs.</p>
          <div className="run-offline-actions">
            {onShowResults && <button className="primary-action" onClick={onShowResults}>Open measured results</button>}
            <button className="secondary-action" onClick={() => void loadCatalog()}><RefreshCw size={14} /> Retry connection</button>
          </div>
          <details>
            <summary>How to start the service</summary>
            <p>Expected at {API_BASE}. On the DGX: <code>scripts/start_server_dgx.sh</code>, then an SSH tunnel to this port. Locally: <code>uvicorn greenquery_server.app:app --port 8000</code>.</p>
          </details>
        </div>
      </div>
    );
  }
  if (!catalog) return <div className="page-stack run-page"><div className="panel run-offline"><p>Loading catalog…</p></div></div>;

  const selected = jobs.find((job) => job.id === selectedJob);
  const lastRun = jobs.find((job) => job.id === lastStarted);
  const overLimit = typeof estimate?.total_usd === "number" && estimate.total_usd > catalog.limits.max_run_cost_usd;

  return (
    <div className="page-stack run-page">
      <div className="panel run-form">
        <div className="panel-heading">
          <div><span className="eyebrow">NEW RUN</span><h3>Run a query through a system and model</h3></div>
          <span>Limit {money(catalog.limits.max_run_cost_usd)} per job</span>
        </div>
        <div className="source-bar run-controls">
          <label>System
            <select value={form.system} onChange={(event) => selectSystem(event.target.value)}>
              {[["Stretto engine", catalog.systems.filter((item) => item.key.startsWith("stretto"))],
                ["Real systems (SemBench)", catalog.systems.filter((item) => !item.key.startsWith("stretto"))]].map(([group, items]) => (
                <optgroup key={group as string} label={group as string}>
                  {(items as typeof catalog.systems).map((item) => (
                    <option key={item.key} value={item.key} disabled={item.status === "planned"}>
                      {item.label}{item.status === "experimental" ? " (experimental)" : item.status === "planned" ? " (needs DGX)" : ""}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>Workload
            <select value={form.workload} onChange={(event) => update({ workload: event.target.value, query: null })}>
              {catalog.workloads.filter((item) => system?.workloads.includes(item.key)).map((item) => <option key={item.key} value={item.key}>{item.dataset}</option>)}
            </select>
          </label>
          <label>Query
            <select value={form.query ?? ""} onChange={(event) => update({ query: event.target.value || null })} disabled={!workload?.single_query_runs}>
              <option value="">All queries</option>
              {workload?.single_query_runs && workload.queries.map((item) => <option key={item} value={item}>{queryLabel(item)}</option>)}
            </select>
          </label>
          <label>Model
            <select value={form.model} onChange={(event) => update({ model: event.target.value })}>
              {modelGroups.map(([provider, models]) => (
                <optgroup key={provider} label={PROVIDER_LABEL[provider as ModelInfo["provider"]] ?? provider}>
                  {models.map((item) => (
                    <option key={item.key} value={item.key} disabled={!item.credential_available}>
                      {item.display_name}{item.credential_available ? "" : ` (${item.credential_env} not set)`}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <label>Repetitions
            <input type="number" min={1} max={catalog.limits.max_repetitions} value={form.repetitions}
              onChange={(event) => update({ repetitions: Math.max(1, Math.min(catalog.limits.max_repetitions, Number(event.target.value) || 1)) })} />
          </label>
          <label>Cache
            <select value={form.cache_policy} onChange={(event) => update({ cache_policy: event.target.value as RunRequest["cache_policy"] })}>
              <option value="disabled">Cold (no LLM cache)</option>
              <option value="default">Allow cache</option>
            </select>
          </label>
        </div>
        <div className="run-add">
          <label>Bring your own model
            <select value={addSource} onChange={(event) => { setAddSource(event.target.value as ModelSource); setAddNote(null); }}>
              {MODEL_SOURCES.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
            </select>
          </label>
          <label>Model
            <input value={addName} maxLength={160} placeholder={MODEL_SOURCES.find((item) => item.key === addSource)?.placeholder}
              onChange={(event) => setAddName(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter" && addName.trim() && !adding) void addModel(); }} />
          </label>
          {(addSource === "openai" || addSource === "openrouter") && (
            <>
              <label>$ / M input{addSource === "openrouter" ? " (optional)" : ""}
                <input type="number" min={0} step="0.01" value={addPrices.price_in}
                  onChange={(event) => setAddPrices((current) => ({ ...current, price_in: event.target.value }))} />
              </label>
              <label>$ / M output{addSource === "openrouter" ? " (optional)" : ""}
                <input type="number" min={0} step="0.01" value={addPrices.price_out}
                  onChange={(event) => setAddPrices((current) => ({ ...current, price_out: event.target.value }))} />
              </label>
            </>
          )}
          <button className="secondary-action" disabled={adding || !addName.trim()} onClick={() => void addModel()}>
            <Plus size={14} /> {adding ? "Checking…" : "Add model"}</button>
          <small>{addNote ?? (MODEL_SOURCES.find((item) => item.key === addSource)?.local
            ? "Open-weight model: downloaded by its first run on one A100, energy measured."
            : `API model: runs through the logging proxy, capped at $${catalog.limits.max_run_cost_usd} per run; energy is not measured (inference runs at the provider). `
              + (addSource === "openrouter" ? "Prices come from OpenRouter unless you enter them." : "Prices are needed for the spending cap."))}</small>
        </div>
        {system?.multi_model && (
          <div className="run-operators">
            <label className="run-mode">Operator models
              <select value={multiModel ? "multi" : "single"} onChange={(event) => update({ operator_models: event.target.value === "multi" ? [] : null })}>
                <option value="single">Single model</option>
                <option value="multi">Multi-model (compare optimizers)</option>
              </select>
            </label>
            {multiModel && (
              <div className="run-operator-list">
                <span>Each semantic operator is offered once per selected model (pick at least two; list order = cheapest first). The model above only plans.</span>
                {systemModels.filter((item) => item.credential_available).map((item) => (
                  <label key={item.key}>
                    <input type="checkbox" checked={form.operator_models?.includes(item.key) ?? false} onChange={() => toggleOperatorModel(item.key)} />
                    {item.display_name}
                  </label>
                ))}
              </div>
            )}
          </div>
        )}
        <div className="run-summary">
          <div>
            <span>Estimated cost</span>
            <strong className={overLimit ? "run-warn" : undefined}>{money(estimate?.total_usd)}</strong>
            <small>{estimate?.total_usd == null ? "No archived token usage for this query yet." : `${estimate.basis}${overLimit ? " · over the limit" : ""}`}</small>
          </div>
          <div><span>Energy source</span><small>{energyNote(model)}</small></div>
          <div><span>About</span><small>{system?.description}{system?.notes ? ` ${system.notes}` : ""}{workload?.preparation ? ` Requires ${workload.preparation} on the server.` : ""}</small></div>
          <div className="run-actions">
            <button className="primary-action" disabled={!runnable || submitting || overLimit} onClick={() => void submit()}>
              <Play size={14} /> {submitting ? "Starting…" : "Start run"}
            </button>
            {lastRun && ACTIVE.has(lastRun.status) && stopButton(lastRun, "Stop run")}
          </div>
        </div>
        {error && <p className="run-error">{error}</p>}
      </div>

      <div className="panel run-table">
        <div className="panel-heading"><div><span className="eyebrow">RUNS</span><h3>Recent jobs</h3></div>
          <div className="run-layout">
            <span>{jobs.filter((job) => ACTIVE.has(job.status)).length} active</span>
            <button className={layout === "cards" ? "active" : ""} title="Cards" aria-label="Show as cards" onClick={() => setLayout("cards")}><LayoutGrid size={14} /></button>
            <button className={layout === "table" ? "active" : ""} title="Table" aria-label="Show as table" onClick={() => setLayout("table")}><List size={14} /></button>
          </div>
        </div>
        {layout === "cards" && (
          <div className="run-cards">
            {cardJobs.length === 0 && <p className="run-empty">No runs yet.</p>}
            {cardJobs.map((job) => (
              <JobCard key={job.id} job={job} progress={progress[job.id]} selected={job.id === selectedJob}
                title={`${catalog.systems.find((item) => item.key === job.spec.system)?.label ?? job.spec.system} · ${job.spec.query ? queryLabel(job.spec.query) : "all queries"}`}
                subtitle={`${catalog.models.find((item) => item.key === job.spec.model)?.display_name ?? job.spec.model} · started ${time(job.created_at)}`}
                onSelect={() => setSelectedJob(job.id)} onInspect={() => { setSelectedJob(job.id); setInspect(job); }}
                stop={ACTIVE.has(job.status) ? stopButton(job) : null} />
            ))}
          </div>
        )}
        {layout === "table" && <>
        <div className="table-head run-row"><span>Started</span><span>System</span><span>Workload · query</span><span>Model</span><span>Status</span><span>Records</span><span /></div>
        {jobs.length === 0 && <p className="run-empty">No runs yet.</p>}
        {jobs.map((job) => (
          <div key={job.id} role="button" tabIndex={0} className={`table-row run-row${job.id === selectedJob ? " active" : ""}`}
            onClick={() => setSelectedJob(job.id)}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedJob(job.id); } }}>
            <span>{time(job.created_at)}</span>
            <span>{catalog.systems.find((item) => item.key === job.spec.system)?.label ?? job.spec.system}</span>
            <span>{catalog.workloads.find((item) => item.key === job.spec.workload)?.dataset ?? job.spec.workload} · {job.spec.query ? queryLabel(job.spec.query) : "all"}</span>
            <span>{catalog.models.find((item) => item.key === job.spec.model)?.display_name ?? job.spec.model}{job.spec.operator_models ? ` · operators: ${job.spec.operator_models.split(",").map((key) => catalog.models.find((item) => item.key === key)?.display_name ?? key).join(" + ")}` : ""}</span>
            <span className={`run-status ${job.status}`}>{job.status}{job.step ? ` · ${job.step}` : ""}</span>
            <span>{job.status === "succeeded" ? `${job.record_count} · ${job.validation_status ?? ""}` : job.error ?? "—"}</span>
            <span>{ACTIVE.has(job.status) && stopButton(job)}</span>
          </div>
        ))}
        </>}
      </div>

      {inspect && (
        <div className="panel run-log-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">RUN INSPECTOR</span><h3>{inspect.id}</h3></div>
            <button className="secondary-action" onClick={() => setInspect(null)}>Close</button>
          </div>
          <RunInspector source={{ kind: "live", jobId: inspect.id, system: inspect.spec.system, query: inspect.spec.query ?? "" }} />
        </div>
      )}

      {selected && (
        <div className="panel run-log-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">LIVE LOG</span><h3>{selected.id}</h3></div>
            {ACTIVE.has(selected.status) && stopButton(selected, "Stop run")}
          </div>
          <div className="run-log" role="log" aria-live="polite">
            {log.map((line, index) => <div key={index}>{line}</div>)}
            <div ref={logEnd} />
          </div>
        </div>
      )}
    </div>
  );
}
