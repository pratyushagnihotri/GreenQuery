"""Asynchronous job execution with live log/status events."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from greenquery_server import settings
from greenquery_server.adapters import SYSTEMS, RunSpec, Step
from greenquery_server.adapters.stretto import embedding_choice, latest_measurement, run_dir_for
from greenquery_server.catalog import WORKLOADS
from greenquery_server.store import Store
from reasondb.reasoning.providers import get_model_spec

TERMINAL = {"succeeded", "failed", "cancelled"}


def resolve_arg(arg: str) -> str:
    """Expand placeholders for files that only exist after the runner finished.

    ``$LATEST:<workload>:<run_dir>`` -> newest raw measurement in run_dir
    ``$SUMMARY:<placeholder>``       -> that measurement's ``*_summary.json``
    """
    if arg.startswith("$SUMMARY:"):
        measurement = Path(resolve_arg(arg.removeprefix("$SUMMARY:")))
        return str(measurement.with_name(measurement.stem + "_summary.json"))
    if arg.startswith("$LATEST:"):
        _, workload, run_dir = arg.split(":", 2)
        return str(latest_measurement(Path(run_dir), workload))
    return arg


class JobManager:
    def __init__(self, store: Store):
        self.store = store
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.subscribers: Dict[str, List[asyncio.Queue]] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.cancelled: set[str] = set()
        self.workers: List[asyncio.Task] = []

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        resumable = self.store.mark_interrupted()
        # SLURM jobs mostly wait in the cluster queue, so the server can watch several.
        workers = settings.MAX_CONCURRENT_JOBS if settings.EXECUTOR == "local" else max(8, settings.MAX_CONCURRENT_JOBS)
        for _ in range(max(1, workers)):
            self.workers.append(asyncio.create_task(self._worker()))
        for job in resumable:  # cluster jobs keep running across server restarts
            self.workers.append(asyncio.create_task(self._resume_slurm(job)))

    async def stop(self) -> None:
        for process in self.processes.values():
            process.kill()
        for worker in self.workers:
            worker.cancel()

    async def submit(self, job_id: str) -> None:
        await self.queue.put(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self.store.job(job_id)
        if not job or job["status"] in TERMINAL:
            return False
        self.cancelled.add(job_id)
        if job.get("slurm_job_id"):
            await self._command(["scancel", job["slurm_job_id"]])
        process = self.processes.get(job_id)
        if process and process.returncode is None:
            process.terminate()
        if job["status"] == "queued":
            self._set(job_id, status="cancelled", step=None)
        return True

    # ----------------------------------------------------------------- events
    def job_dir(self, job_id: str) -> Path:
        return settings.JOBS_DIR / job_id

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    def _publish(self, job_id: str, event: Dict[str, Any]) -> None:
        for queue in self.subscribers.get(job_id, []):
            queue.put_nowait(event)

    def _log(self, job_id: str, line: str) -> None:
        path = self.log_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(line + "\n")
        self._publish(job_id, {"type": "log", "line": line})

    def _set(self, job_id: str, **fields: Any) -> None:
        self.store.update_job(job_id, **fields)
        self._publish(job_id, {"type": "status", "job": self.store.job(job_id)})

    async def events(self, job_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Replay the existing log, then stream new lines and status changes."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.setdefault(job_id, []).append(queue)
        try:
            yield {"type": "status", "job": self.store.job(job_id)}
            if self.log_path(job_id).exists():
                for line in self.log_path(job_id).read_text().splitlines():
                    yield {"type": "log", "line": line}
            while True:
                job = self.store.job(job_id)
                if job is None or job["status"] in TERMINAL and queue.empty():
                    yield {"type": "status", "job": job}
                    return
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield {"type": "ping"}
        finally:
            self.subscribers[job_id].remove(queue)

    # -------------------------------------------------------------- execution
    async def _worker(self) -> None:
        while True:
            job_id = await self.queue.get()
            try:
                if job_id not in self.cancelled:
                    await self._run(job_id)
            except Exception as error:  # keep the worker alive for the next job
                self._log(job_id, f"[server] job crashed: {error!r}")
                self._set(job_id, status="failed", error=str(error), step=None)
            finally:
                self.processes.pop(job_id, None)
                self.queue.task_done()

    async def _run_step(self, job_id: str, step: Step) -> int:
        argv = [resolve_arg(arg) for arg in step.argv]
        env = {**os.environ, **step.env, "PYTHONUNBUFFERED": "1"}
        self._log(job_id, f"[server] $ {' '.join(argv)}")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=settings.REPO_ROOT,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.processes[job_id] = process
        assert process.stdout is not None
        async for raw in process.stdout:
            self._log(job_id, raw.decode(errors="replace").rstrip())
        return await process.wait()

    async def _run(self, job_id: str) -> None:
        job = self.store.job(job_id)
        assert job is not None
        spec = RunSpec(**job["spec"])
        adapter = SYSTEMS[spec.system]
        workload = WORKLOADS[spec.workload]
        if settings.EXECUTOR == "slurm":
            await self._run_slurm(job_id, spec, adapter)
            return
        self._set(job_id, status="running", step="starting")
        self._log(job_id, f"[server] {adapter.label} · {workload.dataset} · {spec.query or 'all queries'} · "
                          f"{spec.model} · {spec.repetitions} repetition(s) · cache {spec.cache_policy}")
        validations: List[str] = []
        for repetition in range(1, spec.repetitions + 1):
            run_dir = run_dir_for(self.job_dir(job_id), spec.model, repetition)
            run_dir.mkdir(parents=True, exist_ok=True)
            for step in adapter.steps(spec, run_dir, repetition):
                if job_id in self.cancelled:
                    self._set(job_id, status="cancelled", step=None)
                    return
                self._set(job_id, step=f"{step.name} ({repetition}/{spec.repetitions})")
                try:
                    code = await self._run_step(job_id, step)
                except FileNotFoundError as error:
                    code = 1
                    self._log(job_id, f"[server] {error}")
                if job_id in self.cancelled:
                    self._set(job_id, status="cancelled", step=None)
                    return
                if step.validation:
                    validations.append("validated" if code == 0 else "validation_failed")
                if code != 0 and step.fatal:
                    self._set(job_id, status="failed", step=None, error=f"{step.name} exited with code {code}")
                    return
        for step in adapter.finalize_steps(spec, self.job_dir(job_id)):
            self._set(job_id, step=step.name)
            code = await self._run_step(job_id, step)
            if code != 0:
                self._set(job_id, status="failed", step=None, error=f"{step.name} exited with code {code}")
                return
        validation_status = (
            "validated" if validations and all(v == "validated" for v in validations)
            else "validation_failed" if validations else "not_validated"
        )
        records = adapter.collect(spec, self.job_dir(job_id))
        self._store_records(job_id, spec, adapter, records, validation_status)

    def _store_records(self, job_id: str, spec: RunSpec, adapter, records: List[dict], validation_status: str) -> None:
        workload = WORKLOADS[spec.workload]
        model = get_model_spec(spec.model)
        stretto_engine = spec.system.startswith("stretto")
        enriched = [
            {
                **record,
                "run": f"live-{job_id.rsplit('-', 1)[-1]}-{int(record.get('run', 1)):02d}"
                if str(record.get("run", "")).isdigit() else f"live-{job_id.rsplit('-', 1)[-1]}",
                "system": spec.system,
                "system_label": adapter.label,
                "provider": model.provider if model else "openai",
                "deployment_label": model.deployment if model else "Online API",
                "ecologits_model": model.ecologits_model if model and model.energy == "ecologits_proxy" else None,
                "validation_status": validation_status,
                "embedding_model": embedding_choice(spec.model) if stretto_engine else None,
                "operator_mode": record.get("operator_mode") or ("multi_model" if spec.operator_models else "single_model"),
                "operator_models": record.get("operator_models") or (spec.operator_models.split(",") if spec.operator_models else None),
            }
            for record in records
        ]
        count = self.store.add_records(job_id, workload.dataset, enriched)
        (self.job_dir(job_id) / "records.json").write_text(json.dumps(enriched, indent=2))
        self._log(job_id, f"[server] stored {count} record(s); validation: {validation_status}")
        self._set(job_id, status="succeeded", step=None, record_count=count, validation_status=validation_status)


    # ------------------------------------------------------------------ slurm
    async def _command(self, argv: List[str], env: Optional[Dict[str, str]] = None) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            # uvloop (uvicorn[standard]) requires a plain dict, not os.environ.
            *argv, cwd=settings.REPO_ROOT, env=dict(env or os.environ),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await process.communicate()
        return process.returncode, output.decode(errors="replace").strip()

    async def _run_slurm(self, job_id: str, spec: RunSpec, adapter) -> None:
        job_dir = self.job_dir(job_id)
        out_dir = job_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        slurm = adapter.slurm_job(spec, out_dir)
        if slurm is None:
            self._set(job_id, status="failed", error=f"{adapter.label} cannot run under SLURM", step=None)
            return
        # Values go through the environment: MODELS/UNITS contain spaces and commas,
        # which sbatch --export=KEY=value cannot carry.
        env = {**os.environ, **slurm.env, slurm.out_dir_env: str(out_dir)}
        argv = [
            "sbatch", "--parsable", f"--job-name=gq-ui-{job_id.rsplit('-', 1)[-1]}",
            f"--gres={slurm.gres}", f"--output={job_dir / 'slurm.out'}", "--export=ALL",
            # GPU jobs often wait for resources; lower their priority so CPU-only (API)
            # jobs are not stuck behind them in SLURM's priority order.
            *(["--nice=100"] if slurm.gres != "none" else []),
            slurm.script,
        ]
        self._log(job_id, f"[server] {adapter.label} · {spec.workload} · {spec.query or 'all queries'} · {spec.model}"
                          f" · {spec.repetitions} repetition(s)"
                          + (f" · operator models {spec.operator_models}" if spec.operator_models else ""))
        self._log(job_id, f"[server] $ {' '.join(argv)}  ({', '.join(f'{k}={v}' for k, v in slurm.env.items() if 'KEY' not in k)})")
        code, output = await self._command(argv, env)
        if code != 0:
            self._log(job_id, output)
            self._set(job_id, status="failed", error=f"sbatch failed: {output[-300:]}", step=None)
            return
        slurm_id = output.splitlines()[-1].split(";")[0]
        self._set(job_id, status="running", step="queued on SLURM", slurm_job_id=slurm_id)
        self._log(job_id, f"[server] submitted SLURM job {slurm_id}")
        await self._watch_slurm(job_id, spec, adapter, slurm_id)

    async def _resume_slurm(self, job: Dict[str, Any]) -> None:
        spec = RunSpec(**job["spec"])
        self._log(job["id"], f"[server] resumed watching SLURM job {job['slurm_job_id']} after restart")
        try:
            await self._watch_slurm(job["id"], spec, SYSTEMS[spec.system], job["slurm_job_id"])
        except Exception as error:
            self._set(job["id"], status="failed", error=str(error), step=None)

    async def _watch_slurm(self, job_id: str, spec: RunSpec, adapter, slurm_id: str) -> None:
        log_file = self.job_dir(job_id) / "slurm.out"
        offset = 0
        pending = True  # also clears a stale "waiting" step of a job resumed after a restart
        while True:
            _, state = await self._command(["squeue", "-j", slurm_id, "-h", "-o", "%T"])
            if log_file.exists():
                with log_file.open("rb") as handle:
                    handle.seek(offset)
                    chunk = handle.read()
                    offset += len(chunk)
                for line in chunk.decode(errors="replace").splitlines():
                    self._log(job_id, line)
                    if line.startswith(("--- ", "=== ")):
                        self._set(job_id, step=line[4:].strip()[:120])
            if not state or state.startswith("slurm_load_jobs error"):
                break
            if state == "PENDING":
                self._set(job_id, step="queued on SLURM (waiting for resources)")
                pending = True
            elif pending:
                if "waiting for resources" in ((self.store.job(job_id) or {}).get("step") or ""):
                    self._set(job_id, step="running on SLURM")
                pending = False
            await asyncio.sleep(settings.SLURM_POLL_SECONDS)
        _, final = await self._command(["sacct", "-j", slurm_id, "-X", "-n", "-P", "-o", "State,ExitCode"])
        final_state = (final.splitlines() or ["UNKNOWN|"])[0]
        if job_id in self.cancelled or final_state.startswith("CANCELLED"):
            self._set(job_id, status="cancelled", step=None)
            return
        if not final_state.startswith("COMPLETED"):
            self._set(job_id, status="failed", step=None, error=f"SLURM job {slurm_id}: {final_state}")
            return
        out_dir = self.job_dir(job_id) / "out"
        records = adapter.collect_slurm(spec, out_dir)
        validation_status = self._slurm_validation(out_dir, records)
        self._store_records(job_id, spec, adapter, records, validation_status)

    @staticmethod
    def _slurm_validation(out_dir: Path, records: List[dict]) -> str:
        status_file = out_dir / "status.tsv"
        if status_file.exists():  # Stretto evaluation script: one row per query run
            rows = [line.split("\t") for line in status_file.read_text().splitlines()[1:] if line.strip()]
            if not rows:
                return "not_validated"
            return "validated" if all(len(r) > 4 and r[4] == "validated" for r in rows) else "partially_validated"
        if records:  # SemBench: scored by its own evaluator
            return "sembench_evaluated"
        return "not_validated"


def spec_to_dict(spec: RunSpec) -> Dict[str, Any]:
    return asdict(spec)


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(3).hex()
