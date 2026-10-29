# Measured joules expose what semantic optimizers ignore

No published semantic query system (Stretto, LOTUS, Palimpzest/Abacus, CAESURA, ThalamusDB, DocETL, ZenDB, UQE, Nirvana) measures, models, or optimizes energy or CO2. They optimize dollars, LLM calls or tokens, latency, and quality. The database energy literature that does measure energy (2009–2025) only covers CPU/DRAM relational operators, measured at the wall or through RAPL and split by utilization regressions. The only DB-venue paper that measures LLM-as-operator energy is EcoLLM (aiDM '26), which is estimation-based, works per task, and is not tied to optimizer plan choices. The gap GreenQuery targets is therefore real, but it is narrower than "nobody measures LLM energy". The ML-systems community already has validated GPU sensor protocols, per-request Shapley attribution under batching, and energy-aware model routers. The defensible novelty lies in measuring and attributing energy inside multi-operator semantic plans with quality guarantees, not in LLM energy measurement as such. GreenQuery's current measurements are precise but not yet shown to be accurate. Its own regression (energy ≈ 180 W × runtime, R² 0.975) means energy currently adds almost no information beyond latency. EcoLogits, applied to Stretto's 97–100%-input-token calls, is structurally unable to tell plans apart, because its GPU term ignores input tokens. The strongest contributions are therefore ones that (a) close the measurement boundary and separate static from dynamic energy, (b) find plan spaces where energy and latency rankings actually diverge, and (c) replace output-token estimators with input-aware ones. A pure "swap the optimizer's cost function for joules" paper carries high novelty risk.

*Verification tags follow the underlying notes and are not upgraded here. **[V-full]** means the primary text was read. **[V-abs]** means only the abstract or metadata was seen. **[V-ref]** means the work was seen only in another paper's reference list. **[S]** means search-snippet or secondary-source only. **[U]** means background knowledge not checked in this research. Where two notes disagreed, the higher verification level comes from the note that read the full text (EcoLLM and Bachras & Jacobsen were read in full in one note but not parsed in another).*

## Three literatures converge, but none measures semantic operators

### Database energy research stopped at CPU operators and utilization regressions

The founding wave of DBMS energy work measured whole servers and then modeled the energy of individual operators. Harizopoulos et al. argued at CIDR 2009 that data-management software, not hardware alone, holds the key to efficiency ([arXiv:0909.1784](https://arxiv.org/abs/0909.1784)) [V-abs]. Tsirogiannis et al. (SIGMOD 2010) then showed that **CPU power for different operators can differ by up to 60% at the same CPU utilization** and does not scale linearly with utilization ([DBScholar](https://rmarcus.info/dbscholar/papers/h71017401299a604a)) [V-abs]. That result undermines every utilization-only attribution model that followed. Lang et al. (PVLDB 2012) are the methodological reference point. They used HP iLO2 readings to fit SysPower = 130.03·C^0.2369, validated it against a WattsUp Pro wall meter at 1 Hz and ±1.5%, and modeled hash-join build and probe phases to **within 5–10% error on normalized energy**. They concluded that "optimizing for parallel DBMS performance does not always result in the lowest energy consumed per query" ([arXiv:1208.1933](https://arxiv.org/pdf/1208.1933)) [V-full].

Energy-aware optimizers also exist, so they are prior art for any "energy-objective optimizer" claim:

- USF's PET, a VLDB 2012 demo, placed a power cost estimator inside PostgreSQL with a DBA-controlled energy/performance knob ([VLDB](https://www.vldb.org/pvldb/volumes/5/paper/PET%3A%20Reducing%20Database%20Energy%20Cost%20via%20Query%20Optimization)) [V-abs].
- The same group's IEEE TC 2015 work estimates relational-operator energy online from resource-consumption models ([USF TC15](https://cse.usf.edu/~tuy/pub/TC15.pdf)) [S].
- Psaroudakis et al. (DaMoN 2014) scheduled main-memory operators using per-operator calibrated hardware energy models and reported **up to 4× efficiency gains** for scans ([EPFL Infoscience](https://infoscience.epfl.ch/record/198730)) [V-abs].

Every one of these models is CPU-bound, driven by cardinality, and deterministic. LLM operators break all three assumptions: they run on GPUs or remote APIs, their cost scales with tokens rather than tuples, and their outputs are stochastic.

The 2024–2026 database revival shifted from energy to **carbon, both operational and embodied, plus water**, but kept the CPU boundary:

- **EcoQuery/ATLAS** (Bachras & Jacobsen, arXiv 2504.18980) measures RAPL package and DRAM only. It explicitly excludes GPUs and cooling, and treats DRAM RAPL as relative-only because of a constant offset ([arXiv](https://arxiv.org/html/2504.18980)) [V-abs+html].
- **Bodner et al.** (CIDR 2025) use turbostat/RAPL and a linear TDP-based utilization model for lifecycle carbon. They find that efficiency gains lag performance gains (**1.48× vs 4.44× on TPC-H** over nine years) ([CIDR 2025](https://vldb.org/cidrdb/papers/2025/p14-bodner.pdf)) [V-full].
- **Bachras & Jacobsen's PVLDB 2025 vision** calls for "work per Joule" at the query level and for carbon and water as first-class optimizer costs. It notes that carbon-aware query scheduling has "few database-specific implementations", warns that grid carbon-accounting schemes can reverse which scheduler looks best, and does not mention LLM operators ([PVLDB 18(11)](https://www.vldb.org/pvldb/vol18/p4064-bachras.pdf)) [V-full].

GPU-database research after 2015 reports performance but not energy (e.g., Crystal, SIGMOD 2020 ([MIT DSpace](https://dspace.mit.edu/entities/publication/030e1e65-9001-4291-9387-063a5e4d8fd7))). No DB-venue study of in-database ML energy was found, but the search was limited and that absence is not confirmed.

### Semantic query systems price everything in dollars, calls, and seconds

Each semantic system names its objective explicitly, and energy is never among them:

- **LOTUS** (PVLDB 18, 2025) minimizes LLM calls and runtime under statistical accuracy guarantees ([PVLDB](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf)) [V].
- **Palimpzest** (CIDR 2025) trades runtime, financial cost, and quality ([CIDR](https://vldb.org/cidrdb/2025/palimpzest-optimizing-ai-powered-analytics-with-declarative-query-processing.html)) [V].
- **Abacus** (arXiv 2505.14661, listed as PVLDB 19) optimizes one of quality, cost, or latency under constraints on the others ([arXiv](https://arxiv.org/pdf/2505.14661)) [V-arXiv; PVLDB listing S].
- **ThalamusDB** (SIGMOD 2024) exposes approximation error, computation time, and labeling overhead ([NSF PAR](https://par.nsf.gov/servlets/purl/10577787)) [V].
- **CAESURA** (CIDR 2024) is an LLM planner with no cost optimizer ([CIDR](https://vldb.org/cidrdb/2024/caesura-language-models-as-multi-modal-query-planners.html)) [V].
- **Stretto** (arXiv 2602.04430, no peer-reviewed venue found) minimizes a per-tuple cost Σ unsure·cost_o under Bayesian precision/recall bounds, using a gradient-based optimizer and KV-cache-derived operator variants. It runs Llama/Llava 8B–72B on **7× A100 80GB**, reports runtime and precision/recall "Target Met", and states that it does not measure monetary cost. It never mentions energy ([arXiv HTML](https://arxiv.org/html/2602.04430v2)) [V-arXiv].
- **SemBench** (PVLDB 19) is the de facto cross-system benchmark and targets LLM calls, input tokens, model size, and accuracy. Whether it contains any energy metric is **unconfirmed** because its PDF could not be parsed ([PVLDB](https://www.vldb.org/pvldb/vol19/p1754-trummer.pdf)).

Before any "first" claim, grep SemBench for "energy" and "carbon", and check industry AI-SQL functions (BigQuery, Snowflake Cortex, Databricks). Neither check has been done.

Systems that run local open models (Stretto, ELEET, ThalamusDB) fall back to runtime as their cost proxy, because API dollars are meaningless for them. This is exactly where GPU joules would be the natural physical cost, and none of them reports it.

### The ML-systems community already occupies the per-request and routing ground

GreenQuery's novelty claims face the most risk here. On the routing side:

- **GreenServ** (arXiv 2601.17551) routes prompts across a model pool with a contextual bandit and an accuracy-energy knob λ. It reports **−31% cumulative energy and +22% accuracy** versus random routing ([arXiv](https://arxiv.org/html/2601.17551v2)) [V-arXiv].
- **"Measured Joules, Learned Routes"** (arXiv 2609.23085) profiles per-query GPU energy and stresses separating router overhead from model energy ([arXiv](https://arxiv.org/html/2609.23085)) [V-arXiv].
- **Carbon-aware routers** (GAR, arXiv 2605.11603; edge-cloud function calling, arXiv 2609.13559) add grid intensity ([arXiv](https://arxiv.org/pdf/2609.13559)) [V-arXiv].

On the attribution side:

- **JouleShare** (arXiv 2608.00026, preprint) computes exact Shapley energy per request under vLLM batching. It finds that **token-proportional attribution deviates from Shapley by normalized L1 ≈ 0.44–0.46**, and that a calibrated predictor reduces this to 0.12–0.18 ([arXiv](https://arxiv.org/abs/2608.00026)) [V-abs].
- **WattsOnLLM** (APNet 2026) argues for phase-aware Shapley attribution in production serving ([APNet slides](https://conferences.sigcomm.org/events/apnet2026/slides/Session%209_%20Next%20Gen%20Internet%20and%20Cloud/Fine-Grained%20Energy%20Accounting%20in%20Production%20LLM%20Serving%20%20%20apnet2026-6.pdf)) [S].

Taken together, these papers cover "energy-aware model selection" and "per-request energy attribution" for single prompts. They leave open the multi-operator relational plan, per-tuple quality constraints, and cross-tuple sharing (cascades, KV-cache reuse), which are what semantic optimizers actually decide.

| Prior work | What it covers | Boundary / method | What it leaves open for GreenQuery |
|---|---|---|---|
| Lang et al. 2012 [V-full] | Per-query, per-join-phase energy, clusters | Wall meter + iLO2 utilization model | GPUs, LLM operators, token-driven cost |
| PET 2012, Psaroudakis 2014 [V-abs] | Energy-aware optimizer/scheduler | CPU per-operator power models | Non-deterministic GPU/API operators |
| EcoQuery 2025 [V-abs], Bodner 2025 [V-full] | Per-query operational + embodied carbon | RAPL CPU+DRAM, ACT, Electricity Maps | Accelerator energy entirely |
| EcoLLM 2026 [V-full] | Per-task LLM data-processing energy | CodeCarbon-style + EcoLogits estimates | Sensor measurement, plans, optimizer choices |
| GreenServ, Measured Joules 2026 [V-arXiv] | Energy-aware model routing | NVML-profiled per-prompt energy | Multi-operator plans with quality bounds |
| JouleShare 2026 [V-abs] | Per-request Shapley under batching | vLLM replay, GPU power | Cross-tuple sharing in semantic plans, non-GPU energy |

## GreenQuery's numbers are precise, but not yet accurate

### A 2% coefficient of variation proves repeatability, not correctness

Yang, Adámek and Armour (arXiv 2312.02741; the DOI indicates SC24) measured over 70 GPUs and found that on **A100 and H100, nvidia-smi `power.draw` averages only about 25 ms of every 100 ms update**. Naive single-run energy errors reached about 70%. Their protocol (≥32 iterations or ≥5 s, phase-shifted delays, multiple trials, discarding rise time) **cut average error from 39.3% to 4.9%** against external instrumentation ([arXiv v3](https://arxiv.org/html/2312.02741v3)) [V-full; venue partially verified].

GreenQuery polls every 0.5 s on what is most likely an A100-class DGX, since Stretto's evaluation used A100s. Each sample is therefore one 25 ms window per 500 ms, so **roughly 5% of runtime is actually observed** (inference from Yang et al.'s figures). Over multi-second query windows with fairly flat power, this sampling is close to unbiased, which explains the ~2% CV. It cannot resolve individual LLM calls. Whether the totals are accurate depends on whether call timing aliases with the sampling phase, and nothing has tested that. Yang et al. also report a steady-state sensor error of about ±5% of reading, not NVIDIA's documented ±5 W.

The newly added NVML cumulative energy counter is the better primary signal. It is cumulative per device in millijoules, so differencing it around operator boundaries avoids the sampling gaps. Its internal update interval is 20–100 ms ([PETSc docs](https://petsc.org/main/manualpages/Log/PetscLogGpuEnergyMeter/)) [S]. **No source found validates the counter against an external meter on A100/H100.** That makes the planned validation job a genuine, citable methodological result if it includes an external or BMC reference, rather than just a cross-check between two NVIDIA sensors.

### The 180 W × runtime fit says energy currently equals latency

With R² = 0.975 against a constant-power model, average power barely varies across GreenQuery queries. **Any energy ranking of plans at this boundary is therefore almost identical to a runtime ranking.** Stated that way, the fit reproduces Tsirogiannis et al.'s 2010 single-server finding that the fastest configuration is also the most efficient, and it is not yet an energy finding.

The steelman is that on a dedicated node, static power is a real cost that the query causes by occupying the machine. Attributional accounting therefore rightly charges it. But the two accounting views answer different questions:

- **Marginal (above-idle) energy** is the right quantity for optimizer decisions.
- **Attributional energy** (including idle) is the right quantity for footprint reporting.

KubeWatt reports static power as an undistributed node-level quantity. JouleShare reportedly integrates only above-idle power [S]. Kepler, in contrast, spreads idle power across processes "by size" and was shown to charge about 6.25 W to each non-running container ([arXiv:2504.10702](https://arxiv.org/html/2504.10702v1)) [V-full]. GreenQuery should report both views side by side.

The finding that about 40% of query energy falls outside LLM inference windows lines up with a NeurIPS 2026 listing reporting that **GPU-only telemetry misses 41–45% of system energy** for LLM agent inference ([NeurIPS 2026](https://neurips.cc/virtual/2026/170717)) [S]. The definitions differ: "outside inference time windows" is not the same as "outside GPU telemetry". The match therefore corroborates that non-inference energy matters, but it does not replicate GreenQuery's number.

Google's full-stack accounting gives the boundary template. Its median prompt is **0.24 Wh comprehensive versus 0.10 Wh accelerator-only**, with explicit buckets for host CPU+DRAM, idle machines, and PUE overhead ([arXiv 2508.15734](https://arxiv.org/html/2508.15734v1)) [V-full]. A GPU-only boundary will understate semantic-query energy, and the gap will be largest for plans heavy in CPU operators, Python extraction, or orchestration.

### EcoLogits cannot see Stretto's main optimization lever

EcoLogits' methodology states that its "energy model does not explicitly include input tokens". GPU energy is #T_out × f_E(P_active, B) at a **fixed batch size of B = 64**, fitted to ML.ENERGY leaderboard data. Prompt length enters only through TTFT in the OpenRouter latency mode, and not at all in the fallback latency mode ([EcoLogits](https://ecologits.ai/latest/methodology/llm_inference/)) [V-full].

For Stretto-style calls with 97–100% input tokens and 1–2 output tokens, the GPU term is therefore nearly constant per call, whatever the prompt length. Plans that cut input tokens, which is the lever semantic optimizers actually pull, **look almost identical in energy under EcoLogits**. This is an invalid use of the tool for ranking plans, not merely an imprecise one.

Measurements show that input tokens do cost energy, at a lower rate than output tokens:

- **Wilkins et al.** (HotCarbon '24) fit e = α0·τ_in + α1·τ_out + α2·τ_in·τ_out with **R² 0.96–0.98** (ANOVA F: input 15.9, output 126.6) ([arXiv 2407.04014](https://arxiv.org/html/2407.04014v1)) [V-full].
- **Caravaca et al.** found that 900 extra input tokens raised energy 2.19× on average, versus 11× for 900 extra output tokens ([arXiv 2511.05597](https://arxiv.org/html/2511.05597v1)) [V-full].

Even these "input-heavy" studies use at least 100 output tokens. No published measurement covers the 1-output-token extreme, where per-call fixed overhead (tokenization, scheduling, kernel launch, HTTP, idle GPU) probably dominates. ML.ENERGY's steady-state, maximum-batch numbers, and therefore EcoLogits' fit, amortize static power over hundreds of concurrent requests. Applied to batch-1 local serving, they **underestimate** per-call energy ([arXiv 2505.06371](https://arxiv.org/html/2505.06371v2)) [V-full].

### Which approaches hold up

| Approach | Verdict for semantic query plans | Basis |
|---|---|---|
| nvidia-smi `power.draw` polled at 0.5 s | Valid only for multi-second aggregates; invalid per call or per short operator | Yang et al. 25/100 ms window on A100/H100 [V-full] |
| NVML cumulative energy counter, differenced at operator boundaries | Best available GPU signal; per-call only by aggregating N calls; accuracy not externally validated | PETSc/Megatron docs [S]; validation gap |
| RAPL package+DRAM | Valid with constant offset on Intel server CPUs since Haswell; inaccurate on AMD Zen 2; root-only since PLATYPUS | Desrochers 2016 [V-full]; Schöne et al. [S]; Lipp et al. [V-full] |
| BMC/Redfish node power | Valid for whole-job totals (<1% total-energy error); unusable per operator (iDRAC updates about once per minute) | Pijnacker et al. [V-full] |
| Utilization-proportional split (Kepler-style) | Acceptable for separating GreenQuery from co-runners; untrustworthy for idle attribution; cannot separate operators inside one Python process | Pijnacker et al. [V-full] |
| Token-proportional per-request split | Invalid under batching (L1 ≈ 0.44 vs Shapley); moot at batch-1 | JouleShare [V-abs] |
| TDP- or FLOP-based estimates | Overestimate by up to 4.1× (TDP); FLOP/utilization estimates "substantially understate" | ML.ENERGY [V-full]; Fernandez et al. ACL 2025 [V-full] |
| EcoLogits for API semantic calls | Invalid for ranking input-heavy plans; usable only as a labeled, provider-side order-of-magnitude estimate | EcoLogits methodology [V-full] |
| Leaderboard per-token figures (ML.ENERGY, AI Energy Score) | Invalid to transfer to batch-1, 1-output-token calls; GPU-only boundary | ML.ENERGY [V-full]; AI Energy Score [S] |
| Carbon: location-based hourly intensity, per-query functional unit | Valid and aligned with GSF SCI; never mix with market-based provider figures (Google uses 94 gCO2e/kWh market-based) | GSF SCI for AI [V-abs]; Google [V-full] |

Two hardware checks are still **[U]** and should be resolved before any CPU-energy claim is published. The first is whether GreenQuery's DGX CPU is AMD EPYC Rome. The notes believe DGX A100 uses it, and if so, Schöne et al.'s warning that Zen 2 RAPL misses data-dependent power applies directly ([Next Platform](https://www.nextplatform.com/2021/08/09/hpc-efficiency-gurus-grapple-with-amds-rapl/)) [S]. The second is whether the DGX BMC exposes sub-second PSU power.

## EcoLLM's hardware description needs reconciling before GreenQuery builds on it

EcoLLM (Agnihotri, Luthra, Binnig; aiDM '26, DOI 10.1145/3814940.3815324) is the closest DB-venue precedent and, as far as these notes show, the only one ([DFKI PDF](https://www-live.dfki.de/fileadmin/user_upload/import/17616_3814940.3815324.pdf)) [V-full]. It contributes an L-shaped energy-accuracy Pareto across 10 local and 3 API models, and shows that **energy and accuracy are not aligned**.

The notes found one concrete inconsistency. **Table 1 lists "CloudLab m510, Intel Xeon D, 48 GB", while §3.1 states "r7525 nodes, dual AMD EPYC 7542, 512 GB, two NVIDIA Tesla V100S".** Only the r7525 description includes the GPUs needed to run the local models. A second point needs clarifying. The energy formula is written as E ≈ Σ P_r·t_r over CPU and GPU, but Table 1 describes the local method as "runtime + CPU utilization". Readers cannot tell whether GPU energy was measured, modeled, or omitted.

Both points can be fixed with an erratum or a sentence in any follow-up. They matter for GreenQuery because a GreenQuery paper will cite EcoLLM as its baseline, and reviewers will check whether the two papers' boundaries are comparable. If §3.1 is correct, the local CPU side ran on AMD Zen 2, where RAPL would have been unreliable anyway. EcoLLM's choice not to rely on RAPL was therefore defensible, but that reasoning should be stated.

EcoLLM's API figures also come from EcoLogits, so the output-token blind spot described above applies to them too. Some of its own numbers deserve a second look in that light. o3-mini at about 5×10⁻⁷ kWh sits **roughly three orders of magnitude below GPT-4.1** (10⁻⁴ to 10⁻³ kWh), even though reasoning-style prompting is reported elsewhere to raise energy by 72–177% ([EuroMLSys 2025](https://euromlsys.eu/pdf/euromlsys25-27.pdf)) [V]. Check whether hidden reasoning tokens and the closed-model parameter-count assumptions entered the estimate. This is a hypothesis about an estimation artifact, not a demonstrated error. Separately, the 365 MWh/yr extrapolation uses the upper end of the GPT-4.1 range.

For positioning, GreenQuery must differentiate from EcoLLM explicitly. The differentiators are sensor measurement instead of estimation, plan- and operator-level granularity instead of per task, and an optimizer in the loop. Otherwise reviewers will read GreenQuery as EcoLLM applied to Stretto.

## Five candidate contributions, ranked by novelty risk

| # | Contribution | Closest prior work | Novelty risk | Core GreenQuery evaluation |
|---|---|---|---|---|
| C1 | Boundary-closed, phase-decomposed energy accounting for semantic plans | Google 2508.15734; Lang 2012 phases; router-overhead accounting (2601.00823, 2609.23085) | Medium (method known, application new) | Energy conservation across operators + gaps vs whole-job NVML+RAPL+BMC |
| C2 | Where energy and latency rankings diverge in semantic plan spaces | Tsirogiannis 2010; Lang 2012; GreenServ; Vidur+Vessim (2507.11417) | Medium, with high null-result risk | Kendall τ between latency and energy orderings over Stretto's plan continuum |
| C3 | Input-aware energy estimator for 1-output-token API operators | EcoLogits; Wilkins 2024; Caravaca 2025; TokenPowerBench | Medium | Held-out-model error on local NVML measurements; ranking fidelity vs EcoLogits |
| C4 | Per-tuple attribution under KV-cache and cascade sharing | JouleShare; WattsOnLLM; LLMVisor; GreenCache | Low–medium (strongest novelty) | Exact marginal/Shapley per tuple by replay vs proportional rules |
| C5 | Energy- or carbon-objective semantic optimizer with deferral | PET 2012; Psaroudakis 2014; Abacus; GreenServ; Bachras vision | High for "swap cost" alone; medium with C3 + carbon deferral | Joules/gCO2e saved at matched Target-Met ratio and latency SLO |

### C1 turns "40% outside inference" into an accounted breakdown

The contribution is a measurement methodology for semantic plans. Every operator invocation is bracketed by monotonic timestamps. NVML energy (all GPUs, including idle reserved ones) and RAPL (all sockets) are differenced at about 10–50 ms. Energy is then reported in buckets:

- optimizer/profiling (Stretto's 15% sample, Abacus's sampling)
- LLM prefill
- non-LLM operators (BLIP filter, generated Python)
- orchestration and waiting
- static versus dynamic energy for each of the above

The design follows region-based counter differencing as in CPPJoules and Megatron-Core ([CPPJoules](https://arxiv.org/html/2412.13555v1)) and Google's component buckets.

What is new is the application and the accounting of optimizer overhead. The routing literature already separates router energy from model energy ([arXiv 2609.23085](https://arxiv.org/html/2609.23085)), so the idea transfers but has not been applied to semantic optimizers. Reviewers will call C1 engineering unless it is validated. The validation should therefore be the headline: the sum of all buckets must match whole-job NVML+RAPL within a stated tolerance, and whole-job totals must be compared against BMC or PDU power so that the remainder (wall − RAPL − ΣNVML) is quantified.

In GreenQuery, run C1 on the existing Movie, RotoWire, and Medical suites with SLURM `--exclusive`, measure idle baselines per device right before each run, and apply Yang et al.'s repetition protocol to short phases. C1 is a prerequisite for C2–C5 and alone is a solid experiments-and-analysis paper, but probably not a full research-track paper.

### C2 asks the question the 180 W fit cannot answer

Under GreenQuery's current setup, energy tracks latency. The interesting scientific question is **where that breaks**. Average power must differ across plans for energy and latency to diverge, and Stretto's plan space contains configurations where it should:

- a 70B model sharded over several A100s versus an 8B model on one
- KV-cache-reusing operator variants versus recomputation
- CPU-only operators (generated Python, embedding filters) versus LLM operators
- local versus API execution, where local power drops toward idle while remote energy appears off-boundary

The closest analogues are Lang et al.'s cluster-level "fastest ≠ greenest" result, GreenServ's per-prompt routing, and a Vidur+Vessim study showing that throughput-optimal LLM configurations do not always minimize energy ([arXiv 2507.11417](https://arxiv.org/html/2507.11417v1)) [V-arXiv]. None of them studies semantic plans under quality constraints.

The evaluation enumerates the plan continuum for each GreenQuery query and computes runtime and energy under both attributional (with idle) and marginal (above-idle) accounting. It then reports Kendall τ between the latency and energy orderings, the shift between the energy-Pareto and latency-Pareto frontiers at fixed precision/recall targets, and the joules left on the table by Stretto's runtime-optimal choice.

The honest risk is a null result for single-GPU plans at batch-1. That null result would still be publishable if framed as evidence that latency is a sufficient energy proxy for this regime, with the boundary conditions under which it stops being one.

### C3 replaces an output-token estimator with an input-aware one

For API-backed semantic operators, nothing can be measured locally. The best available option is a defensible estimate. Calibrate a Wilkins-form model (α0·τ_in + α1·τ_out + α2·τ_in·τ_out + a fixed per-call term) on GreenQuery's own NVML measurements of open models at Stretto's actual prompt-length distribution and 1–2 output tokens. Scale it by parameter-count and PUE assumptions, as EcoLogits does. Report every result as an estimate with an uncertainty band.

The missing piece in the literature is the **per-call fixed term at 1 output token**, which no study measures. Caravaca et al.'s random-forest predictor reached **R² 0.92 and 10.5% MAPE** on a held-out model ([arXiv 2511.05597](https://arxiv.org/html/2511.05597v1)), so predictor accuracy alone is not novel. The novelty lies in covering the prefill-only regime and in showing ranking fidelity. Evaluate by hold-out-model prediction error on local measurements, and by checking whether the estimator preserves the measured energy ordering of plans where EcoLogits collapses them.

Ground truth for closed providers does not exist (OpenAI and Anthropic publish no methodology), so claims must stop at "consistent with local physics", not "correct for GPT-4.1".

### C4 is the contribution the ML-serving literature has not touched

JouleShare and WattsOnLLM attribute energy across *concurrent requests* in a batch. Semantic plans create a different sharing structure: **across tuples within one query**. Stretto's KV-cache reuse means a shared prompt prefix is computed once and reused by later tuples. LOTUS-style cascades mean a proxy model's cost is paid for every tuple while the large model's cost is paid only for the uncertain ones. Stretto's own cost model already encodes this (Σ unsure_{t,i−1}·cost_{oᵢ}). Per-tuple marginal energy therefore depends on tuple order and cache state, and proportional rules will misattribute it in a structured way.

The contribution is a definition (marginal, attributional, and Shapley per tuple and per operator), an exact replay-based ground truth for small inputs (with/without cache, subsets of tuples), and a cheap approximation suitable for an optimizer's cost model. GreenCache's work on carbon-aware KV-cache management ([arXiv 2505.23970](https://arxiv.org/abs/2505.23970v2)) is adjacent but concerns cache sizing, not attribution.

The novelty risk is low, but the relevance risk is real: per-tuple attribution must be shown to change a decision. Example decisions are an optimizer choosing operator order differently, or a chargeback differing materially. In GreenQuery, compute exact per-tuple energy by replay on sampled RotoWire and Movie queries. Report the L1 deviation of time- and token-proportional rules from the exact values (directly comparable to JouleShare's 0.44), and show whether an attribution-aware cost changes Stretto's plan.

### C5 is the obvious paper and the riskiest

Replacing Stretto's cost_{oᵢ} with joules per tuple is a few lines of code. Reviewers will cite PET (2012), which already put energy into a PostgreSQL optimizer. They will note that Abacus already optimizes arbitrary objectives under constraints, and that GreenServ already does energy-aware model selection. On its own, C5 risks rejection as incremental.

It becomes defensible when built on C3 and C4 (a validated, input-aware, sharing-aware energy cost model) and when extended to **carbon-aware deferral**. Bachras & Jacobsen identify carbon-aware query scheduling as a vision with few database implementations ([PVLDB 18(11)](https://www.vldb.org/pvldb/vol18/p4064-bachras.pdf)) [V-full]. LLM-heavy semantic queries are expensive and often batch-tolerant, which makes them a strong first case. Lechowicz et al.'s carbon- and precedence-aware cluster scheduling (CoRR 2025) is [V-ref] and unread, so it must be read before claiming novelty here.

Evaluate C5 on joules and gCO2e saved at matched Target-Met ratios and latency SLOs, using hourly location-based intensity traces. Include a sensitivity analysis showing whether the winning plan changes under different carbon-accounting schemes, which is exactly the fragility Bachras & Jacobsen warn about.

## Conclusion

The field has methods for measuring energy (from HPC and ML), for attributing it (from cloud and serving systems), and for optimizing it (DB optimizers from 2010–2015 and 2026 LLM routers). It has not combined them where semantic query processing makes decisions: per operator, per tuple, under quality constraints. GreenQuery's own data argues that the first deliverable is not an energy-aware optimizer but a demonstration of *when energy carries information beyond latency*. Until average power varies across plans, an energy objective is a relabeled runtime objective, and reviewers who know Tsirogiannis 2010 will say so.

The most defensible path runs in order: C1 as the validated foundation, C2 as the scientific question, C4 as the novel attribution problem unique to semantic plans, and C3 as the fix for API operators. C5 should come only after these, because it inherits all of their validity. Two cheap actions should come first. Grep SemBench and the industry AI-SQL documentation for energy reporting before claiming "first". Close the EcoLLM hardware inconsistency so that GreenQuery's own prior work does not become the weakest citation in its related-work section.
