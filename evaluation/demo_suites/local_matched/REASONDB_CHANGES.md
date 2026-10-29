# ReasonDB changes for local matched demos

This is the concise record of core changes required to make the local DGX experiments reliable. It describes purpose, not every changed file.

## Shared reliability

- Route both planning and execution to the selected local Qwen or Llama model.
- Disable answer/planner caches for cold-cache runs; isolate each job with its own ReasonDB cache/database namespace.
- Start local text and embedding services on job-specific ports and reject an already-busy allocated GPU.
- Record local requests, tokens, direct-QA batch size, NVML energy, CO2, and telemetry sidecars; hydrate sidecars when validating or building compact summaries.
- Split large direct-QA work into safe GPU batches. A batch preserves the logical query and token totals while reducing peak memory.

## Movie

- Run four fixed queries with semantic text operators and deterministic physical-plan configuration.
- Use Q4's corrected Taken 3 positive-ratio workload and deterministic dataset-label scoring, including same-sentiment-pair scoring for Q3.

## RotoWire

- Run both basketball queries: player points/assists and player points where the home team won.
- Keep semantic self-joins deterministic while retaining required database joins.
- Use safe direct-QA batches and the visible `game_id` + player `name` identity for ground-truth matching.
- The currently published RotoWire resource collection predates that scorer. Its energy, CO2, runtime, tokens, and requests are valid; its quality must remain unavailable until recollected with this code.

## Medical

- Prepare image data outside the measurement window and route text work to Qwen/Llama and image work to local LLaVA.
- Require the vision backend for image queries, prevent traditional filters from claiming image work, and configure sole image filters and traditional projects deterministically.
- Record both text and vision batch metadata plus the two-GPU NVML measurement evidence for Q3 and Q9.
