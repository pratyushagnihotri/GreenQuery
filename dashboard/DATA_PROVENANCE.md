# Archived workbench data provenance

The dashboard imports these eight compact snapshots through `src/data/benchmarks.ts`. Hashes are SHA-256 of the imported JSON bytes. These are **archived records**, not 592 universally accepted answers. The reconstruction audit classified 540 records as validated and 52 as validated-partial; the latter remain visible in the UI and must be filtered before scientific comparison. The dashboard's `verify-data` script checks schema/coverage, not raw-to-compact provenance or ground-truth correctness.

| Snapshot in `src/data/` | SHA-256 | Source relation |
| --- | --- | --- |
| `animals_query_metrics.json` | `d09a139451cfbaba10445b6056c78f3857d7369f8381a0b26ca24fe5f7b298b9` | Byte-identical to local Animals compact file |
| `artwork_query_metrics.json` | `c3fadeffb277995c829b7598eea2de82ad3009d37a2aca8d93a96d9b66329f5d` | Workbench snapshot; raw/compact lineage documented in project reconstruction, not bundled here |
| `medical_online_query_metrics.json` | `88fbe322ec26de8d54cfcbd05781fbe06ab9099936a4cd88ee1543005240c37c` | Byte-identical to API Medical compact file |
| `medical_query_metrics.json` | `e33c42386ba38c6706b7cd7584206ef1f8e3f3b4c88d9bf014410847f9bf9426` | Byte-identical to local Medical compact file |
| `movie_local_query_metrics.json` | `c44b12b04589984a8adc1e2f5afc7abdc26ed856eccdd4f9261cb3e73ddda1b1` | Byte-identical to local Movie compact file |
| `movie_query_metrics.json` | `125fd921ece5686196d4e3afd5b1bfce3c684e7882a8adb267d3cb951257ae9b` | Byte-identical to API Movie compact file |
| `rotowire_online_query_metrics.json` | `3b1a22fecefa883de240c2dce57005c973a9375336298fe556fba233791f6fcc` | Workbench snapshot; raw/compact lineage documented in project reconstruction, not bundled here |
| `rotowire_query_metrics.json` | `0e94b5c89ca07f3b84617dd54f3cb562223bfee411ad85ad9b92b2b65df54e2c` | Byte-identical to local RotoWire compact file |

Coverage is 400 local and 192 API records across five workloads, with 160 structured Movie plans. Local NVML values represent measured GPU operational energy; API EcoLogits values are estimates, not provider-hardware measurements. Local RotoWire R2 quality (20 records) and online RotoWire quality (32 records) remain partial/qualified; do not call all archive values validated accuracy. Missing telemetry must not be interpreted as zero. Different workloads' quality metrics are not interchangeable.

The original source branches and DGX artifacts remain the provenance of record. The dashboard is an archived visual workbench, not the experiment runner or final scientific evidence gate. Refreshing a snapshot requires a separate validated run, compact rebuild, intentional file replacement, and `npm run verify-data`.
