# Green Query Lab: Detailed Execution Plan

**Created:** 2026-05-23
**Last Updated:** 2026-05-23
**Author:** Alankrit Sharma
**Current Branch:** `docs/reproducibility-evaluation-baseline`
**Status:** Phase 1 Complete ✅ | Phase 2 In Progress

---

## 🎉 Phase Completion Status

| Phase | Status | Completion Date | Notes |
|-------|--------|----------------|-------|
| **Phase 1: Infrastructure Setup** | ✅ Complete | 2026-05-23 | All measurement infrastructure implemented and tested |
| **Phase 2: Movie Demo** | 🔄 Ready to Start | - | Infrastructure ready for instrumentation |
| **Phase 3: Artwork Demo** | ⏳ Pending | - | Awaiting Phase 2 completion |
| **Phase 4: Analysis** | ⏳ Pending | - | - |

### Phase 1 Deliverables ✅

**Location:** `project/greenQueryLab/evaluation/`

**Files Created:**
- ✅ `evaluation/measurement/metrics.py` (91 lines) - Data classes for all metric types
- ✅ `evaluation/measurement/wrapper.py` (195 lines) - Measurement orchestration
- ✅ `evaluation/measurement/config.py` (40 lines) - Configuration management
- ✅ `evaluation/measurement/reporters.py` (96 lines) - JSON/CSV/Markdown output
- ✅ `evaluation/scripts/test_measurement.py` (31 lines) - Verification test
- ✅ `evaluation/PHASE_1_COMPLETION_REPORT.md` (485 lines) - Comprehensive report with diagrams

**Dependencies Installed:**
- ✅ EcoLogits 0.10.1 (energy/carbon tracking)
- ✅ tiktoken 0.13.0 (tokenizer)
- ✅ All project requirements (OpenAI, DuckDB, pandas, torch, etc.)

**Test Results:**
- ✅ Verification test passed
- ✅ Sample measurement generated: `test_20260523_223537.json`
- ✅ All metrics captured correctly

**Documentation:**
- ✅ Architecture diagrams (6 Mermaid diagrams)
- ✅ Implementation details
- ✅ Data flow documentation
- ✅ Phase 2 roadmap

**Total Lines of Code:** 453 lines + 485 lines documentation = 938 lines

---

## 📝 Implementation Notes

### What Changed from Original Plan

The original plan (below) proposed creating measurement infrastructure in `reasondb/evaluation/measurement/`.

**Actual Implementation:**
- Created in `evaluation/` (top-level, not under reasondb/)
- More modular design with separate metrics, wrapper, config, and reporters
- Added comprehensive reporting with multiple output formats
- Included detailed completion report with architecture diagrams

**Why the Change:**
- Better separation of concerns
- Easier to maintain and extend
- More flexible for different demo types
- Cleaner project structure

### Next Steps

Phase 2 will instrument the Movie demo using the infrastructure built in Phase 1. The original plan's Phase 3 code can be adapted to use our new measurement framework.

---

## 🎯 Primary Goal

Establish reproducible energy and performance measurements for the Movie and Artwork baseline demos using EcoLogits to track OpenAI API consumption.

---

## 📋 Context Summary

### What You've Achieved
- ✅ Python 3.10 environment with ReasonDB installed
- ✅ Movie and Artwork demos working across team (branch: `demo-fixes-windows-obaid`)
- ✅ OpenAI API configured with GPT-4o-mini
- ✅ DuckDB 1.2.0 compatibility established
- ✅ Evaluation protocol documented in `greenQueryLab/docs/evaluation_protocol.md`

### Current Working Directory
`project/greenQueryLab/` (branch: `docs/reproducibility-evaluation-baseline`)

### Mentor Guidance
1. Start with API models (OpenAI) before local models
2. Establish Movie demo as first baseline (simpler, text-only)
3. Then extend to Artwork demo (multimodal)
4. Use EcoLogits for API energy tracking
5. Focus on reproducible measurements

---

## 🚀 Phase 1: Measurement Infrastructure Setup

### Task 1.1: Create Measurement Module Structure
**Location:** `project/greenQueryLab/reasondb/evaluation/`

Create new directory and files:
```
reasondb/evaluation/measurement/
├── __init__.py
├── energy_tracker.py      # EcoLogits wrapper
├── metrics_collector.py   # Runtime, API calls, tokens
├── result_logger.py       # JSON/CSV output
└── demo_runner.py         # Instrumented demo execution
```

**Why:** Separate measurement logic from demo code for reusability

**Commands:**
```bash
cd project/greenQueryLab
mkdir -p reasondb/evaluation/measurement
touch reasondb/evaluation/measurement/__init__.py
touch reasondb/evaluation/measurement/energy_tracker.py
touch reasondb/evaluation/measurement/metrics_collector.py
touch reasondb/evaluation/measurement/result_logger.py
touch reasondb/evaluation/measurement/demo_runner.py
```

### Task 1.2: Install EcoLogits
```bash
cd project/greenQueryLab
source ../reasondb/reasondb-env/bin/activate  # Use existing Python 3.10 env
python -m pip install ecologits
python -m pip install ecologits[openai]
```

**Verify:**
```bash
python -c "import ecologits; print(ecologits.__version__)"
```

---

## 🔧 Phase 2: Implement Measurement Components

### Task 2.1: Create Energy Tracker (`energy_tracker.py`)

**File:** `reasondb/evaluation/measurement/energy_tracker.py`

```python
from ecologits import EcoLogits
from openai import OpenAI
import time
from typing import Dict, Any

class EnergyTracker:
    """Wrapper for tracking OpenAI API calls with EcoLogits energy metrics"""
    
    def __init__(self):
        EcoLogits.init()
        self.client = OpenAI()
        self.measurements = []
    
    def track_call(self, **kwargs) -> Dict[str, Any]:
        """Track a single OpenAI API call with energy metrics
        
        Args:
            **kwargs: Arguments to pass to OpenAI chat.completions.create()
            
        Returns:
            Dict containing duration, energy, carbon, tokens, and model info
        """
        start_time = time.time()
        response = self.client.chat.completions.create(**kwargs)
        end_time = time.time()
        
        # Extract EcoLogits data
        impacts = response.impacts
        
        measurement = {
            'duration': end_time - start_time,
            'energy_kwh': impacts.energy.value,
            'gwp_kgco2eq': impacts.gwp.value,
            'tokens': response.usage.total_tokens,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'model': kwargs.get('model'),
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def get_total_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics across all tracked calls"""
        if not self.measurements:
            return {}
            
        return {
            'total_calls': len(self.measurements),
            'total_duration': sum(m['duration'] for m in self.measurements),
            'total_energy_kwh': sum(m['energy_kwh'] for m in self.measurements),
            'total_gwp_kgco2eq': sum(m['gwp_kgco2eq'] for m in self.measurements),
            'total_tokens': sum(m['tokens'] for m in self.measurements),
        }
```

### Task 2.2: Create Metrics Collector (`metrics_collector.py`)

**File:** `reasondb/evaluation/measurement/metrics_collector.py`

```python
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import json

@dataclass
class DemoMetrics:
    """Data class for storing demo measurement results"""
    demo_name: str
    query_name: str
    start_time: float
    end_time: float
    total_runtime: float
    api_calls: int
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_energy_kwh: float
    total_gwp_kgco2eq: float
    model_name: str
    result_count: int
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Query: {self.query_name}\n"
            f"Runtime: {self.total_runtime:.2f}s\n"
            f"API Calls: {self.api_calls}\n"
            f"Tokens: {self.total_tokens}\n"
            f"Energy: {self.total_energy_kwh:.6f} kWh\n"
            f"Carbon: {self.total_gwp_kgco2eq:.6f} kgCO2eq\n"
            f"Results: {self.result_count} rows"
        )

class MetricsCollector:
    """Collects and aggregates metrics during demo execution"""
    
    def __init__(self, demo_name: str):
        self.demo_name = demo_name
        self.measurements = []
        self.start_time = None
        self.current_query = None
        self.api_calls = []
        
    def start_measurement(self, query_name: str):
        """Start measuring a new query"""
        self.current_query = query_name
        self.start_time = time.time()
        self.api_calls = []
        
    def record_api_call(self, call_data: Dict[str, Any]):
        """Record data from a single API call"""
        self.api_calls.append(call_data)
        
    def end_measurement(self, result_count: int, errors: Optional[List[str]] = None) -> DemoMetrics:
        """End measurement and create metrics object"""
        end_time = time.time()
        
        metrics = DemoMetrics(
            demo_name=self.demo_name,
            query_name=self.current_query,
            start_time=self.start_time,
            end_time=end_time,
            total_runtime=end_time - self.start_time,
            api_calls=len(self.api_calls),
            total_tokens=sum(c['tokens'] for c in self.api_calls),
            prompt_tokens=sum(c['prompt_tokens'] for c in self.api_calls),
            completion_tokens=sum(c['completion_tokens'] for c in self.api_calls),
            total_energy_kwh=sum(c['energy_kwh'] for c in self.api_calls),
            total_gwp_kgco2eq=sum(c['gwp_kgco2eq'] for c in self.api_calls),
            model_name=self.api_calls[0]['model'] if self.api_calls else 'unknown',
            result_count=result_count,
            errors=errors or []
        )
        
        self.measurements.append(metrics)
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all measurements"""
        if not self.measurements:
            return {}
            
        return {
            'total_queries': len(self.measurements),
            'total_runtime': sum(m.total_runtime for m in self.measurements),
            'total_api_calls': sum(m.api_calls for m in self.measurements),
            'total_tokens': sum(m.total_tokens for m in self.measurements),
            'total_energy_kwh': sum(m.total_energy_kwh for m in self.measurements),
            'total_gwp_kgco2eq': sum(m.total_gwp_kgco2eq for m in self.measurements),
            'avg_runtime_per_query': sum(m.total_runtime for m in self.measurements) / len(self.measurements),
        }
```

### Task 2.3: Create Result Logger (`result_logger.py`)

**File:** `reasondb/evaluation/measurement/result_logger.py`

```python
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from .metrics_collector import DemoMetrics

class ResultLogger:
    """Logs measurement results to JSON and CSV files"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def log_measurement(self, metrics: DemoMetrics):
        """Log single measurement to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metrics.demo_name}_{metrics.query_name}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"✅ Logged measurement to: {filepath}")
            
    def log_batch(self, measurements: List[DemoMetrics], batch_name: str):
        """Log multiple measurements to CSV file"""
        if not measurements:
            print("⚠️  No measurements to log")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=measurements[0].to_dict().keys())
            writer.writeheader()
            for m in measurements:
                writer.writerow(m.to_dict())
        
        print(f"✅ Logged batch to: {filepath}")
        
    def log_summary(self, summary_stats: Dict[str, Any], filename: str):
        """Log summary statistics to JSON"""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"✅ Logged summary to: {filepath}")
```

---

## 🎬 Phase 3: Instrument Movie Demo

### Task 3.1: Create Instrumented Movie Demo

**File:** `demos/movie_measured.py`

```python
"""
Movie Demo with Energy Measurement
Tracks runtime, API usage, energy consumption, and carbon emissions
"""
import logging
from pathlib import Path
from reasondb.database.indentifier import InPlaceColumn
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.evaluation.measurement.metrics_collector import MetricsCollector
from reasondb.evaluation.measurement.result_logger import ResultLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize measurement infrastructure
collector = MetricsCollector("movie_demo")
result_logger = ResultLogger(Path("measurement_results"))

csv = (
    Path(__file__).parents[1]
    / "reasondb"
    / "evaluation"
    / "benchmarks"
    / "files"
    / "reviews_100.csv"
)

print("=" * 80)
print("MOVIE DEMO - ENERGY MEASUREMENT RUN")
print("=" * 80)

with RaccoonDB("movie_sampled", device="cpu") as rc:
    reviews = rc.add_table(
        path=csv,
        table_name="reviews",
        text_columns=[InPlaceColumn("reviews.reviewtext")],
    )

    # ========================================================================
    # Q1: Find 5 clearly positive reviews
    # ========================================================================
    print("\n" + "=" * 80)
    print("Q1: Which are five clearly positive reviews?")
    print("=" * 80)
    
    collector.start_measurement("Q1_positive_reviews")
    
    df_query = (
        reviews.filter("{reviewtext} is clearly positive")
        .project("Keep {reviewid}")
        .limit("Keep 5 rows")
    )
    
    try:
        result = df_query.execute(
            "positive_reviews",
            PrecisionGuarantee(0.8),
            RecallGuarantee(0.8),
        )
        
        metrics = collector.end_measurement(result_count=len(result))
        result_logger.log_measurement(metrics)
        
        print("\n📊 Measurement Results:")
        print(metrics.summary())
        print("\n📋 Query Results:")
        result.pprint()
        
    except Exception as e:
        logger.error(f"Q1 failed: {e}")
        metrics = collector.end_measurement(result_count=0, errors=[str(e)])
        result_logger.log_measurement(metrics)

    # ========================================================================
    # Q2: Find 5 positive reviews for 'taken_3'
    # ========================================================================
    print("\n" + "=" * 80)
    print("Q2: Which are five clearly positive reviews for the movie 'taken_3'?")
    print("=" * 80)
    
    collector.start_measurement("Q2_positive_taken3")
    
    df_query = (
        reviews.filter("{id} is taken_3")
        .filter("{reviewtext} is clearly positive")
        .project("Keep {reviewid}")
        .limit("Keep 5 rows")
    )
    
    try:
        result = df_query.execute(
            "positive_reviews_for_taken_3",
            PrecisionGuarantee(0.8),
            RecallGuarantee(0.8),
        )
        
        metrics = collector.end_measurement(result_count=len(result))
        result_logger.log_measurement(metrics)
        
        print("\n📊 Measurement Results:")
        print(metrics.summary())
        print("\n📋 Query Results:")
        result.pprint()
        
    except Exception as e:
        logger.error(f"Q2 failed: {e}")
        metrics = collector.end_measurement(result_count=0, errors=[str(e)])
        result_logger.log_measurement(metrics)

    # ========================================================================
    # Q3: Count positive reviews for 'taken_3'
    # ========================================================================
    print("\n" + "=" * 80)
    print("Q3: How many are the positive reviews for the movie 'taken_3'?")
    print("=" * 80)
    
    collector.start_measurement("Q3_count_positive_taken3")
    
    df_query = (
        reviews.filter("{id} is taken_3")
        .filter("{reviewtext} is clearly positive")
        .aggregate("Count reviews as [positive_reviews_for_taken_3_count]")
    )
    
    try:
        result = df_query.execute(
            "positive_reviews_for_taken_3_count",
            PrecisionGuarantee(0.8),
            RecallGuarantee(0.8),
        )
        
        metrics = collector.end_measurement(result_count=len(result))
        result_logger.log_measurement(metrics)
        
        print("\n📊 Measurement Results:")
        print(metrics.summary())
        print("\n📋 Query Results:")
        result.pprint()
        
    except Exception as e:
        logger.error(f"Q3 failed: {e}")
        metrics = collector.end_measurement(result_count=0, errors=[str(e)])
        result_logger.log_measurement(metrics)

# ============================================================================
# Save batch summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

result_logger.log_batch(collector.measurements, "movie_baseline")
summary_stats = collector.get_summary_stats()
result_logger.log_summary(summary_stats, "movie_baseline_summary.json")

print("\n📊 Overall Statistics:")
for key, value in summary_stats.items():
    print(f"  {key}: {value}")

print("\n✅ Measurement run complete!")
print("=" * 80)
```

---

## 📊 Phase 4: Execute Baseline Measurements

### Task 4.1: Prepare Environment

```bash
cd project/greenQueryLab
source ../reasondb/reasondb-env/bin/activate

# Verify environment
python --version  # Should be 3.10.x
python -c "import reasondb; print('ReasonDB OK')"
python -c "import ecologits; print('EcoLogits OK')"
echo $OPENAI_API_KEY  # Should show your API key
```

### Task 4.2: Clear Cache for Clean Run

```bash
# Clear DuckDB cache
rm -rf ~/.cache/reasondb/movie_sampled

# Or if cache is elsewhere, find and clear it
find ~ -name "movie_sampled" -type d 2>/dev/null
```

### Task 4.3: Run First Measurement

```bash
python demos/movie_measured.py
```

**Expected Output:**
- Console output showing query execution
- Measurement results for each query
- JSON files in `measurement_results/` directory
- CSV summary file
- Summary statistics

### Task 4.4: Run Multiple Iterations

```bash
# Run 3-5 times to get average values
for i in {1..5}; do
    echo "=== Run $i ==="
    python demos/movie_measured.py
    sleep 5  # Brief pause between runs
done
```

### Task 4.5: Validate Results

Check that each measurement includes:
- ✅ Runtime (seconds)
- ✅ API call count
- ✅ Token usage (prompt + completion)
- ✅ Energy consumption (kWh)
- ✅ Carbon emissions (kgCO2eq)
- ✅ Result count
- ✅ No errors

---

## 📝 Phase 5: Documentation and Analysis

### Task 5.1: Analyze Results

```bash
cd measurement_results

# Count measurement files
ls -l *.json | wc -l

# View a sample measurement
cat movie_demo_Q1_positive_reviews_*.json | head -1 | python -m json.tool

# View summary
cat movie_baseline_summary.json | python -m json.tool
```

### Task 5.2: Create Measurement Report

**File:** `docs/measurement_results_movie.md`

```markdown
# Movie Demo Baseline Measurements

## Setup Information

- **Date:** [YYYY-MM-DD]
- **Branch:** docs/reproducibility-evaluation-baseline
- **Commit:** [commit hash]
- **Python Version:** 3.10.x
- **Model:** GPT-4o-mini
- **Device:** CPU
- **DuckDB Version:** 1.2.0
- **Number of Runs:** 5

## Results Summary

### Query 1: Positive Reviews

| Metric | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Average |
|--------|-------|-------|-------|-------|-------|---------|
| Runtime (s) | X.XX | X.XX | X.XX | X.XX | X.XX | X.XX |
| API Calls | X | X | X | X | X | X.X |
| Total Tokens | XXX | XXX | XXX | XXX | XXX | XXX |
| Energy (kWh) | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| CO2 (kgCO2eq) | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX | X.XXXX |
| Results | X | X | X | X | X | X |

### Query 2: Positive Reviews for Taken_3

[Same table structure]

### Query 3: Count Positive Reviews

[Same table structure]

## Overall Statistics

- **Total Runtime:** X.XX seconds (average across all queries)
- **Total API Calls:** XX
- **Total Tokens:** XXXX
- **Total Energy:** X.XXXX kWh
- **Total Carbon:** X.XXXX kgCO2eq

## Observations

### Performance
- [Note any patterns in runtime]
- [Note any variations between runs]

### Energy Consumption
- [Note energy usage patterns]
- [Compare across queries]

### Issues Encountered
- [List any errors or warnings]
- [Note any cache-related issues]

### Reproducibility
- [Note consistency across runs]
- [Note any environmental factors]

## Recommendations

- [Suggestions for optimization]
- [Next steps for measurement]
```

### Task 5.3: Update Evaluation Protocol

Add to `docs/evaluation_protocol.md`:

```markdown
## Measurement Implementation

### Data Schema

Each measurement record contains:
- `demo_name`: Name of the demo (e.g., "movie_demo")
- `query_name`: Identifier for the query (e.g., "Q1_positive_reviews")
- `start_time`: Unix timestamp when measurement started
- `end_time`: Unix timestamp when measurement ended
- `total_runtime`: Duration in seconds
- `api_calls`: Number of OpenAI API calls made
- `total_tokens`: Total tokens used (prompt + completion)
- `prompt_tokens`: Tokens in prompts
- `completion_tokens`: Tokens in completions
- `total_energy_kwh`: Energy consumption in kilowatt-hours
- `total_gwp_kgco2eq`: Carbon emissions in kg CO2 equivalent
- `model_name`: Model used (e.g., "gpt-4o-mini")
- `result_count`: Number of rows in result
- `errors`: List of any errors encountered

### Measurement Procedure

1. **Environment Setup**
   - Activate Python 3.10 environment
   - Verify EcoLogits installation
   - Confirm OpenAI API key is set

2. **Cache Management**
   - Clear DuckDB cache before first run
   - Use warm cache for subsequent runs
   - Document cache state in results

3. **Execution**
   - Run demo 3-5 times
   - Record all metrics automatically
   - Log results to JSON and CSV

4. **Validation**
   - Check for errors in logs
   - Verify all metrics are captured
   - Compare results across runs

5. **Analysis**
   - Calculate averages and standard deviations
   - Identify outliers
   - Document observations
```

---

## 🔄 Phase 6: Extend to Artwork Demo

### Task 6.1: Create Instrumented Artwork Demo

**File:** `demos/artwork_measured.py`

Follow the same pattern as `movie_measured.py`:
- Use MetricsCollector
- Track each query separately
- Log results to JSON/CSV
- Print summary statistics

### Task 6.2: Handle Image Download

```python
# First run: download images (don't measure this)
print("Downloading images (not measured)...")
# ... image download code ...

# Second run: measure with warm cache
print("Running measurement with warm cache...")
collector.start_measurement("Q1_artwork_query")
# ... query execution ...
```

### Task 6.3: Document Artwork-Specific Considerations

- Image download time vs. query time
- Cache size for images
- Network dependency
- Multimodal model usage

---

## 🤝 Phase 7: Team Synchronization

### Task 7.1: Commit Changes to Current Branch

```bash
cd project/greenQueryLab
git status
git add reasondb/evaluation/measurement/
git add demos/movie_measured.py
git add docs/EXECUTION_PLAN.md
git add docs/measurement_results_movie.md
git commit -m "feat: add energy measurement infrastructure with EcoLogits

- Create measurement module with energy tracking
- Implement metrics collector and result logger
- Add instrumented movie demo
- Document measurement methodology and results"
```

### Task 7.2: Merge to Team Branch

```bash
git checkout demo-fixes-windows-obaid
git merge docs/reproducibility-evaluation-baseline
git push origin demo-fixes-windows-obaid
```

### Task 7.3: Share Results with Team

**WhatsApp Message Template:**
```
🎉 Energy Measurement Infrastructure Complete!

✅ Implemented EcoLogits integration
✅ Movie demo baseline measurements done (5 runs)
✅ Results: [X.XX]s runtime, [XXX] tokens, [X.XXXX] kWh

📊 Key findings:
- [Finding 1]
- [Finding 2]

📁 Branch: demo-fixes-windows-obaid
📄 Docs: docs/EXECUTION_PLAN.md, docs/measurement_results_movie.md

Next: Extend to Artwork demo
```

---

## ⚠️ Important Notes

### Environment
1. **Always use Python 3.10 environment** from `reasondb-env/`
2. **Verify EcoLogits is installed** before running measurements
3. **Check OpenAI API key** is set in environment

### Cache Management
1. **Clear cache between runs** for consistent measurements
2. **Document cache state** (cold vs. warm) in results
3. **Note cache location** for reproducibility

### Compatibility
1. **Use `device="cpu"`** for Mac compatibility
2. **Use `python -m pip`** not just `pip` for installations
3. **Convert Path to string** for DuckDB: `str(path)`

### Measurement Best Practices
1. **Run 3-5 iterations** to get average values
2. **Document any errors** in measurement logs
3. **Record environment details** (OS, Python version, etc.)
4. **Save raw data** (JSON) and summaries (CSV)

---

## 🎯 Success Criteria

- [ ] Measurement infrastructure created in `reasondb/evaluation/measurement/`
- [ ] EcoLogits successfully installed and verified
- [ ] Energy tracker implemented and tested
- [ ] Metrics collector capturing all required data
- [ ] Result logger saving JSON and CSV files
- [ ] Movie demo instrumented with measurements
- [ ] 3-5 measurement runs completed successfully
- [ ] Results validated (no missing metrics)
- [ ] Measurement report documented
- [ ] Framework ready for Artwork demo extension
- [ ] Changes committed to git
- [ ] Team synchronized on findings

---

## 📞 Next Steps After Completion

### Immediate (This Week)
1. ✅ Complete Movie demo measurements
2. ⏭️ Extend to Artwork demo
3. ⏭️ Compare Movie vs. Artwork energy profiles

### Short-term (Next 2 Weeks)
1. Discuss results with Matthias and team
2. Identify optimization opportunities
3. Document energy-accuracy tradeoffs

### Medium-term (Next Month)
1. Consider DGX cluster for larger-scale measurements
2. Explore CodeCarbon for local model comparison
3. Investigate KV cache compression impact on energy

### Long-term (Project Goals)
1. Establish comprehensive energy benchmarks
2. Publish reproducible evaluation methodology
3. Contribute findings back to Stretto project

---

## 🆘 Troubleshooting

### EcoLogits Installation Issues
```bash
# If installation fails, try:
python -m pip install --upgrade pip
python -m pip install ecologits --no-cache-dir
```

### Import Errors
```bash
# Verify Python path
python -c "import sys; print(sys.path)"

# Verify package installation
python -m pip list | grep ecologits
python -m pip list | grep reasondb
```

### API Key Issues
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Set temporarily
export OPENAI_API_KEY="your-key-here"

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your-key-here"' >> ~/.bashrc
```

### Cache Issues
```bash
# Find cache location
python -c "from pathlib import Path; print(Path.home() / '.cache' / 'reasondb')"

# Clear all ReasonDB caches
rm -rf ~/.cache/reasondb/*

# Clear specific demo cache
rm -rf ~/.cache/reasondb/movie_sampled
```

---

## 📚 References

- **EcoLogits Documentation:** https://ecologits.ai/
- **Stretto Paper:** https://arxiv.org/pdf/2602.04430
- **ReasonDB Evaluation Protocol:** `docs/evaluation_protocol.md`
- **Demo Status Matrix:** `docs/demo_status_matrix.md`
- **Setup Issues and Fixes:** `docs/setup_issues_and_fixes.md`

---

**Last Updated:** 2026-05-23  
**Status:** Ready for execution - Start with Phase 1, Task 1.1