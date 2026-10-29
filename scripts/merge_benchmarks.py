import shutil
import json
from pathlib import Path

results_dirs = [
    "benchmark_results_t05",
    "benchmark_results_t07",
    "benchmark_results_t09",
]

target_dir = "benchmark_results"

for r in results_dirs:
    shutil.copytree(r, target_dir, dirs_exist_ok=True)


for dataset_dir in Path(target_dir).iterdir():
    if not dataset_dir.is_dir():
        continue
    for subset_dir in dataset_dir.iterdir():
        cache_dir = subset_dir / "cache"
        for json_file in cache_dir.glob("*.json"):
            full_data = {}
            for r in results_dirs:
                orig_json = (
                    Path(r)
                    / dataset_dir.name
                    / subset_dir.name
                    / "cache"
                    / json_file.name
                )
                with open(orig_json) as f:
                    data = json.load(f)
                    updated_data = {}
                    for k, v in data.items():
                        v["data_filepath"] = v["data_filepath"].replace(r, target_dir)
                        updated_data[k] = v
                    full_data.update(updated_data)
                with json_file.open("w") as f:
                    json.dump(full_data, f, indent=4)
