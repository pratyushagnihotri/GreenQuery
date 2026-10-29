"""Run two ground-truth RotoWire player benchmark queries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evaluation.measurement.config import MeasurementConfig
from evaluation.measurement.wrapper import MeasurementWrapper
from evaluation.scripts.run_movie_measurement import load_env_file
from reasondb.backends.text_qa import direct_qa_batch_size
from reasondb.database.indentifier import InPlaceColumn
from reasondb.interface.config import Config as ReasonDBConfig
from reasondb.interface.connect import RaccoonDB
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.reasoning.providers import model_requires_ecologits, provider_for, require_model_credentials
from reasondb.utils.cache import CACHE_DIR


ROOT = Path(__file__).parents[2]
FILES = ROOT / "reasondb/evaluation/benchmarks/files"
GROUND_TRUTH = ROOT / "reasondb/evaluation/ground_truth/rotowire/rotowire_players_ground_truth.csv"
PLAYERS = FILES / "players.csv"


def _count_rows(path: Path) -> int:
    with path.open(newline="") as source:
        return sum(1 for _ in csv.DictReader(source))


def _database_name() -> str:
    name = os.environ.get("GREENQUERY_RUN_NAMESPACE", "rotowire_matched_demo")
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("GREENQUERY_RUN_NAMESPACE must use only letters, numbers, and underscores")
    return name


def _clear_cache(database_name: str) -> None:
    path = (CACHE_DIR.resolve() / database_name).resolve()
    if path.parent != CACHE_DIR.resolve():
        raise RuntimeError(f"Unexpected cache path: {path}")
    if path.exists():
        shutil.rmtree(path)


def _quality(frame, fields: tuple[tuple[str, str], ...]) -> dict:
    with PLAYERS.open(newline="") as source:
        player_names = [row["name"] for row in csv.DictReader(source)]
    with GROUND_TRUTH.open(newline="") as source:
        labels = {
            (str(row["_index_reports"]), player_names[int(row["_index_players"])]): row
            for row in csv.DictReader(source)
        }

    def game_id(value) -> str:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)

    checked = correct = 0
    for _, row in frame.iterrows():
        key = (game_id(row.get("game_id")), str(row.get("name")))
        expected = labels.get(key)
        if not expected:
            continue
        for column, label in fields:
            if expected[label] and column in row:
                checked += 1
                try:
                    correct += float(row[column]) == float(expected[label])
                except (TypeError, ValueError):
                    pass
    return {
        "status": "evaluated" if checked else "not_evaluated",
        "metric": "exact_numeric_field_accuracy",
        "checked_fields": checked,
        "correct_fields": correct,
        "accuracy": correct / checked if checked else None,
        "ground_truth": str(GROUND_TRUTH.relative_to(ROOT)),
        **({"reason": "no result rows matched RotoWire ground truth"} if not checked else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend-profile",
        choices=["api", "dgx-local-text"],
        default="dgx-local-text",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--energy-measurement", choices=["none", "nvml"], default="nvml")
    parser.add_argument("--cold-cache", action="store_true")
    parser.add_argument("--matched-planner", action="store_true")
    parser.add_argument(
        "--online-cache-policy", choices=["default", "disabled"], default="default"
    )
    parser.add_argument("--split-artifacts", action="store_true")
    args = parser.parse_args()
    if args.backend_profile != "api" and not args.matched_planner:
        raise ValueError("RotoWire matched demo requires --matched-planner")
    if args.backend_profile == "api":
        load_env_file()
        require_model_credentials(args.model_name)
        os.environ["GREENQUERY_OPENAI_MODEL"] = args.model_name
    database_name = _database_name()
    if args.cold_cache:
        _clear_cache(database_name)

    reports = FILES / "reports.csv"
    is_api = args.backend_profile == "api"
    config = MeasurementConfig(
        demo_name="rotowire",
        device="cpu" if is_api else args.device,
        output_dir=args.output_dir,
        model_name=args.model_name,
        planner_model_name=args.model_name,
        model_provider=provider_for(args.model_name) if is_api else "local",
        model_type="api" if is_api else "dgx",
        backend_profile=args.backend_profile,
        energy_measurement="none" if is_api else args.energy_measurement,
        co2_g_per_kwh=None if is_api else float(os.environ["GREENQUERY_CO2_G_PER_KWH"]),
        hardware_provider=None if is_api else "dgx",
        gpu_count=None if is_api else 1,
        direct_qa_batch_size=None if is_api else direct_qa_batch_size(),
        reasondb_database_name=database_name,
        dataset_name="rotowire_players_reports",
        dataset_source="ReasonDB RotoWire benchmark files",
        dataset_rows=_count_rows(reports),
        dataset_sha256=hashlib.sha256(reports.read_bytes()).hexdigest(),
        cache_state="disabled" if args.cold_cache else "not_controlled",
        model_cache_state="disabled" if args.cold_cache else "not_controlled",
        local_cache_policy="disabled" if args.cold_cache else "default",
        planner_cache_state="disabled" if args.cold_cache else "not_controlled",
        online_cache_policy=args.online_cache_policy,
        query_set="ground_truth_demo",
        operator_policy="semantic-text-only",
        baseline_type="rotowire_ground_truth",
    )
    config_dict = config.to_dict()
    if is_api:
        config_dict.update(
            require_ecologits=model_requires_ecologits(args.model_name),
            online_measurement_boundary="provider_estimated_from_ecologits"
            if model_requires_ecologits(args.model_name)
            else "provider_usage_only_no_energy_estimate",
        )
    with MeasurementWrapper(
        "rotowire", args.output_dir, config_dict, split_artifacts=args.split_artifacts
    ) as wrapper:
        reason_config = ReasonDBConfig(
            database_name,
            device=config.device,
            operator_policy=config.operator_policy,
            backend_profile=config.backend_profile,
            local_cache_policy=config.local_cache_policy,
            openai_model_id=config.model_name if is_api else None,
            openai_cache_enabled=config.online_cache_policy != "disabled",
            local_text_model_id=None if is_api else config.model_name,
            local_planner_model_id=None if is_api else config.model_name,
        )
        with RaccoonDB(database_name, config=reason_config, device=config.device) as rc:
            players = rc.add_table(FILES / "players.csv", "players")
            reports_table = rc.add_table(reports, "reports", text_columns=[InPlaceColumn("reports.report")])
            player_games = rc.add_table(FILES / "players_to_games.csv", "players_to_games")
            points = (
                players.join(player_games, "Join on {name}")
                .join(reports_table, "Join on {game_id}")
                .extract("How many [points] did {name} score according to {report}?")
            )
            points_assists = (
                points
                .extract("How many [assists] did {name} have according to {report}?")
                .project("Keep {game_id}, {name}, {points}, and {assists}")
            )
            with wrapper.measure_query("R1_player_points_assists", "join_extract_extract_project") as tracker:
                result = points_assists.execute("rotowire_points_assists", RecallGuarantee(0.8), PrecisionGuarantee(0.8))
                tracker.set_result(result)
                tracker.set_result_summary_fields(
                    quality=_quality(result.to_df(), (("points", "Points"), ("assists", "Assists")))
                )

            home_team_won = (
                points
                .filter("Only keep if the home team won the game in {report}")
                .project("Keep {game_id}, {name}, and {points}")
            )
            with wrapper.measure_query("R2_player_points_home_team_won", "join_extract_filter_project") as tracker:
                result = home_team_won.execute("rotowire_points_home_team_won", RecallGuarantee(0.8), PrecisionGuarantee(0.8))
                tracker.set_result(result)
                tracker.set_result_summary_fields(
                    quality={
                        **_quality(result.to_df(), (("points", "Points"),)),
                        "filter_ground_truth": "not_available: home-team winner labels are not in the shipped RotoWire ground truth",
                    }
                )


if __name__ == "__main__":
    main()
