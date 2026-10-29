"""Workloads and queries that can be started from the dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Workload:
    key: str
    dataset: str  # dataset label used by the dashboard
    queries: List[str]
    description: str
    # Runner flag value per query when single-query runs are supported.
    query_flags: Dict[str, str] = field(default_factory=dict)
    # One-off preparation script that must have been run on the server.
    preparation: Optional[str] = None

    def to_public_dict(self) -> dict:
        return {
            "key": self.key,
            "dataset": self.dataset,
            "queries": self.queries,
            "description": self.description,
            "single_query_runs": bool(self.query_flags),
            "preparation": self.preparation,
        }


WORKLOADS: Dict[str, Workload] = {
    "movie": Workload(
        key="movie",
        dataset="Movie",
        queries=[
            "Q1_positive_reviews",
            "Q2_positive_reviews_for_taken_3",
            "Q3_reviews_same_sentiment",
            "Q4_positive_reviews_for_taken_3_ratio",
        ],
        description="Semantic filters, joins and ratios over SemBench movie reviews.",
        query_flags={
            "Q1_positive_reviews": "Q1",
            "Q2_positive_reviews_for_taken_3": "Q2",
            "Q3_reviews_same_sentiment": "Q3",
            "Q4_positive_reviews_for_taken_3_ratio": "Q4",
        },
    ),
    "medical": Workload(
        key="medical",
        dataset="Medical",
        queries=[
            "Q1_allergy_patients",
            "Q3_cancer_family_lung_problems_patients",
            "Q4_average_age_acne_patients",
            "Q9_sick_patients_skin_xray",
        ],
        description="Text and X-ray image queries over synthetic patient records.",
        preparation="evaluation/scripts/prepare_medical_dataset.py",
    ),
    "rotowire": Workload(
        key="rotowire",
        dataset="RotoWire",
        queries=["R1_player_points_assists", "R2_player_points_home_team_won"],
        description="Extraction joins over NBA game reports.",
    ),
}
