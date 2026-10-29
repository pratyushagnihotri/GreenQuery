import os
import logging
import re
from pathlib import Path
from typing import Dict, Literal, Sequence, Union
from reasondb.database.database import ExperimentalDatabase
from reasondb.database.indentifier import (
    RemoteColumn,
    VirtualTableIdentifier,
)
from reasondb.evaluation.benchmark import Benchmark, RandomBenchmark, LabelsDefinition
from reasondb.query_plan.logical_plan import (
    LogicalFilter,
    LogicalJoin,
    LogicalPlan,
    LogicalExtract,
    LogicalRename,
)
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    Queries,
    Query,
    QueryShape,
    RandomOrder,
)

logger = logging.getLogger(__name__)


ARTWORK_QUERIES = Queries(
    # Query(
    #    "What are the paintings that depict Madonna and Child and in which century have they been created?",
    #    _ground_truth_logical_plan=LogicalPlan(
    #        [
    #            LogicalFilter(
    #                explanation="First, we need to filter the paintings that depict Madonna and Child.",
    #                inputs=[VirtualTableIdentifier("artworks")],
    #                output=VirtualTableIdentifier("madonna_and_child"),
    #                expression="{:0:.image} depicts Madonna and Child",
    #                labels=LabelsDefinition(
    #                    Path(
    #                        "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
    #                    ),
    #                    "m&c",
    #                    ["artworks"],
    #                ),
    #            ),
    #            LogicalExtract(
    #                explanation="We also need to extract the century from the inception date.",
    #                inputs=[VirtualTableIdentifier("madonna_and_child")],
    #                output=VirtualTableIdentifier("madonna_and_child_with_century"),
    #                expression="Extract the [century] from {madonna_and_child.inception}",
    #                labels=LabelsDefinition(
    #                    Path(
    #                        "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
    #                    ),
    #                    "century",
    #                    ["artworks"],
    #                ),
    #            ),
    #        ]
    #    ),
    # ),
    Query(
        "What are the paintings that depict Madonna and child?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the paintings that depict Madonna and Child.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("madonna_and_child"),
                    expression="{:0:.image} depicts Madonna and Child",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
                        ),
                        "m&c",
                        ["artworks"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which paintings depict more than two people?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the paintings that depict more than two people.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("more_than_two_people"),
                    expression="{:0:.image} depicts more than two people",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
                        ),
                        "more_than_2_people",
                        ["artworks"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which paintings depict more than three people?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the paintings that depict more than three people.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("more_than_three_people"),
                    expression="{:0:.image} depicts more than three people",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
                        ),
                        "more_than_3_people",
                        ["artworks"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which paintings depict saints identifiable by their halos?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the paintings that depict saints identifiable by their halos.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("saints_with_halos"),
                    expression="{:0:.image} depicts saints identifiable by their halos",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
                        ),
                        "saints_with_halos",
                        ["artworks"],
                    ),
                ),
            ]
        ),
    ),
    Query(
        "Which paintings depict a scene in which death is a dominant theme?",
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalFilter(
                    explanation="We need to filter the paintings that depict a scene in which death is a dominant theme.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("death_dominant_theme"),
                    expression="{:0:.image} depicts a scene in which death is a dominant theme",
                    labels=LabelsDefinition(
                        Path(
                            "reasondb/evaluation/ground_truth/artwork/artwork_no_duplicated.csv"
                        ),
                        "death_theme",
                        ["artworks"],
                    ),
                ),
            ]
        ),
    ),
)


class Artwork(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "artwork_no_duplicated"

    @property
    def has_ground_truth(self) -> bool:
        return True

    @staticmethod
    def get_csv():
        orig_csv = (
            Path(__file__).parent / "files" / "paintings_sampled_no_duplicated.csv"
        )
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(Artwork.dir(), exist_ok=True)
        return Artwork.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(Artwork.dir(), exist_ok=True)
        benchmark = Artwork(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=Artwork.name(),
                split=split,
                table_names=["artworks"],
                paths=[Artwork.get_csv()],
                image_columns=[
                    RemoteColumn("artworks.image_url", "artworks.image", url=True)
                ],
            ),
            ARTWORK_QUERIES,
        )
        return benchmark


class ArtworkLarge(Benchmark):
    @classmethod
    def name(cls) -> str:
        return "artwork_large"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "paintings_large.csv"
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(ArtworkLarge.dir(), exist_ok=True)
        return ArtworkLarge.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(ArtworkLarge.dir(), exist_ok=True)
        benchmark = ArtworkLarge(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=ArtworkLarge.name(),
                split=split,
                table_names=["artworks"],
                paths=[ArtworkLarge.get_csv()],
                image_columns=[
                    RemoteColumn("artworks.image_url", "artworks.image", url=True)
                ],
            ),
            ARTWORK_QUERIES,
        )
        return benchmark


class ArtworkRandom(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "artwork_random"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = (
            Path(__file__).parent / "files" / "paintings_sampled_no_duplicated.csv"
        )
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(ArtworkRandom.dir(), exist_ok=True)
        return ArtworkRandom.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(ArtworkRandom.dir(), exist_ok=True)
        benchmark = ArtworkRandom(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=ArtworkRandom.name(),
                split=split,
                table_names=["artworks"],
                paths=[ArtworkRandom.get_csv()],
                image_columns=[
                    RemoteColumn("artworks.image_url", "artworks.image", url=True)
                ],
            ),
            ArtworkRandom.generate_random_queries(split),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return ARTWORK_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return ARTWORK_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE

    @classmethod
    def get_join_queries(cls) -> "Queries":
        return ARTWORK_JOIN_QUERIES


class ArtworkRandomMedium(RandomBenchmark):
    @classmethod
    def name(cls) -> str:
        return "artwork_random_medium"

    @property
    def has_ground_truth(self) -> bool:
        return False

    @staticmethod
    def get_csv():
        orig_csv = Path(__file__).parent / "files" / "paintings_medium.csv"
        return orig_csv

    @staticmethod
    def urls():
        return {}

    @staticmethod
    def download(split: Literal["train", "dev", "test"]) -> Benchmark:
        assert split == "dev"
        os.makedirs(ArtworkRandomMedium.dir(), exist_ok=True)
        return ArtworkRandomMedium.load_from_disk(split)

    @staticmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        assert split == "dev"
        os.makedirs(ArtworkRandomMedium.dir(), exist_ok=True)
        benchmark = ArtworkRandomMedium(
            split,
            ExperimentalDatabase.load_from_files(
                db_name=ArtworkRandomMedium.name(),
                split=split,
                table_names=["artworks"],
                paths=[ArtworkRandomMedium.get_csv()],
                image_columns=[
                    RemoteColumn("artworks.image_url", "artworks.image", url=True)
                ],
            ),
            ArtworkRandomMedium.generate_random_queries(split),
        )
        return benchmark

    @classmethod
    def _get_query_shapes(cls) -> Sequence[QueryShape]:
        return ARTWORK_QUERY_SHAPES

    @classmethod
    def _get_operator_options(cls) -> Sequence[OperatorOption]:
        return ARTWORK_OPERATOR_OPTIONS

    @classmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        return SINGLE_FILTER_SHAPE

    @classmethod
    def get_join_queries(cls) -> "Queries":
        return ARTWORK_JOIN_QUERIES


SINGLE_FILTER_SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("artworks")],
        output=VirtualTableIdentifier("output"),
    ),
)


painting_genres = [
    "History Painting",
    "Portraiture",
    "Landscape",
    "Still Life",
    "Genre Painting",
    "Animal Painting",
    "Marine Art",
    "Abstract Art",
    "Impressionism",
    "Expressionism",
    "Surrealism",
    "Cubism",
    "Pop Art",
    "Photorealism",
    "Street Art",
]

colors = [
    "Red",
    "Blue",
    "Yellow",
    "Green",
    "Orange",
    "Purple",
    "Black",
    "White",
]


ARTWORK_OPERATOR_OPTIONS = [
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts Madonna and Child",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts more than two people",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts more than three people",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts saints identifiable by their halos",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a scene in which death is a dominant theme",
    ),
    OperatorOption(LogicalFilter, "{:0:.image} depicts a religous scene"),
    OperatorOption(LogicalFilter, "{:0:.image} shows a still life"),
    OperatorOption(LogicalFilter, "{:0:.image} a scene of war"),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts an angel with wings",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a crucifixion scene",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows a single seated figure",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows a figure holding a book or scroll",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} includes an animal as a central element",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a landscape with visible mountains",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts an interior scene with architectural elements",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a figure wearing armor",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a scene involving water or the sea",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows a figure playing a musical instrument",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} includes symbolic objects such as skulls, hourglasses, or candles",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a mythological figure identifiable by attributes",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a royal or noble figure wearing a crown",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a battle or combat scene",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows a domestic scene with everyday activities",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a figure in prayer",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows architectural ruins",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a nighttime scene",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} shows a figure with a visible halo or radiance",
    ),
    OperatorOption(
        LogicalFilter,
        "{:0:.image} depicts a narrative scene from classical mythology",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [century] from {artworks.inception}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [year] from {artworks.inception}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of people [num_people] depicted in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        f"What is the genre [estimated_genre] of each artwork in {{:0:.image}}? Choose from {', '.join(painting_genres)}.",
    ),
    OperatorOption(
        LogicalExtract,
        f"Extract the primary background color [background] of each artwork in {{:0:.image}}. Choose from {', '.join(colors)}.",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of saints with halos [saints_with_halos] from {:0:.image}.",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of animals [num_animals] from {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of swords [num_swords] from {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the [gender] of the main character (male / female / undefined) from {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of angels [num_angels] depicted in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the dominant emotion [dominant_emotion] expressed by the central figure in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of visible halos [num_halos] in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the estimated historical period [period] depicted in {:0:.image} (e.g., Antiquity, Middle Ages, Renaissance, Baroque, Modern)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of musical instruments [num_instruments] shown in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of architectural elements [num_architectural_elements] visible in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of identifiable mythological beings [num_mythological_beings] in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the type of setting [setting_type] of {:0:.image} (interior / exterior / undefined)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the dominant material [dominant_material] depicted (stone / wood / metal / fabric / undefined) from {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of weapons [num_weapons] visible in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of written texts or inscriptions [num_texts] visible in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of human faces [num_faces] visible in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the approximate lighting type [lighting] in {:0:.image} (natural / candle / undefined)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the level of movement [movement_level] in {:0:.image} (static / moderate / dynamic)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the landscape type [landscape_type] depicted in {:0:.image} (mountain / forest / sea / plain / undefined)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of visible clouds [num_clouds] in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of distinct symbolic objects [num_symbols] in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract whether a crown is present [has_crown] in {:0:.image} (yes / no)",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the approximate age group [age_group] of the main figure (child / youth / adult / elderly) in {:0:.image}",
    ),
    OperatorOption(
        LogicalExtract,
        "Extract the number of visible candles [num_candles] in {:0:.image}",
    ),
    # 10 join for artworks (self-join: {:0:.image} = left table image, {:1:.image_other} = right table image)
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict paintings with a visible halo or radiance",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict a religious scene",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both feature a portrait of a single person",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict a landscape with no human figures",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict a crucifixion scene",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict a scene set indoors",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both show more than two human figures",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict animals",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both use a dark or somber color scheme",
    ),
    OperatorOption(
        LogicalJoin,
        "{:0:.image} and {:1:.image_other} both depict a battle or combat scene",
    ),
]

ARTWORK_QUERY_SHAPES = [
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 2,
    #        "num_sem_filter": 1,
    #        "num_sem_extract": 1,
    #    },
    #),
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("intermediate")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 2,
    #        "num_sem_filter": 2,
    #        "num_sem_extract": 0,
    #    },
    #),
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate1"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate1")],
    #            output=VirtualTableIdentifier("intermediate2"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate2")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 3,
    #        "num_sem_filter": 1,
    #        "num_sem_extract": 2,
    #    },
    #),
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate1"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("intermediate1")],
    #            output=VirtualTableIdentifier("intermediate2"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate2")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 3,
    #        "num_sem_filter": 2,
    #        "num_sem_extract": 1,
    #    },
    #),
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate1"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate1")],
    #            output=VirtualTableIdentifier("intermediate2"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate2")],
    #            output=VirtualTableIdentifier("intermediate3"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate3")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 4,
    #        "num_sem_filter": 1,
    #        "num_sem_extract": 3,
    #    },
    #),
    #QueryShape(
    #    RandomOrder(
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("artworks")],
    #            output=VirtualTableIdentifier("intermediate1"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalFilter,
    #            inputs=[VirtualTableIdentifier("intermediate1")],
    #            output=VirtualTableIdentifier("intermediate2"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate2")],
    #            output=VirtualTableIdentifier("intermediate3"),
    #        ),
    #        OperatorPlaceholder(
    #            LogicalExtract,
    #            inputs=[VirtualTableIdentifier("intermediate3")],
    #            output=VirtualTableIdentifier("output"),
    #        ),
    #    ),
    #    additional_info={
    #        "num_semops": 4,
    #        "num_sem_filter": 2,
    #        "num_sem_extract": 2,
    #    },
    #),
        QueryShape(
        LogicalRename(
            explanation="Rename artworks table for self-join.",
            inputs=[VirtualTableIdentifier("artworks")],
            output=VirtualTableIdentifier("artworks_other"),
            expression="Rename {artworks.image} to [image_other]",
            labels=None,
        ),
        OperatorPlaceholder(
            LogicalJoin,
            inputs=[
                VirtualTableIdentifier("artworks"),
                VirtualTableIdentifier("artworks_other"),
            ],
            output=VirtualTableIdentifier("output"),
        ),
        additional_info={
            "num_semops": 1,
            "num_sem_filter": 0,
            "num_sem_join": 1,
            "num_sem_extract": 0,
        },
    ),
    QueryShape(
        LogicalRename(
            explanation="Rename artworks table for self-join.",
            inputs=[VirtualTableIdentifier("artworks")],
            output=VirtualTableIdentifier("artworks_other"),
            expression="Rename {artworks.image} to [image_other]",
            labels=None,
        ),
        OperatorPlaceholder(
            LogicalFilter,
            inputs=[
                VirtualTableIdentifier("artworks"),
            ],
            output=VirtualTableIdentifier("filtered"),
        ),
        OperatorPlaceholder(
            LogicalJoin,
            inputs=[
                VirtualTableIdentifier("filtered"),
                VirtualTableIdentifier("artworks_other"),
            ],
            output=VirtualTableIdentifier("output"),
        ),
        additional_info={
            "num_semops": 2,
            "num_sem_filter": 1,
            "num_sem_join": 1,
            "num_sem_extract": 0,
        },
    ),
]


def _make_join_query(expression: str) -> Query:
    expression = re.sub(
        r"{\:0:\.([a-z_][a-z0-9_]*)}",
        r"artworks.\1",
        expression,
    )
    expression = re.sub(
        r"{\:1:\.([a-z_][a-z0-9_]*)}",
        r"artworks_other.\1",
        expression,
    )
    return Query(
        expression,
        _ground_truth_logical_plan=LogicalPlan(
            [
                LogicalRename(
                    explanation="Rename artworks table for self-join.",
                    inputs=[VirtualTableIdentifier("artworks")],
                    output=VirtualTableIdentifier("artworks_other"),
                    expression="Rename {artworks.image} to [image_other]",
                    labels=None,
                ),
                LogicalJoin(
                    explanation=expression,
                    inputs=[
                        VirtualTableIdentifier("artworks"),
                        VirtualTableIdentifier("artworks_other"),
                    ],
                    output=VirtualTableIdentifier("output"),
                    expression=expression,
                    labels=None,
                ),
            ]
        ),
    )


ARTWORK_JOIN_QUERIES = Queries(
    *[
        _make_join_query(opt.expression)
        for opt in ARTWORK_OPERATOR_OPTIONS
        if opt.operator_type == LogicalJoin
    ]
)
