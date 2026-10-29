from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
import argparse
from enum import IntEnum
from functools import partial
import math
from typing import List, Literal, Optional, Sequence, Tuple
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import optuna

DATABASE_SIZE = 20_000
USE_CUDA_IF_AVAILABLE = False
CONFIDENCE = 0.95
NUM_NEW_SAMPLES_MAX = NUM_NEW_SAMPLES_MIN = 30
UPPER_TARGET = 0.9
LOWER_TARGET = 0.5
MAX_NUM_ITERATIONS = 100


class OptimizationStage(IntEnum):
    CHOOSE_OPERATORS = 0
    CHOOSE_PARAMETERS = 1


class Target:
    def __init__(self, precision_target, recall_target):
        if not (0 <= precision_target <= 1) or not (0 <= recall_target <= 1):
            raise ValueError("Targets must be between 0 and 1.")
        self.precision_target = precision_target
        self.recall_target = recall_target

    def __repr__(self):
        return f"Target(precision_target={self.precision_target}, recall_target={self.recall_target})"

    def __str__(self):
        return f"Target: precision={self.precision_target}, recall={self.recall_target}"


class Prediction(IntEnum):
    KEEP = 1
    DISCARD = -1
    UNCLEAR = 0


class OperatorPruningStrategy(IntEnum):
    NONE = 0
    NEVER_PICKED = 1
    INDIVIDUAL_COMPARISON = 2


class SamplingStrategy(IntEnum):
    UNIFORM = 0
    PRODUCT_ALL = 1
    SQRT_PRODUCT_ALL = 2
    PRODUCT_ANY = 3
    SQRT_PRODUCT_ANY = 4


PRUNING_COST_THRESHOLD = 0.01


class Filter:
    def __init__(self, cost: float):
        self.cost = cost


class DecisionFilter(Filter):
    def __init__(
        self,
        fnr: float,
        fpr: float,
        unclear_rate: float,
        cost: float,
        num_values: int,
        rng: torch.Generator,
    ):
        self.rng = rng
        self.fpr = fpr
        self.fnr = fnr
        self.unclear_rate = unclear_rate
        self.num_eval_values = num_values
        self._noise_vector_unclear = torch.rand(num_values, generator=self.rng)
        self._noise_vector_fnr = torch.rand(num_values, generator=self.rng)
        self._noise_vector_fpr = torch.rand(num_values, generator=self.rng)
        self._prediction = torch.zeros(0, dtype=torch.int8)
        super().__init__(cost)

    def get_noise_vector_unclear(self, value_indexes: torch.Tensor) -> torch.Tensor:
        """Generate a noise vector based on the noise level."""
        return self._noise_vector_unclear[value_indexes]

    def get_noise_vector_fnr(self, value_indexes: torch.Tensor) -> torch.Tensor:
        """Generate a noise vector based on the noise level."""
        return self._noise_vector_fnr[value_indexes]

    def get_noise_vector_fpr(self, value_indexes: torch.Tensor) -> torch.Tensor:
        """Generate a noise vector based on the noise level."""
        return self._noise_vector_fpr[value_indexes]

    def predict(
        self, labels: torch.Tensor, value_indexes: torch.Tensor
    ) -> torch.Tensor:
        """Predict whether to keep, discard, or mark as unclear based on the value and selectivity."""
        unclear_mask = self.get_noise_vector_unclear(value_indexes) < self.unclear_rate
        fp_mask = self.get_noise_vector_fpr(value_indexes) < self.fpr
        fn_mask = self.get_noise_vector_fnr(value_indexes) < self.fnr

        positives = ((labels == Prediction.DISCARD) & fp_mask) | (
            (labels == Prediction.KEEP) & ~fn_mask
        )
        negatives = ~positives
        positives = positives & ~unclear_mask
        negatives = negatives & ~unclear_mask

        result = (
            (positives * Prediction.KEEP)
            + (negatives * Prediction.DISCARD)
            + (unclear_mask * Prediction.UNCLEAR)
        )
        self._prediction = result.to(torch.int8)
        return self._prediction


class SimilarityFilter(Filter):
    def __init__(
        self,
        cost: float,
        noise_level: float,
        num_values: int,
        rng: torch.Generator,
    ):
        self.rng = rng
        self.cost = cost
        self.noise_level = noise_level
        self.num_eval_values = num_values
        self._noise_vector = (
            2 * torch.rand(num_values, generator=self.rng) - 1
        ) * self.noise_level
        super().__init__(cost)

    def get_noise_vector(self, value_indexes) -> torch.Tensor:
        """Generate a noise vector based on the noise level."""
        return self._noise_vector[value_indexes]

    def predict_keep_score(
        self,
        values: torch.Tensor,
        params: torch.Tensor,
        value_indexes: torch.Tensor,
    ) -> torch.Tensor:
        noised_values = values + self.get_noise_vector(value_indexes).to(params.device)
        score = params[:, 0].reshape(-1, 1) - noised_values.reshape(1, -1)
        return score  # Shape: (num_jobs, num_values)

    def predict_discard_score(
        self,
        values: torch.Tensor,
        params: torch.Tensor,
        value_indexes: torch.Tensor,
    ):
        noised_values = values + self.get_noise_vector(value_indexes).to(values.device)
        score = noised_values.reshape(1, -1) - params[:, 1].reshape(-1, 1)
        return score  # Shape: (num_jobs, num_values)

    def predict_unclear_score(
        self,
        values: torch.Tensor,
        params: torch.Tensor,
        value_indexes: torch.Tensor,
    ) -> torch.Tensor:
        noised_values = values + self.get_noise_vector(value_indexes).to(values.device)

        score_smaller = noised_values.reshape(1, -1) - params[:, 0].reshape(-1, 1)
        score_larger = params[:, 1].reshape(-1, 1) - noised_values.reshape(1, -1)
        scores_stacked = torch.stack([score_smaller, score_larger], dim=1)
        scores = scores_stacked.min(dim=1)
        return scores.values


class FilterCascade:
    def __init__(self, filters: List[Filter]):
        self.filters = sorted(filters, key=lambda f: f.cost)

    def cost_per_value(
        self,
        decision_probabilities: torch.Tensor,  # Shape: (num_jobs, num_values, 3, num_filters + 1)
        pick_score: torch.Tensor,
        temperature: float = 1.0,
        remove_last_filter: bool = True,
    ) -> torch.Tensor:
        """Calculate the total cost of the filter cascade."""
        num_jobs = pick_score.shape[0]
        sigmoid = torch.nn.Sigmoid()
        input_probabs = decision_probabilities[
            :, :, 2, : decision_probabilities.shape[3] - int(remove_last_filter)
        ]  # Shape: (num_jobs, num_values, num_filters)
        cost_vector = torch.tensor(
            [f.cost for f in self.filters], device=pick_score.device
        )
        use_probabs = sigmoid(pick_score / temperature)
        individual_cost = (
            input_probabs.permute(0, 2, 1)  # Shape: (num_jobs, num_filters, num_values)
            * (use_probabs * cost_vector.reshape(1, -1)).reshape(num_jobs, -1, 1)
        ).permute(0, 2, 1)  # Shape: (num_jobs, num_values, num_filters)
        total_cost = individual_cost.sum(dim=2)  # Shape: (num_values,)
        cost_per_value = total_cost.mean(dim=1)
        return cost_per_value

    def sample_cost_per_value(self, prune_mask: np.ndarray):
        mask: List[bool] = (~prune_mask.all(axis=0)).tolist()  # type: ignore
        return sum((f.cost * p) for f, p in zip(self.filters, mask))

    def keep_proba(
        self,
        decision_probabilities: torch.Tensor,  # Shape: (num_jobs, num_values, 3, num_filters + 1)
    ) -> torch.Tensor:
        """Calculate the probability of keeping an item after the whole cascade."""
        return decision_probabilities[:, :, 0, -1]  # Shape: (num_jobs, num_values)

    def decision_probabilities(
        self,
        values: torch.Tensor,  # Shape: (num_values, )
        labels: torch.Tensor,  # Shape: (num_values, )
        value_indexes: torch.Tensor,  # Shape: (num_values, )
        pick_score: torch.Tensor,  # Shape: (num_jobs, num_filters)
        params: torch.Tensor,  # Shape: (num_jobs, 2)
        temperature_pick: float,
        temperature_params: float,
    ) -> torch.Tensor:
        """Predict the keep score for the whole pipeline"""
        assert len(values) == len(labels), "Values and labels must have the same length"
        sigmoid = torch.nn.Sigmoid()
        softmax = torch.nn.Softmax(dim=2)
        num_jobs = pick_score.shape[0]
        num_values = len(values)
        device = pick_score.device

        # keep, discard, unclear
        probabilities = [torch.zeros((num_jobs, num_values, 3), device=device)]
        # initially all are unsure
        probabilities[0][:, :, 2] = 1.0  # Shape: (num_jobs, num_values, 3)

        for i, filter in enumerate(self.filters):
            use_proba = sigmoid(pick_score[:, i] / temperature_pick)
            if isinstance(filter, DecisionFilter):
                filter_preds = filter.predict(
                    labels=labels, value_indexes=value_indexes
                )
                keep_bitvector = filter_preds == Prediction.KEEP
                discard_bitvector = filter_preds == Prediction.DISCARD
                keep_proba = torch.mul(
                    use_proba.reshape(-1, 1), keep_bitvector.float().reshape(1, -1)
                )  # Shape: (num_jobs, num_values)
                discard_proba = torch.mul(
                    use_proba.reshape(-1, 1), discard_bitvector.float().reshape(1, -1)
                )  # Shape: (num_jobs, num_values)
                unclear_proba = 1.0 - keep_proba - discard_proba
                this_probas = torch.stack(
                    [keep_proba, discard_proba, unclear_proba], dim=2
                )  # Shape: (num_jobs, num_values, 3)

            elif isinstance(filter, SimilarityFilter):
                keep_score_used = filter.predict_keep_score(
                    values=values.to(device), params=params, value_indexes=value_indexes
                )
                discard_score_used = filter.predict_discard_score(
                    values=values.to(device), params=params, value_indexes=value_indexes
                )
                unclear_score_used = filter.predict_unclear_score(
                    values=values.to(device), params=params, value_indexes=value_indexes
                )
                stacked = torch.stack(
                    [
                        keep_score_used,
                        discard_score_used,
                        unclear_score_used,
                    ],
                    dim=2,
                )  # Shape: (num_jobs, num_values, 3)
                probas_when_used = softmax(stacked / temperature_params)
                keep_scores = use_proba.reshape(-1, 1) * probas_when_used[:, :, 0]
                discard_scores = use_proba.reshape(-1, 1) * probas_when_used[:, :, 1]
                unclear_scores = 1.0 - keep_scores - discard_scores
                this_probas = torch.stack(
                    [keep_scores, discard_scores, unclear_scores], dim=2
                )
            else:
                raise ValueError(f"Unknown filter type: {type(filter)}")

            next_probas = torch.stack(
                [
                    # keep when previous was keep or previous was unclear and this is keep
                    probabilities[-1][:, :, 0]
                    + probabilities[-1][:, :, 2] * this_probas[:, :, 0],
                    # discard when previous was discard or previous was unclear and this is discard
                    probabilities[-1][:, :, 1]
                    + probabilities[-1][:, :, 2] * this_probas[:, :, 1],
                    # unclear when previous was unclear and this is unclear
                    probabilities[-1][:, :, 2] * this_probas[:, :, 2],
                ],
                dim=2,
            )  # Shape: (num_jobs, num_values, 3)
            if not torch.all(next_probas >= -0.0001):
                print(f"Probabilities must be non-negative, got {next_probas.min()}")
            next_probas = torch.max(next_probas, torch.tensor(0.0, device=device))
            sum_prob = next_probas.sum(dim=2)
            if not torch.all(torch.isclose(sum_prob, torch.tensor(1.0, device=device))):
                print(
                    f"Probabilities must sum to 1, got range ({sum_prob.min()}, {sum_prob.min()})"
                )
                next_probas = next_probas / sum_prob.unsqueeze(2)

            probabilities.append(next_probas)

        # Shape: (num_jobs, num_values, 3, num_filters + 1)
        return torch.stack(probabilities, dim=3)  # skip the initial unsure state


class Pipeline:
    """A class to represent a pipeline of filter cascades."""

    def __init__(self, filter_cascades: List[FilterCascade]):
        self.filters_cascades = filter_cascades
        self.order = self.get_initial_order()

    def get_initial_order(self) -> List[Tuple[int, int]]:
        cost_odered = sorted(
            [
                (filter.cost, (i, j))
                for i, cascade in enumerate(self.filters_cascades)
                for j, filter in enumerate(cascade.filters)
            ]
        )
        order = [index for _, index in cost_odered]
        return order

    def decision_probabilities(
        self,
        value_indexes: torch.Tensor,  # Shape: (num_values, )
        all_values: List[torch.Tensor],  # Shape: (num_cascades, num_values)
        all_labels: List[torch.Tensor],  # Shape: (num_cascades, num_values)
        all_pick_scores: List[
            torch.Tensor
        ],  # Shape: (num_jobs, num_cascades, num_filters)
        all_params: List[torch.nn.Parameter],  # Shape: (num_jobs, num_cascades, 2)
        temperature_pick: float,
        temperature_params: float,
    ) -> List[torch.Tensor]:
        """Calculate the decision probabilities for the entire pipeline."""
        assert len(self.filters_cascades) > 0, "Pipeline must have at least one cascade"
        all_probabilities = []
        for i, cascade in enumerate(self.filters_cascades):
            probabilities = cascade.decision_probabilities(
                value_indexes=value_indexes,
                values=all_values[i],
                labels=all_labels[i],
                pick_score=all_pick_scores[i],
                params=all_params[i],
                temperature_pick=temperature_pick,
                temperature_params=temperature_params,
            )  # Shape: (num_jobs, num_values, 3, num_filters + 1)
            all_probabilities.append(probabilities)
        return all_probabilities

    def keep_proba(
        self,
        all_decision_probabilities: List[
            torch.Tensor
        ],  # Shape: num_cascades x (num_jobs, num_values, 3, num_filters + 1)
    ) -> torch.Tensor:
        """Calculate the keep probabilities for the entire pipeline."""
        assert len(self.filters_cascades) > 0, "Pipeline must have at least one cascade"
        keep_probabilities = []
        for i, cascade in enumerate(self.filters_cascades):
            keep_proba = cascade.keep_proba(all_decision_probabilities[i])
            keep_probabilities.append(keep_proba)
        stacked = torch.stack(
            keep_probabilities, dim=1
        )  # Shape: (num_jobs, num_cascades, num_values)

        result = stacked.prod(dim=1)  # Shape: (num_jobs, num_values)
        return result

    def cost_per_value(
        self,
        all_decision_probabilities: List[
            torch.Tensor
        ],  # Shape: num_cascades x (num_jobs, num_values, 3, num_filters + 1)
        all_pick_scores: List[
            torch.Tensor
        ],  # Shape: num_cascades x (num_jobs, num_filters)
        temperature: float = 1.0,
    ) -> torch.Tensor:  # Shape: (num_jobs, )
        """Calculate the total cost for the entire pipeline."""
        assert len(self.filters_cascades) > 0, "Pipeline must have at least one cascade"
        num_jobs = all_pick_scores[0].shape[0]
        cascade_pointer = [0 for _ in range(len(self.filters_cascades))]

        combined_decision_probabilities = [
            [] for _ in range(len(self.filters_cascades))
        ]
        for cascade_id, filter_id in self.order:
            other_cascades_keep = [
                all_decision_probabilities[i][:, :, (0, 2), j].sum(dim=2)
                for i, j in enumerate(cascade_pointer)
                if i != cascade_id
            ]
            # Shape: (num_jobs, num_values)
            other_cascades_keep = torch.stack(other_cascades_keep).prod(dim=0)
            # Shape: (num_jobs, num_values, 3)
            this_probabs = all_decision_probabilities[cascade_id][:, :, :, filter_id]
            combined_keep = this_probabs[:, :, 0] * other_cascades_keep
            combined_unsure = this_probabs[:, :, 2] * other_cascades_keep
            combined_discard = 1 - combined_keep - combined_unsure
            stacked = torch.stack(
                [combined_keep, combined_discard, combined_unsure], dim=2
            )
            combined_decision_probabilities[cascade_id].append(stacked)
            cascade_pointer[cascade_id] += 1

        total_cost = torch.zeros(num_jobs, device=all_pick_scores[0].device)
        for i, cascade in enumerate(self.filters_cascades):
            this_combined = torch.stack(combined_decision_probabilities[i], dim=3)
            cost_per_value = cascade.cost_per_value(
                decision_probabilities=this_combined,
                pick_score=all_pick_scores[i],
                temperature=temperature,
                remove_last_filter=False,
            )
            total_cost += cost_per_value
        return total_cost

    def sample_cost_per_value(self, pruning_mask: List[torch.Tensor]):
        total_cost = 0
        for cascade, mask in zip(self.filters_cascades, pruning_mask):
            mask = mask.detach().numpy()
            total_cost += cascade.sample_cost_per_value(prune_mask=mask)
        return total_cost


class Sampler(ABC):
    def __init__(self, defensive_mixing_fraction: float = 0.1):
        self.defensive_mixing_fraction = defensive_mixing_fraction

    def get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        """Get the weights for sampling based on the evaluation values."""
        weights = self._get_weights(all_eval_values)
        num_values = all_eval_values[0].shape[0]
        uniform_weights = torch.ones(num_values, dtype=torch.float32)
        weights = (
            1 - self.defensive_mixing_fraction
        ) * weights + self.defensive_mixing_fraction * uniform_weights
        return weights

    @abstractmethod
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        """Get the weights for sampling based on the evaluation values."""


class UniformSampler(Sampler):
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        """Get uniform weights for sampling."""
        num_values = all_eval_values[0].shape[0]
        weights = torch.ones(num_values, dtype=torch.float32)
        return weights


class ProductSamplerAll(Sampler):
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        # proxy scores = 1 - eval_scores, since low eval_scores means more likely to be positive
        weights = (1 - torch.stack(all_eval_values)).prod(0)
        return weights


class SqrtProductSamplerAll(Sampler):
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        weights = (1 - torch.stack(all_eval_values)).prod(0)
        return weights.sqrt()


class ProductSamplerAny(Sampler):
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        weights = 1 - torch.stack(all_eval_values).prod(0)
        return weights


class SqrtProductSamplerAny(Sampler):
    def _get_weights(self, all_eval_values: List[torch.Tensor]) -> torch.Tensor:
        weights = 1 - torch.stack(all_eval_values).prod(0)
        return weights.sqrt()


class Experiment:
    def __init__(
        self,
        num_sample_values_per_batch: int,
        num_eval_values: int,
        selectivities: List[float],
        rng: torch.Generator,
    ):
        self.selectivities = selectivities
        self.num_cascades = len(selectivities)
        self.num_sample_values_per_batch = num_sample_values_per_batch
        self.num_sample_values = 0
        self.num_eval_values = num_eval_values
        self.rng = rng

        self.all_eval_values = [
            torch.rand(num_eval_values, generator=self.rng)
            for _ in range(self.num_cascades)
        ]  # Shape: (num_cascades, num_values)
        self.all_eval_labels = [
            torch.Tensor(
                [
                    (
                        Prediction.KEEP
                        if self.all_eval_values[i][j] < selectivities[i]
                        else Prediction.DISCARD
                    )
                    for j in range(num_eval_values)
                ]
            )
            for i in range(self.num_cascades)
        ]
        self.sampled_mask = torch.zeros(num_eval_values, dtype=torch.bool)
        self.sampled_ids = torch.zeros(0, dtype=torch.int64)
        self.current_sample = torch.zeros(0, dtype=torch.int64)
        self.eval_ids = torch.arange(num_eval_values, dtype=torch.int64)

    def set_sampler(self, sampler: Sampler):
        self.sampler = sampler
        self.sample_weights = self.sampler.get_weights(self.all_eval_values)
        self.sample_new()  # Initialize the first sample values and labels

    def sample_new(self, num_new_samples: Optional[int] = None):
        self.num_new_samples = (
            self.num_sample_values_per_batch
            if num_new_samples is None
            else num_new_samples
        )
        weights = self.sample_weights * (~self.sampled_mask).float()
        self.current_sample = torch.multinomial(
            input=weights,
            num_samples=self.num_new_samples,
            replacement=False,
            generator=self.rng,
        )
        self.sampled_ids = torch.cat((self.sampled_ids, self.current_sample))
        self.num_sample_values += self.num_new_samples
        self.all_sample_values = [
            self.all_eval_values[i][self.sampled_ids] for i in range(self.num_cascades)
        ]  # Shape: (num_cascades, num_values)
        self.all_sample_labels = [
            self.all_eval_labels[i][self.sampled_ids] for i in range(self.num_cascades)
        ]  # Shape: (num_cascades, num_values)
        self.sampled_mask[self.current_sample] = True

        self.current_sample_values = [
            self.all_eval_values[i][self.current_sample]
            for i in range(self.num_cascades)
        ]
        self.current_sample_labels = [
            self.all_eval_labels[i][self.current_sample]
            for i in range(self.num_cascades)
        ]

    def reset(self):
        self.num_sample_values = 0
        self.sample_new()

    def compute_metrics(
        self,
        keep_proba: torch.Tensor,  # Shape: (num_jobs, num_values)
        batch_size: int,
        num_budgets_to_test: int,
        data_subset: Literal["full", "full_sample", "current_sampled"],
        use_single_cascades_positives: int = -1,
        one_sided_bounds: bool = True,
        confidence: float = CONFIDENCE,
    ) -> torch.Tensor:  # Shape: (num_jobs, 4)
        """Compute precision and recall based on the keep probabilities and labels."""
        device = keep_proba.device
        match data_subset:
            case "full":
                all_labels = self.all_eval_labels
                sample_weights = torch.ones(len(all_labels[0]), device=device)
            case "full_sample":
                all_labels = self.all_sample_labels
                sample_weights = self.sample_weights[self.sampled_ids].to(device)
            case "current_sampled":
                all_labels = self.current_sample_labels
                sample_weights = self.sample_weights[self.current_sample].to(device)

        reweighing_factor = 1 / (sample_weights / sample_weights.sum())
        positives = torch.stack(
            [labels == Prediction.KEEP for labels in all_labels],
        )  # Shape: (num_cascades, num_values)
        if use_single_cascades_positives >= 0:
            positives = positives[use_single_cascades_positives]  # Shape: (num_values,)
        else:
            positives = positives.all(dim=0)  # Shape: (num_values,)
        negatives = ~positives  # Invert the positives to get negatives
        total_num_values = len(positives)
        num_jobs = keep_proba.shape[0]

        true_positives = keep_proba * positives.reshape(1, -1)
        false_negatives = (1 - keep_proba) * (positives).reshape(1, -1)
        false_positives = keep_proba * (negatives).reshape(1, -1)

        precision = (true_positives * reweighing_factor).sum(dim=1) / (
            ((true_positives + false_positives) * reweighing_factor).sum(1) + 0.0001
        )
        recall = (true_positives * reweighing_factor).sum(dim=1) / (
            ((true_positives + false_negatives) * reweighing_factor).sum(dim=1) + 0.0001
        )

        # num_samples_precision = (
        #     true_positives.sum(dim=1) + false_positives.sum(dim=1).detach()
        # )
        num_samples_precision = (
            (true_positives + false_positives).round() * reweighing_factor
        ).sum(dim=1).square() / (
            (true_positives + false_positives).round() * reweighing_factor + 0.0001
        ).square().sum(dim=1).detach()
        num_samples_precision = num_samples_precision.clamp(min=1.0)

        what_if_more_precision_samples = self.get_what_if_more_samples(
            num_samples=num_samples_precision,
            total_num_values=total_num_values,
            num_jobs=num_jobs,
            num_budgets_to_test=num_budgets_to_test,
            batch_size=batch_size,
            device=device,
        )
        precision_lower, precision_upper = self.bounds(
            value=precision,
            num_samples=num_samples_precision + what_if_more_precision_samples,
            confidence=confidence,
            one_sided=one_sided_bounds,
        )
        # num_samples_recall = (
        #     true_positives.sum(dim=1) + false_negatives.sum(dim=1).detach()
        # )
        num_samples_recall = (
            (true_positives + false_negatives).round() * reweighing_factor
        ).sum(dim=1).square() / (
            (true_positives + false_negatives).round() * reweighing_factor + 0.0001
        ).square().sum(dim=1).detach()
        num_samples_recall = num_samples_recall.clamp(min=1.0)
        what_if_more_recall_samples = self.get_what_if_more_samples(
            num_samples=num_samples_recall,
            total_num_values=total_num_values,
            num_jobs=num_jobs,
            num_budgets_to_test=num_budgets_to_test,
            batch_size=batch_size,
            device=device,
        )
        recall_lower, recall_upper = self.bounds(
            value=recall,
            num_samples=num_samples_recall + what_if_more_recall_samples,
            confidence=confidence,
            one_sided=one_sided_bounds,
        )
        return torch.stack(
            [
                precision_lower,
                precision,
                precision_upper,
                recall_lower,
                recall,
                recall_upper,
            ],
            dim=1,
        )

    @staticmethod
    def get_what_if_more_samples(
        num_samples: torch.Tensor,
        total_num_values: int,
        num_jobs: int,
        num_budgets_to_test: int,
        batch_size: int,
        device: torch.device,
    ):
        per_sample = num_samples / total_num_values
        multipliers = Experiment.get_what_if_multipliers(
            batch_size, num_jobs, num_budgets_to_test, device=device
        )
        what_if_more_samples = per_sample * multipliers
        return what_if_more_samples

    @staticmethod
    def get_what_if_multipliers(
        batch_size: int, num_jobs: int, num_budgets_to_test: int, device: torch.device
    ):
        multiplier = batch_size * 2 ** (
            torch.arange(num_budgets_to_test, device=device) - 1
        )
        multiplier = multiplier.repeat_interleave(
            num_jobs // num_budgets_to_test
        )  # Shape: (num_jobs, num_budgets_to_test)
        return multiplier

    def bounds(self, value, num_samples, confidence=CONFIDENCE, one_sided=True):
        if one_sided:
            z = torch.tensor(norm.ppf(confidence))
        else:
            z = torch.tensor(norm.ppf((1 + confidence) / 2))
        demon = 1 + z**2 / num_samples
        center = value + z**2 / (2 * num_samples)
        margin = z * torch.sqrt(
            (value * (1 - value) + z**2 / (4 * num_samples)) / num_samples
        )
        lower = ((center - margin) / demon).clamp(min=0.0)
        upper = ((center + margin) / demon).clamp(max=1.0)
        return lower, upper


class Model:
    def __init__(
        self,
        pipeline: Pipeline,
        rng: torch.Generator,
        num_initializations: int = 1,
        num_budgets_to_test: int = 1,
    ):
        self.pipeline = pipeline
        self.num_cascades = len(pipeline.filters_cascades)
        self.num_jobs = num_initializations * num_budgets_to_test
        self.num_initializations = num_initializations
        self.num_budgets = num_budgets_to_test
        self.rng = rng
        self.init()

    def prune(self, cascade_id, filter_id):
        self.pruning_mask[cascade_id][:, filter_id] = True

    @property
    def all_pick_scores(self):
        return [
            torch.where(self.pruning_mask[i], -1000, x)
            for i, x in enumerate(self._all_pick_scores)
        ]

    def init(self, reset_pruning_mask: bool = True):
        """Initialize the model parameters."""
        self._all_pick_scores = [
            torch.nn.Parameter(torch.zeros(self.num_jobs, len(cascade.filters)))
            for cascade in self.pipeline.filters_cascades
        ]  # Shape: (num_cascades, num_filters)
        self.all_params = [
            torch.nn.Parameter(torch.tensor([[0.0, 1.0]] * self.num_jobs))
            for _ in self.pipeline.filters_cascades
        ]  # Shape: (num_cascades, 2)

        # Randomly initialize all but the one for each budget
        first_initialization_indices = (
            torch.arange(self.num_budgets, device=self._all_pick_scores[0].device)
            * self.num_initializations
        )
        with torch.no_grad():
            for pick_scores in self._all_pick_scores:
                pick_scores.uniform_(-1.0, 1.0, generator=self.rng)
                pick_scores[first_initialization_indices, :] = 0.0
                pick_scores[:, -1] = 1.0  # Last filter is always used
            for params in self.all_params:
                params.uniform_(0.0, 1.0, generator=self.rng)
                params[first_initialization_indices, 0] = 0.0
                params[first_initialization_indices, 1] = 1.0

        if reset_pruning_mask:
            self.pruning_mask = [
                torch.zeros_like(x, dtype=torch.bool) for x in self._all_pick_scores
            ]

    def to(self, device: torch.device):
        """Move the model to the specified device."""
        for pick_scores in self._all_pick_scores:
            pick_scores.data = pick_scores.data.to(device)
        for params in self.all_params:
            params.data = params.data.to(device)
        for mask in self.pruning_mask:
            mask.to(device)
        return self


@dataclass
class OptimizationConfig:
    selectivity_threshold_for_splitting_target: float = 0.1
    num_steps: int = 300
    model_mixing: bool = False
    learning_rate: float = 0.02
    lr_decay: float = 0.004
    temperature_decay_pick: float = 0.0001
    temperature_decay_params: float = 0.009
    begin_temperature: float = 0.1
    begin_violation_loss_multiplier: float = 1.0
    violation_loss_multiplier_growth: float = 0.0004
    num_initializations: int = 1
    proportion_choose_operators: float = 0.26
    operator_pruning_strategy: OperatorPruningStrategy = (
        OperatorPruningStrategy.INDIVIDUAL_COMPARISON
    )
    sampling_strategy: SamplingStrategy = SamplingStrategy.SQRT_PRODUCT_ANY
    defensive_mixing_fraction: float = 0.1


class OptimizationResult:
    def __init__(
        self,
        num_budgets_to_test: int,
        num_initializations: int,
    ):
        self.split_target: bool = False
        self.num_budgets_to_test: int = num_budgets_to_test
        self.num_initializations: int = num_initializations
        self.num_jobs: int = num_budgets_to_test * num_initializations
        self.cost_full: np.ndarray  # Shape: (num_jobs, )
        self.cost_sample: np.ndarray  # Shape: (num_jobs, )
        self.cost_split: np.ndarray  # Shape: (num_jobs, num_cascades)
        self.pick_scores: List[np.ndarray]
        self.params: List[np.ndarray]
        self.precision_lower: np.ndarray  # Shape: (num_jobs, )
        self.recall_lower: np.ndarray  # Shape: (num_jobs, )
        self.precision_full: np.ndarray  # Shape: (num_jobs, )
        self.recall_full: np.ndarray  # Shape: (num_jobs, )
        self.precision_sample: np.ndarray  # Shape: (num_jobs, )
        self.recall_sample: np.ndarray  # Shape: (num_jobs, )
        self.precision_lower_split: np.ndarray  # Shape: (num_jobs, num_cascades)
        self.recall_lower_split: np.ndarray  # Shape: (num_jobs, num_cascades)
        self.precision_split: np.ndarray  # Shape: (num_jobs, num_cascades)
        self.recall_split: np.ndarray  # Shape: (num_jobs, num_cascades)
        self.actual_profiling_cost: float = 0.0

    def update(
        self,
        cost_full: np.ndarray,
        cost_sample: np.ndarray,
        cost_split: np.ndarray,
        pick_scores: List[np.ndarray],
        params: List[np.ndarray],
        precision_lower: np.ndarray,
        recall_lower: np.ndarray,
        precision_full: np.ndarray,
        recall_full: np.ndarray,
        precision_sample: np.ndarray,
        recall_sample: np.ndarray,
        precision_lower_split: np.ndarray,
        recall_lower_split: np.ndarray,
        precision_split: np.ndarray,
        recall_split: np.ndarray,
        actual_profiling_cost: float,
    ):
        self.cost_full = cost_full
        self.cost_sample = cost_sample
        self.cost_split = cost_split
        self.pick_scores = pick_scores
        self.params = params
        self.precision_lower = precision_lower
        self.recall_lower = recall_lower
        self.precision_full = precision_full
        self.recall_full = recall_full
        self.precision_sample = precision_sample
        self.recall_sample = recall_sample
        self.precision_lower_split = precision_lower_split
        self.recall_lower_split = recall_lower_split
        self.precision_split = precision_split
        self.recall_split = recall_split
        self.actual_profiling_cost += actual_profiling_cost

    def satisfies_target_lower_global(self, target: Target) -> np.ndarray:
        result = (self.precision_lower >= target.precision_target) & (
            self.recall_lower >= target.recall_target
        )
        return result

    def satisfies_target_lower_split(self, target: Target) -> np.ndarray:
        result = (self.precision_lower >= target.precision_target) & (
            self.recall_lower >= target.recall_target
        )
        return ~result

    def satisfies_target_full(self, target: Target) -> np.ndarray:
        result = (self.precision_full >= target.precision_target) & (
            self.recall_full >= target.recall_target
        )
        return result

    def achieved_precision(self, target: Target, experiment: Experiment, model: Model):
        """Get the achieved precision for the full dataset."""
        cost_analysis_df = self.get_cost_analysis_df(
            target=target, experiment=experiment, model=model
        )
        df = cost_analysis_df.loc[0].reset_index(drop=True)
        cheapest_idx = df["total_cost_and_target_violation"].argmin()
        return self.precision_full[cheapest_idx]

    def achieved_recall(self, target: Target, experiment: Experiment, model: Model):
        """Get the achieved recall for the full dataset."""
        cost_analysis_df = self.get_cost_analysis_df(
            target=target, experiment=experiment, model=model
        )
        df = cost_analysis_df.loc[0].reset_index(drop=True)
        cheapest_idx = df["total_cost_and_target_violation"].argmin()
        return self.recall_full[cheapest_idx]

    def get_metrics_df(self, experiment: Experiment):
        """Convert the optimization result to a DataFrame."""
        data = {
            "cost_sample": self.cost_sample,
            "cost_full": self.cost_full,
            "precision_lower": self.precision_lower,
            "recall_lower": self.recall_lower,
            "precision_full": self.precision_full,
            "recall_full": self.recall_full,
            "precision_sample": self.precision_sample,
            "recall_sample": self.recall_sample,
        }
        index = Experiment.get_what_if_multipliers(
            batch_size=experiment.num_sample_values_per_batch,
            num_jobs=len(self.cost_sample),
            num_budgets_to_test=self.num_budgets_to_test,
            device=torch.device("cpu"),
        ).tolist()

        df = pd.DataFrame(data, index=index)
        return df

    def get_split_metrics_df(self, experiment: Experiment):
        """Convert the split metrics to a DataFrame."""
        num_cascades = len(self.pick_scores)
        data = {}
        for i in range(num_cascades):
            data.update(
                {
                    f"precision_lower_{i}": self.precision_lower_split[:, i],
                    f"recall_lower_{i}": self.recall_lower_split[:, i],
                    f"precision_{i}": self.precision_split[:, i],
                    f"recall_{i}": self.recall_split[:, i],
                    f"cost_{i}": self.cost_split[:, i],
                }
            )
        index = Experiment.get_what_if_multipliers(
            batch_size=experiment.num_sample_values_per_batch,
            num_jobs=len(self.cost_sample),
            num_budgets_to_test=self.num_budgets_to_test,
            device=torch.device("cpu"),
        ).tolist()
        df = pd.DataFrame(data, index=index)
        return df

    def get_pipeline_config_df(self, experiment: Experiment):
        num_cascades = len(self.pick_scores)
        data = {}
        for i in range(num_cascades):
            for j in range(self.pick_scores[i].shape[1]):
                data[f"pick_score_{i}_{j}"] = self.pick_scores[i][:, j].tolist()
        for i in range(num_cascades):
            for j in range(self.params[i].shape[1]):
                data[f"params_{i}_{j}"] = self.params[i][:, j].tolist()

        index = Experiment.get_what_if_multipliers(
            batch_size=experiment.num_sample_values_per_batch,
            num_jobs=len(self.cost_sample),
            num_budgets_to_test=self.num_budgets_to_test,
            device=torch.device("cpu"),
        ).tolist()
        df = pd.DataFrame(data, index=index)
        return df

    def get_cost_analysis_df(
        self, target: Target, experiment: Experiment, model: Model
    ):
        target_met_global = self.satisfies_target_lower_global(target)
        target_met_split = self.satisfies_target_lower_split(target)
        target_met = target_met_split if self.split_target else target_met_global

        sampling_cost_per_value = model.pipeline.sample_cost_per_value(
            model.pruning_mask
        )
        num_new_samples = Experiment.get_what_if_multipliers(
            batch_size=experiment.num_sample_values_per_batch,
            num_jobs=len(self.cost_sample),
            num_budgets_to_test=self.num_budgets_to_test,
            device=torch.device("cpu"),
        ).numpy()
        sampling_cost = num_new_samples * sampling_cost_per_value

        remaining_tuples = (
            experiment.num_eval_values - experiment.num_sample_values
        ) - num_new_samples
        remaining_tuples = np.clip(remaining_tuples, a_min=0, a_max=None)

        execution_cost = self.cost_sample * remaining_tuples
        total_cost = execution_cost + sampling_cost
        total_cost_and_target_violation = total_cost + (1 - target_met) * 1000000

        data = {
            "sampling_cost": sampling_cost,
            "remaining_tuples": remaining_tuples,
            "execution_cost": execution_cost,
            "total_cost": total_cost,
            "target_met": target_met,
            "target_met_global": target_met_global,
            "target_met_split": target_met_split,
            "total_cost_and_target_violation": total_cost_and_target_violation,
        }
        df = pd.DataFrame(data, index=num_new_samples.tolist())  # type: ignore
        return df

    def pprint(self, experiment: Experiment, target: Target, model: Model, full=False):
        """Pretty print the optimization result."""
        metrics_df = self.get_metrics_df(experiment)
        split_df = self.get_split_metrics_df(experiment)
        config_df = self.get_pipeline_config_df(experiment)
        cost_analysis_df = self.get_cost_analysis_df(
            target=target, experiment=experiment, model=model
        )
        metrics_df["precision_target"] = target.precision_target
        metrics_df["recall_target"] = target.recall_target
        metrics_df["target_met_split"] = self.satisfies_target_lower_split(target)
        metrics_df["target_met_global"] = self.satisfies_target_lower_global(target)
        metrics_df["target_met_full"] = self.satisfies_target_full(target)

        if full:
            index = np.arange(len(cost_analysis_df))
        else:
            num_initializations = int(len(cost_analysis_df) / self.num_budgets_to_test)

            optimal_offset = cost_analysis_df.groupby(cost_analysis_df.index).apply(
                lambda x: x["total_cost_and_target_violation"].argmin()  # type: ignore
            )
            index = (
                np.arange(self.num_budgets_to_test) * num_initializations
                + optimal_offset
            )

        print("\nOptimization Result:")
        print(metrics_df.iloc[index])
        print("\nSplit Metrics:")
        print(split_df.iloc[index])
        print("\nPipeline Configuration:")
        print(config_df.iloc[index])
        print("\nCost Analysis:")
        print(cost_analysis_df.iloc[index])


class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def optimize(
        self,
        experiment: Experiment,
        target: Target,
        config: OptimizationConfig,
    ):
        """Optimize the model for multiple experiments."""
        match config.sampling_strategy:
            case SamplingStrategy.UNIFORM:
                sampler = UniformSampler(config.defensive_mixing_fraction)
            case SamplingStrategy.PRODUCT_ALL:
                sampler = ProductSamplerAll(config.defensive_mixing_fraction)
            case SamplingStrategy.SQRT_PRODUCT_ALL:
                sampler = SqrtProductSamplerAll(config.defensive_mixing_fraction)
            case SamplingStrategy.PRODUCT_ANY:
                sampler = ProductSamplerAny(config.defensive_mixing_fraction)
            case SamplingStrategy.SQRT_PRODUCT_ANY:
                sampler = SqrtProductSamplerAny(config.defensive_mixing_fraction)
            case default:
                raise NotImplementedError(default)
        experiment.set_sampler(sampler)

        result = OptimizationResult(
            num_budgets_to_test=self.model.num_budgets,
            num_initializations=self.model.num_initializations,
        )
        i = 0
        while True:
            self.model.init(reset_pruning_mask=False)
            self.optimize_best_strategy(
                experiment=experiment, result=result, target=target, config=config
            )
            self.prune_operators(
                experiment=experiment,
                pruning_strategy=config.operator_pruning_strategy,
                num_initializations=config.num_initializations,
            )
            terminate, num_new_samples, min_index = self.get_num_new_sample_next_run(
                experiment, result, target
            )
            if terminate or i >= MAX_NUM_ITERATIONS:
                break

            experiment.sample_new(num_new_samples=num_new_samples)
            i += 1
        cost, target_met = self.print_optimization_summary(
            experiment, result, target=target, index=min_index
        )
        experiment.reset()
        return cost, target_met, result

    def optimize_best_strategy(
        self,
        experiment: Experiment,
        result: OptimizationResult,
        target: Target,
        config: OptimizationConfig,
    ):
        all_labels = experiment.all_sample_labels
        sample_selectivity = (
            torch.stack([label == Prediction.KEEP for label in all_labels])
            .prod(0)
            .float()
            .mean()
        )

        if sample_selectivity >= config.selectivity_threshold_for_splitting_target:
            result.split_target = False
            self.optimize_single_global(
                experiment=experiment,
                result=result,
                target=target,
                config=config,
            )
        else:
            result.split_target = True
            self.optimize_single_split_target(
                experiment=experiment,
                result=result,
                target=target,
                config=config,
            )

    def prune_operators(
        self,
        experiment: Experiment,
        pruning_strategy: OperatorPruningStrategy,
        num_initializations: int,
    ):
        match pruning_strategy:
            case OperatorPruningStrategy.NONE:
                return
            case OperatorPruningStrategy.NEVER_PICKED:
                prune_collect = self.never_picked_pruning(experiment=experiment)
            case OperatorPruningStrategy.INDIVIDUAL_COMPARISON:
                prune_collect = self.individual_comparison_pruning(
                    experiment=experiment, num_initializations=num_initializations
                )
            case default:
                raise NotImplementedError(default)

        for i, j in prune_collect:
            self.model.prune(i, j)

    def individual_comparison_pruning(
        self, experiment: Experiment, num_initializations: int
    ) -> List[Tuple[int, int]]:
        prune_collect = []
        for i, cascade in enumerate(self.model.pipeline.filters_cascades):
            for j_cheaper, filter_cheaper in enumerate(cascade.filters[:-1]):
                already_pruned = self.model.pruning_mask[i][:, j_cheaper].all().item()
                if already_pruned:
                    continue

                for j_costlier, filter_costlier in zip(
                    range(j_cheaper + 1, len(cascade.filters)),
                    cascade.filters[j_cheaper + 1 :],
                ):
                    already_pruned = (
                        self.model.pruning_mask[i][:, j_costlier].all().item()
                    )
                    if already_pruned:
                        continue

                    print(
                        f"(Cascade {i}) Comparing filter {j_cheaper} (cost {filter_cheaper.cost}) with {j_costlier} (cost {filter_costlier.cost})"
                    )
                    (
                        cheaper_precision_lower,
                        _,
                        _,
                        cheaper_recall_lower,
                        _,
                        _,
                        _,
                    ) = self.training_evaluate_split(
                        experiment=experiment,
                        temperature_pick=0.000001,
                        temperature_params=0.000001,
                        data_subset="full_sample",
                        simulate_removal=[
                            (i, j) for j in range(j_costlier + 1) if j != j_cheaper
                        ],
                        simulate_inclusion=[(i, j_cheaper)],
                    )
                    (
                        _,
                        _,
                        costlier_precision_upper,
                        _,
                        _,
                        costlier_recall_upper,
                        _,
                    ) = self.training_evaluate_split(
                        experiment=experiment,
                        temperature_pick=0.000001,
                        temperature_params=0.000001,
                        data_subset="full_sample",
                        simulate_removal=[(i, j) for j in range(j_costlier)],
                        simulate_inclusion=[(i, j_costlier)],
                    )
                    cheaper_precision_lower = cheaper_precision_lower[
                        :num_initializations, i
                    ]
                    cheaper_recall_lower = cheaper_recall_lower[:num_initializations, i]
                    costlier_precision_upper = costlier_precision_upper[
                        :num_initializations, i
                    ]
                    costlier_recall_upper = costlier_recall_upper[
                        :num_initializations, i
                    ]
                    precision_no_overlap = (
                        cheaper_precision_lower > costlier_precision_upper
                    )
                    recall_no_overlap = cheaper_recall_lower > costlier_recall_upper

                    prune_decision = (
                        (precision_no_overlap & recall_no_overlap).float().mean()
                    )
                    if prune_decision > 0.5:
                        print(
                            f"Pruning filter {j_costlier} with cost {filter_costlier.cost} since it is dominated by filter {j_cheaper} with cost {filter_cheaper.cost}. Prune decision: {prune_decision}"
                        )
                        prune_collect.append((i, j_costlier))

        return prune_collect

    def never_picked_pruning(self, experiment: Experiment) -> List[Tuple[int, int]]:
        (
            base_precision_lower,
            base_precision,
            _,
            base_recall_lower,
            base_recall,
            _,
            base_cost_sample,
        ) = self.training_evaluate(
            experiment=experiment,
            temperature_pick=0.000001,
            temperature_params=0.000001,
            data_subset="full_sample",
        )
        prune_collect = []
        for i, cascade in enumerate(self.model.pipeline.filters_cascades):
            for j, filter in enumerate(cascade.filters):
                already_pruned = self.model.pruning_mask[i][:, j].all().item()
                if already_pruned:
                    continue
                (
                    rm_precision_lower,
                    rm_precision,
                    _,
                    rm_recall_lower,
                    rm_recall,
                    _,
                    rm_cost_sample,
                ) = self.training_evaluate(
                    experiment=experiment,
                    temperature_pick=0.000001,
                    temperature_params=0.000001,
                    data_subset="full_sample",
                    simulate_removal=[(i, j)],
                )
                num_improvements_with_operator = sum(
                    [
                        (rm_precision_lower < base_precision_lower).sum(),
                        (rm_recall_lower < base_recall_lower).sum(),
                        (rm_precision < base_precision).sum(),
                        (rm_recall < base_recall).sum(),
                        (rm_cost_sample > base_cost_sample).sum(),
                    ]
                )
                print(
                    f"Num improvements with operator {i}, {j}: {num_improvements_with_operator}"
                )
                if (
                    num_improvements_with_operator < 1
                    and filter.cost > PRUNING_COST_THRESHOLD
                    and j < len(cascade.filters) - 1
                ):
                    print(f"pruning filter {i}, {j} with cost {filter.cost}")
                    prune_collect.append((i, j))
        return prune_collect

    def print_optimization_summary(
        self,
        experiment: Experiment,
        result: OptimizationResult,
        target: Target,
        index: int,
    ):
        """Print a summary of the optimization results."""
        sampling_cost_per_value_perfect_pruning = (
            self.model.pipeline.sample_cost_per_value(self.model.pruning_mask)
        )

        sampling_cost_per_value_no_pruning = self.model.pipeline.sample_cost_per_value(
            [(x & False) for x in self.model.pruning_mask]
        )
        num_profiling_samples = experiment.num_sample_values
        perfect_pruning_profiling_cost = (
            num_profiling_samples * sampling_cost_per_value_perfect_pruning
        )
        no_pruning_profiling_cost = (
            num_profiling_samples * sampling_cost_per_value_no_pruning
        )

        num_execution_samples = experiment.num_eval_values - num_profiling_samples
        execution_cost = result.cost_sample * num_execution_samples

        precision = result.precision_full[index]
        recall = result.recall_full[index]
        actual_profiling_cost = result.actual_profiling_cost
        total_cost = execution_cost + actual_profiling_cost
        meets_target = (
            precision >= target.precision_target and recall >= target.recall_target
        )
        print("\nOptimization Summary:")
        print(f"Total Cost: {total_cost[index]:.4f}")
        print(
            f"Profiling Cost: {actual_profiling_cost:.4f} (No pruning: {no_pruning_profiling_cost} / Instant pruning: {perfect_pruning_profiling_cost:.4f})"
        )
        print(f"Execution Cost: {execution_cost[index]:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Meets Target: {meets_target}")
        return total_cost[index], meets_target

    def get_num_new_sample_next_run(
        self, experiment: Experiment, result: OptimizationResult, target: Target
    ) -> Tuple[bool, int, int]:
        result.pprint(experiment=experiment, target=target, model=self.model)
        df = result.get_cost_analysis_df(
            target=target, experiment=experiment, model=self.model
        )
        cheapest_idx = df["total_cost_and_target_violation"].argmin()
        cheapest_num_samples = int(df.index[int(cheapest_idx)])
        target_met = df.iloc[cheapest_idx]["target_met"]
        terminate = (cheapest_num_samples == 0) and target_met
        print("Expected required samples:", cheapest_num_samples)
        print(f"Terminate: {terminate}")
        cheapest_num_samples = max(
            min(int(cheapest_num_samples), NUM_NEW_SAMPLES_MAX), NUM_NEW_SAMPLES_MIN
        )
        print("Sampling new tuples for profiling", cheapest_num_samples)
        return terminate, int(cheapest_num_samples), int(cheapest_idx)

    def optimize_single_global(
        self,
        experiment: Experiment,
        result: OptimizationResult,
        target: Target,
        config: OptimizationConfig,
    ):
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and USE_CUDA_IF_AVAILABLE) else "cpu"
        )
        self.model.to(device)
        relu = torch.nn.ReLU()
        sum_cost = sum(
            f.cost
            for cascade in self.model.pipeline.filters_cascades
            for f in cascade.filters
        )

        for stage in [
            OptimizationStage.CHOOSE_OPERATORS,
            OptimizationStage.CHOOSE_PARAMETERS,
        ]:
            num_steps = int(
                config.num_steps * config.proportion_choose_operators
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else config.num_steps * (1 - config.proportion_choose_operators)
            )
            optimizer_weights = (
                self.model._all_pick_scores + self.model.all_params
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else self.model.all_params
            )
            optimizer = torch.optim.Adam(
                optimizer_weights,
                lr=config.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=1 - config.lr_decay
            )
            for i in tqdm(
                range(num_steps),
                desc=f"Optimization steps {stage.name} (Global)",
                leave=False,
                position=1,
            ):
                temperature_pick, temperature_params = self.get_temp(
                    stage=stage, i=i, config=config
                )

                violation_loss_multiplier = (
                    config.begin_violation_loss_multiplier
                    * math.exp(i * config.violation_loss_multiplier_growth)
                )

                optimizer.zero_grad()
                precision_lower, _, _, recall_lower, _, _, cost = (
                    self.training_evaluate(
                        experiment=experiment,
                        temperature_pick=temperature_pick,
                        temperature_params=temperature_params,
                        data_subset="full_sample",
                    )
                )
                precision_violation = (
                    relu(target.precision_target - precision_lower)
                    * violation_loss_multiplier
                )
                recall_violation = (
                    relu(target.recall_target - recall_lower)
                    * violation_loss_multiplier
                )
                cost_scaled = cost / sum_cost
                loss = precision_violation + recall_violation + cost_scaled

                loss.sum().backward()
                optimizer.step()
                scheduler.step()

        final_temp = 0.000001 if not config.model_mixing else config.begin_temperature
        self.update_optimization_result(
            experiment=experiment,
            result=result,
            temperature=final_temp,
        )

    def get_temp(self, stage: OptimizationStage, i: int, config: OptimizationConfig):
        temperature_pick = config.begin_temperature
        if not config.model_mixing and stage == OptimizationStage.CHOOSE_OPERATORS:
            temperature_pick = config.begin_temperature * math.exp(
                -i * config.temperature_decay_pick
            )
        if stage == OptimizationStage.CHOOSE_PARAMETERS:
            temperature_pick = 0.000001
        temperature_params = config.begin_temperature * math.exp(
            -i * config.temperature_decay_params
        )
        return temperature_pick, temperature_params

    def optimize_single_split_target(
        self,
        experiment: Experiment,
        result: OptimizationResult,
        target: Target,
        config: OptimizationConfig,
    ):
        relu = torch.nn.ReLU()
        sum_cost = sum(
            f.cost
            for cascade in self.model.pipeline.filters_cascades
            for f in cascade.filters
        )

        for stage in [
            OptimizationStage.CHOOSE_OPERATORS,
            OptimizationStage.CHOOSE_PARAMETERS,
        ]:
            num_steps = int(
                config.num_steps * config.proportion_choose_operators
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else config.num_steps * (1 - config.proportion_choose_operators)
            )
            optimizer_weights = (
                self.model._all_pick_scores + self.model.all_params
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else self.model.all_params
            )
            optimizer = torch.optim.Adam(
                optimizer_weights,
                lr=config.learning_rate,
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=1 - config.lr_decay
            )
            for i in tqdm(
                range(num_steps),
                desc=f"Optimization steps {stage.name} (Split Target)",
                leave=False,
                position=1,
            ):
                temperature_pick, temperature_params = self.get_temp(
                    stage=stage, i=i, config=config
                )
                violation_loss_multiplier = (
                    config.begin_violation_loss_multiplier
                    * math.exp(i * config.violation_loss_multiplier_growth)
                )

                optimizer.zero_grad()
                precision_lower, _, _, recall_lower, _, _, cost = (
                    self.training_evaluate_split(
                        experiment=experiment,
                        temperature_pick=temperature_pick,
                        temperature_params=temperature_params,
                        data_subset="full_sample",
                    )
                )
                combined_precision_lower = precision_lower.prod(dim=1)
                combined_recall_lower = recall_lower.prod(dim=1)

                precision_violation = (
                    relu(target.precision_target - combined_precision_lower)
                    * violation_loss_multiplier
                )
                recall_violation = (
                    relu(target.recall_target - combined_recall_lower)
                    * violation_loss_multiplier
                )
                cost_scaled = cost.sum(1) / sum_cost
                loss = precision_violation + recall_violation + cost_scaled

                loss.sum().backward()
                optimizer.step()
                scheduler.step()

        final_temp = 0.000001 if not config.model_mixing else config.begin_temperature
        self.update_optimization_result(
            experiment=experiment, result=result, temperature=final_temp
        )

    def update_optimization_result(
        self, experiment: Experiment, result: OptimizationResult, temperature: float
    ):
        _, precision_full, _, _, recall_full, _, cost_full = self.training_evaluate(
            experiment=experiment,
            temperature_pick=temperature,
            temperature_params=0.000001,
            data_subset="full",
        )
        (
            _,
            precision_split_full,
            _,
            _,
            recall_split_full,
            _,
            _,
        ) = self.training_evaluate_split(
            experiment=experiment,
            temperature_pick=temperature,
            temperature_params=0.000001,
            data_subset="full",
        )
        (
            precision_lower,
            precision_sample,
            _,
            recall_lower,
            recall_sample,
            _,
            cost_full_sample,
        ) = self.training_evaluate(
            experiment=experiment,
            temperature_pick=temperature,
            temperature_params=0.000001,
            data_subset="full_sample",
        )
        (
            precision_lower_split,
            _,
            _,
            recall_lower_split,
            _,
            _,
            cost_full_sample_split,
        ) = self.training_evaluate_split(
            experiment=experiment,
            temperature_pick=temperature,
            temperature_params=0.000001,
            data_subset="full_sample",
        )
        actual_profiling_cost = (
            self.model.pipeline.sample_cost_per_value(self.model.pruning_mask)
            * experiment.num_new_samples
        )

        result.update(
            cost_full=cost_full.detach().cpu().numpy(),
            cost_split=cost_full_sample_split.detach().cpu().numpy(),
            cost_sample=cost_full_sample.detach().cpu().numpy(),
            pick_scores=[x.detach().cpu().numpy() for x in self.model.all_pick_scores],
            params=[x.detach().cpu().numpy() for x in self.model.all_params],
            precision_lower=precision_lower.detach().cpu().numpy(),
            recall_lower=recall_lower.detach().cpu().numpy(),
            precision_full=precision_full.detach().cpu().numpy(),
            recall_full=recall_full.detach().cpu().numpy(),
            precision_sample=precision_sample.detach().cpu().numpy(),
            recall_sample=recall_sample.detach().cpu().numpy(),
            precision_lower_split=precision_lower_split.detach().cpu().numpy(),
            recall_lower_split=recall_lower_split.detach().cpu().numpy(),
            precision_split=precision_split_full.detach().cpu().numpy(),
            recall_split=recall_split_full.detach().cpu().numpy(),
            actual_profiling_cost=actual_profiling_cost,
        )

    def training_evaluate(
        self,
        experiment: Experiment,
        temperature_pick: float,
        temperature_params: float,
        data_subset: Literal["full", "full_sample", "current_sampled"],
        simulate_removal: Sequence[Tuple[int, int]] = (),
    ):
        match data_subset:
            case "full":
                sample_values = experiment.all_eval_values
                sample_labels = experiment.all_eval_labels
                value_indexes = experiment.eval_ids
            case "full_sample":
                sample_values = experiment.all_sample_values
                sample_labels = experiment.all_sample_labels
                value_indexes = experiment.sampled_ids
            case "current_sampled":
                sample_values = experiment.current_sample_values
                sample_labels = experiment.current_sample_labels
                value_indexes = experiment.current_sample

        all_pick_scores = list(self.model.all_pick_scores)
        for i, j in simulate_removal:
            remove_pick_score = all_pick_scores[i].clone().detach()
            remove_pick_score[:, j] -= 1000
            all_pick_scores[i] = remove_pick_score
        all_decision_probabilities = self.model.pipeline.decision_probabilities(
            value_indexes=value_indexes,
            all_values=sample_values,  # Values to filter
            all_labels=sample_labels,
            all_pick_scores=all_pick_scores,
            all_params=self.model.all_params,
            temperature_pick=temperature_pick,
            temperature_params=temperature_params,
        )

        cost = self.model.pipeline.cost_per_value(
            all_decision_probabilities=all_decision_probabilities,
            all_pick_scores=self.model.all_pick_scores,
            temperature=temperature_pick,
        )
        keep_proba = self.model.pipeline.keep_proba(all_decision_probabilities)
        (
            precisions_lower,
            precisions,
            precisions_upper,
            recalls_lower,
            recalls,
            recalls_upper,
        ) = experiment.compute_metrics(
            keep_proba,
            batch_size=experiment.num_sample_values_per_batch,
            num_budgets_to_test=self.model.num_budgets,
            data_subset=data_subset,
        ).T
        return (
            precisions_lower,
            precisions,
            precisions_upper,
            recalls_lower,
            recalls,
            recalls_upper,
            cost,
        )

    def training_evaluate_split(
        self,
        experiment: Experiment,
        temperature_pick,
        temperature_params,
        data_subset: Literal["full", "full_sample", "current_sampled"],
        simulate_removal: Sequence[Tuple[int, int]] = (),
        simulate_inclusion: Sequence[Tuple[int, int]] = (),
        one_sided_bounds: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        match data_subset:
            case "full":
                sample_values = experiment.all_eval_values
                sample_labels = experiment.all_eval_labels
                value_indexes = experiment.eval_ids
            case "full_sample":
                sample_values = experiment.all_sample_values
                sample_labels = experiment.all_sample_labels
                value_indexes = experiment.sampled_ids
            case "current_sampled":
                sample_values = experiment.current_sample_values
                sample_labels = experiment.current_sample_labels
                value_indexes = experiment.current_sample

        all_pick_scores = list(self.model.all_pick_scores)
        for i, j in simulate_removal:
            remove_pick_score = all_pick_scores[i].clone().detach()
            remove_pick_score[:, j] -= 1000
            all_pick_scores[i] = remove_pick_score
        for i, j in simulate_inclusion:
            include_pick_score = all_pick_scores[i].clone().detach()
            include_pick_score[:, j] += 1000
            all_pick_scores[i] = include_pick_score

        all_decision_probabilities = self.model.pipeline.decision_probabilities(
            value_indexes=value_indexes,
            all_values=sample_values,  # Values to filter
            all_labels=sample_labels,
            all_pick_scores=all_pick_scores,
            all_params=self.model.all_params,
            temperature_pick=temperature_pick,
            temperature_params=temperature_params,
        )

        collected_costs = []
        collected_precisions_lower = []
        collected_precisions_upper = []
        collected_recalls_lower = []
        collected_recalls_upper = []
        collected_precisions = []
        collected_recalls = []
        for i, cascade in enumerate(self.model.pipeline.filters_cascades):
            cost = cascade.cost_per_value(
                decision_probabilities=all_decision_probabilities[i],
                pick_score=self.model.all_pick_scores[i],
                temperature=temperature_pick,
            )
            keep_proba = cascade.keep_proba(all_decision_probabilities[i])
            (
                precisions_lower,
                precisions,
                precision_upper,
                recalls_lower,
                recalls,
                recalls_upper,
            ) = experiment.compute_metrics(
                keep_proba,
                batch_size=experiment.num_sample_values_per_batch,
                num_budgets_to_test=self.model.num_budgets,
                use_single_cascades_positives=i,
                data_subset=data_subset,
                one_sided_bounds=one_sided_bounds,
                confidence=(
                    CONFIDENCE ** (1 / (len(self.model.pipeline.filters_cascades)))
                ),
            ).T
            collected_costs.append(cost)
            collected_precisions_lower.append(precisions_lower)
            collected_precisions_upper.append(precision_upper)
            collected_recalls_lower.append(recalls_lower)
            collected_recalls_upper.append(recalls_upper)
            collected_precisions.append(precisions)
            collected_recalls.append(recalls)
        result = (
            torch.stack(collected_precisions_lower, dim=1),
            torch.stack(collected_precisions, dim=1),
            torch.stack(collected_precisions_upper, dim=1),
            torch.stack(collected_recalls_lower, dim=1),
            torch.stack(collected_recalls, dim=1),
            torch.stack(collected_recalls_upper, dim=1),
            torch.stack(collected_costs, dim=1),
        )
        return result  # Shape: 7x (num_jobs, num_cascades)


class ExperimentRunner:
    def __init__(self, seed):
        self.seed = seed

    def random(self, rng: torch.Generator, lower: float, upper: float, log=False):
        if not log:
            return torch.rand((1,), generator=rng).item() * (upper - lower) + lower
        else:
            u = torch.rand((1,))

            log_lower, log_upper = (
                torch.log(torch.tensor(lower)),
                torch.log(torch.tensor(upper)),
            )
            sample = torch.exp(log_lower + (log_upper - log_lower) * u)
            return sample.item()

    def get_pipeline(
        self, num_cascades: int, experiment: Experiment, rng: torch.Generator
    ) -> Pipeline:
        num_eval_values = experiment.num_eval_values

        num_llm_operators = 8
        cascades = []
        for _ in range(num_cascades):
            regex_filter = DecisionFilter(
                fpr=self.random(rng, 0.05, 0.5),
                fnr=self.random(rng, 0.05, 0.5),
                unclear_rate=self.random(rng, 0.0, 0.8),
                cost=0.0,
                num_values=num_eval_values,
                rng=rng,
            )
            similarity_filter = SimilarityFilter(
                cost=0.0,
                noise_level=self.random(rng, 0.1, 0.8),
                num_values=num_eval_values,
                rng=rng,
            )
            llm_filters = []
            costs = []
            for i in range(num_llm_operators):
                cost = self.random(rng, 1.0, (i + 1) * 10.0)
                llm_filter = DecisionFilter(
                    fpr=self.random(rng, 0.1 ** (i + 1), 0.5 ** (i + 1), log=True),
                    fnr=self.random(rng, 0.1 ** (i + 1), 0.5 ** (i + 1), log=True),
                    unclear_rate=self.random(rng, 0.00001, 0.5, log=True),
                    cost=cost,
                    num_values=num_eval_values,
                    rng=rng,
                )
                llm_filters.append(llm_filter)
                costs.append(cost)
            oracle_filter = DecisionFilter(
                fpr=0.0,
                fnr=0.0,
                unclear_rate=0.0,
                cost=self.random(rng, max(costs) + 1, (num_llm_operators + 2) * 10.0),
                num_values=num_eval_values,
                rng=rng,
            )
            cascade = FilterCascade(
                [regex_filter, similarity_filter, *llm_filters, oracle_filter]
            )
            cascades.append(cascade)

        pipeline = Pipeline(cascades)
        return pipeline

    def get_targets(
        self, num_targets, lower=LOWER_TARGET, upper=UPPER_TARGET
    ) -> List[Target]:
        """Define the target precision and recall for the optimization."""
        grid_size = int(np.ceil(np.sqrt(num_targets)))
        precision_values = np.linspace(lower, upper, grid_size)
        recall_values = np.linspace(lower, upper, grid_size)

        targets = []
        for precision in precision_values:
            for recall in recall_values:
                target = Target(
                    precision_target=precision,
                    recall_target=recall,
                )
                targets.append(target)
        targets = sorted(
            targets,
            key=lambda x: (x.precision_target * x.recall_target)
            / (x.precision_target + x.recall_target),
        )
        targets = targets[-num_targets:]  # Limit to num_targets
        return targets

    def get_experiment(self, num_cascades: int, rng: torch.Generator) -> Experiment:
        num_sample_values = NUM_NEW_SAMPLES_MIN
        num_eval_values = DATABASE_SIZE
        upper_selectivity = 0.9
        lower_selectivity = 0.1
        cascade_selectivities = [
            torch.rand(1, generator=rng).item()
            * (upper_selectivity - lower_selectivity)
            + lower_selectivity
            for _ in range(num_cascades)
        ]

        experiment = Experiment(
            num_sample_values_per_batch=num_sample_values,
            num_eval_values=num_eval_values,
            selectivities=cascade_selectivities,
            rng=rng,
        )
        total_pipeline_selectivity = 1.0
        for sel in cascade_selectivities:
            total_pipeline_selectivity *= sel
        print("Total Pipeline Selectivity:", total_pipeline_selectivity)
        return experiment

    def get_config(self, num_initializations: int, num_steps: int, trial: optuna.Trial):
        model_mixing = False
        result = OptimizationConfig(
            model_mixing=model_mixing,
            num_steps=num_steps,
            num_initializations=num_initializations,
            temperature_decay_pick=trial.suggest_float(
                "temperature_decay_pick", 0.0001, 0.01, log=True
            ),
            temperature_decay_params=trial.suggest_float(
                "temperature_decay_params", 0.0001, 0.01, log=True
            ),
            learning_rate=trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            lr_decay=trial.suggest_float("lr_decay", 0.0001, 0.01, log=True),
            begin_temperature=trial.suggest_float(
                "begin_temperature", 0.1, 10.0, log=True
            ),
            begin_violation_loss_multiplier=trial.suggest_float(
                "begin_violation_loss_multiplier", 1.0, 10.0, log=True
            ),
            violation_loss_multiplier_growth=trial.suggest_float(
                "violation_loss_multiplier_growth", 0.0001, 0.01, log=True
            ),
            selectivity_threshold_for_splitting_target=trial.suggest_float(
                "selectivity_threshold_for_splitting_target", 0.0, 1.0
            ),
            proportion_choose_operators=trial.suggest_float(
                "proportion_choose_operators", 0.0, 1.0
            ),
            operator_pruning_strategy=OperatorPruningStrategy(
                trial.suggest_categorical(
                    "operator_pruning_strategy",
                    [
                        OperatorPruningStrategy.NONE,
                        OperatorPruningStrategy.NEVER_PICKED,
                        OperatorPruningStrategy.INDIVIDUAL_COMPARISON,
                    ],
                )
            ),
            sampling_strategy=SamplingStrategy(
                trial.suggest_categorical(
                    "sampling_strategy",
                    [
                        SamplingStrategy.UNIFORM,
                        SamplingStrategy.SQRT_PRODUCT_ALL,
                        SamplingStrategy.SQRT_PRODUCT_ANY,
                        SamplingStrategy.PRODUCT_ALL,
                        SamplingStrategy.PRODUCT_ANY,
                    ],
                )
            ),
            defensive_mixing_fraction=trial.suggest_float(
                "defensive_mixing_fraction", 0.0, 0.5
            ),
        )
        return result

    def fixed_configs(self, num_initializations: int, num_steps: int):
        model_mixing = False
        result = [
            OptimizationConfig(
                model_mixing=model_mixing,
                num_steps=num_steps,
                num_initializations=num_initializations,
                temperature_decay_pick=0.0002,
                temperature_decay_params=0.009,
                learning_rate=0.001,
                lr_decay=0.0044,
                begin_temperature=0.71,
                begin_violation_loss_multiplier=1.54,
                violation_loss_multiplier_growth=0.0003,
                selectivity_threshold_for_splitting_target=0.9,
                proportion_choose_operators=0.0223,
                operator_pruning_strategy=OperatorPruningStrategy.INDIVIDUAL_COMPARISON,
                sampling_strategy=SamplingStrategy.PRODUCT_ALL,
                defensive_mixing_fraction=0.42,
            )
        ]
        return result

    def get_envirionments(
        self,
        num_environments: int,
        torch_rng: torch.Generator,
    ):
        result = []
        all_num_cascades = torch.randint(
            2, 5, (num_environments,), generator=torch_rng
        ).tolist()
        for num_cascades in all_num_cascades:
            experiment = self.get_experiment(num_cascades=num_cascades, rng=torch_rng)
            pipeline = self.get_pipeline(
                num_cascades=num_cascades, experiment=experiment, rng=torch_rng
            )
            result.append((experiment, pipeline))
        return result

    def get_rng(self):
        torch_rng = torch.Generator()
        torch_rng.manual_seed(self.seed)
        return torch_rng

    def run(
        self,
        num_steps: int = 300,
        max_num_environments: int = 100,
        num_targets: int = 10,
    ):
        torch_rng = self.get_rng()
        environments = self.get_envirionments(
            num_environments=max_num_environments,
            torch_rng=torch_rng,
        )
        targets = self.get_targets(num_targets)
        num_initializations = 10
        num_budgets_to_test = 10
        configs = self.fixed_configs(
            num_initializations=num_initializations,
            num_steps=num_steps,
        )
        start_seed = self.seed
        collect_data = []

        total_runs = len(configs) * len(environments) * len(targets)
        with tqdm(
            total=total_runs,
            desc="Running Guarantee Pushdown Optimization",
            position=0,
            leave=True,
        ) as pbar:
            for i, target in enumerate(targets):
                for j, config in enumerate(configs):
                    for k, (experiment, pipeline) in enumerate(environments):
                        torch_rng = self.get_rng()
                        model = Model(
                            pipeline,
                            num_initializations=num_initializations,
                            num_budgets_to_test=num_budgets_to_test,
                            rng=torch_rng,
                        )

                        optimizer = Optimizer(model)
                        cost, target_met, result = optimizer.optimize(
                            experiment=experiment,
                            target=target,
                            config=config,
                        )
                        achieved_precision = result.achieved_precision(
                            target=target, experiment=experiment, model=model
                        )
                        achieved_recall = result.achieved_recall(
                            target=target, experiment=experiment, model=model
                        )
                        collect_data.append(
                            {
                                "start_seed": start_seed,
                                "seed": self.seed,
                                "config_index": j,
                                "environment_index": k,
                                "target_index": i,
                                "cost": cost,
                                "target_met": target_met,
                                "target_precision": target.precision_target,
                                "target_recall": target.recall_target,
                                "achieved_precision": achieved_precision,
                                "achieved_recall": achieved_recall,
                            }
                        )
                        pbar.update(1)
        df = pd.DataFrame(collect_data)
        df.to_csv("guarantee_pushdown_results.csv", index=False)

        fraction_target_met = df["target_met"].mean()
        num_target_met = df["target_met"].sum()
        print("\nFinal Results:")
        print(
            "Percent of targets met:",
            fraction_target_met * 100,
            f"({num_target_met}/{len(df)})",
        )
        print(
            "Confidence level satisfied:",
            fraction_target_met >= CONFIDENCE,
            f"({CONFIDENCE=})",
        )

    def plot_results(self, csv_path: Path):
        """Plot the results of the optimization."""
        df = pd.read_csv(csv_path)
        precision_range = df["target_precision"].unique()
        recall_range = df["target_recall"].unique()
        precision_range.sort()
        recall_range.sort()
        norm = plt.Normalize(df["cost"].min(), df["cost"].max())  # type: ignore
        cmap = "RdYlGn_r"

        # Grid of plots for each target precision and recall
        g = sns.FacetGrid(
            df,
            row="target_precision",
            col="target_recall",
            margin_titles=True,
        )
        g.map_dataframe(
            sns.scatterplot,
            x="achieved_precision",
            y="achieved_recall",
            hue="cost",
            palette=cmap,
            hue_norm=norm,
            legend=False,  # avoid categorical legend
        )

        for (target_precision, target_recall), ax in g.axes_dict.items():
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axhline(
                y=target_recall, color="r", linestyle="--", label="Target Recall"
            )
            ax.axvline(
                x=target_precision, color="g", linestyle="--", label="Target Precision"
            )

        # One shared continuous colorbar
        g.figure.subplots_adjust(right=0.85, top=0.9, bottom=0.1)
        cbar_ax = g.figure.add_axes(
            (0.88, 0.3, 0.02, 0.4)
        )  # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore
        sm.set_array([])
        g.figure.colorbar(sm, cax=cbar_ax, label="Cost")
        plt.show()

    def run_optimization(
        self,
        storage: str,
        max_num_environments: int = 100,
        num_trials: int = 1,
        worker_id: int = 0,
        num_steps: int = 300,
    ):
        torch_rng = self.get_rng()
        environments = self.get_envirionments(
            num_environments=max_num_environments,
            torch_rng=torch_rng,
        )
        targets = self.get_targets(max_num_environments)
        targets = sorted(targets, key=lambda _: torch.rand(1).item())  # shuffle targets

        study = optuna.load_study(
            study_name=f"guarantee-pushdown-study-{num_steps}",
            sampler=optuna.samplers.TPESampler(seed=self.seed + worker_id),
            storage=storage,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=max_num_environments,
                reduction_factor=2,
            ),
        )
        objective = partial(
            self.objective,
            environments=environments,
            targets=targets,
            num_steps=num_steps,
            rng=torch_rng,
        )
        study.optimize(objective, n_trials=num_trials)

    def objective(
        self,
        trial: optuna.Trial,
        environments: List[Tuple[Experiment, Pipeline]],
        targets: List[Target],
        rng: torch.Generator,
        num_steps: int = 300,
    ):
        assert len(targets) == len(environments)
        num_initializations = 10
        num_budgets_to_test = 10
        config = self.get_config(
            num_initializations=num_initializations, trial=trial, num_steps=num_steps
        )
        sum_cost = 0
        num_target_met = 0

        for i, ((experiment, pipeline), target) in enumerate(
            zip(environments, targets)
        ):
            model = Model(
                pipeline,
                num_initializations=num_initializations,
                num_budgets_to_test=num_budgets_to_test,
                rng=rng,
            )

            optimizer = Optimizer(model)
            cost, target_met, _ = optimizer.optimize(
                experiment=experiment,
                target=target,
                config=config,
            )
            sum_cost += cost
            num_target_met += int(target_met)

            intermediate_reward = self.compute_reward(
                sum_cost=sum_cost,
                num_target_met=num_target_met,
                num_total=i + 1,
            )

            trial.report(intermediate_reward, i)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Trial {trial.number} was pruned at step {i} with intermediate reward {intermediate_reward:.4f}."
                )

        final_reward = self.compute_reward(
            sum_cost=sum_cost,
            num_target_met=num_target_met,
            num_total=len(environments),
        )
        return final_reward

    def compute_reward(self, sum_cost: float, num_target_met: int, num_total: int):
        goal = int(CONFIDENCE * num_total)
        if num_total <= 2:  # no grace if few environments
            goal = num_total

        penality = 0
        if num_target_met < goal:
            num_missing = goal - num_target_met
            assert num_missing > 0
            penality = 10_000_000 * num_missing

        return sum_cost / num_total + penality


def main(
    storage,
    worker_id,
    max_num_environments,
    num_trials,
    search,
    num_steps,
    num_targets,
    plot_results,
):
    runner = ExperimentRunner(seed=18)
    if plot_results:
        runner.plot_results(plot_results)
        return

    if search:
        runner.run_optimization(
            storage=storage,
            max_num_environments=max_num_environments,
            num_trials=num_trials,
            worker_id=worker_id,
            num_steps=num_steps,
        )
    else:
        runner.run(
            max_num_environments=max_num_environments,
            num_steps=num_steps,
            num_targets=num_targets,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///guarantee_pushdown_study.db",
        help="The storage URL for the Optuna study.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="The ID of the worker for distributed optimization.",
    )
    parser.add_argument(
        "--max-num-environments",
        type=int,
        default=10,
        help="Maximum number of environments to run in each trial.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials to run for optimization.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=300,
        help="Number of trials to run for optimization.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run a search for the best hyperparameters.",
    )
    parser.add_argument(
        "--plot-results",
        type=Path,
        default=None,
        help="Path to a CSV file to plot the results.",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=1,
        help="Number of targets to use in the optimization.",
    )

    args = parser.parse_args()
    main(
        args.storage,
        args.worker_id,
        args.max_num_environments,
        args.n_trials,
        args.search,
        args.n_steps,
        args.n_targets,
        args.plot_results,
    )
