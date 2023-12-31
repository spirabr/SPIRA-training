from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NewType, cast

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from typing_extensions import Self

from spira.adapter.config import TrainingOptionsConfig

Prediction = NewType("Prediction", torch.Tensor)
Label = NewType("Label", torch.Tensor)
Loss = torch.FloatTensor


@dataclass
class Validation:
    prediction: Prediction
    label: Label


class SingleLossCalculator(ABC):
    @abstractmethod
    def calculate(self, validation: Validation) -> Loss:
        pass

    @abstractmethod
    def recalculate_weights(self):
        pass

    @abstractmethod
    def clone(self):
        pass


class BCELossCalculator(SingleLossCalculator):
    def __init__(self, reduction: str):
        self.reduction = reduction
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def calculate(self, validation: Validation) -> Loss:
        return Loss(self.bce_loss(validation.prediction, validation.label))

    def recalculate_weights(self):
        self.bce_loss.backward()

    def clone(self) -> Self:
        return cast(Self, BCELossCalculator(reduction=self.reduction))


class ClipBCELossCalculator(SingleLossCalculator):
    def calculate(self, validation: Validation) -> Loss:
        return Loss(binary_cross_entropy(validation.prediction, validation.label))

    def recalculate_weights(self):
        return

    def clone(self) -> Self:
        return cast(Self, ClipBCELossCalculator())


class MultipleLossCalculator(ABC):
    @abstractmethod
    def calculate(self, validations: list[Validation]) -> Loss:
        pass

    @abstractmethod
    def recalculate_weights(self):
        pass


class AverageMultipleLossCalculator(MultipleLossCalculator):
    def __init__(self, single_loss_calculator: SingleLossCalculator):
        super().__init__()
        self.single_loss_calculator = single_loss_calculator

    def calculate(self, validations: list[Validation]) -> Loss:
        if len(validations) == 0:
            return Loss(0.0)

        losses = [
            self.single_loss_calculator.calculate(validation)
            for validation in validations
        ]
        return Loss(sum(losses) / len(losses))

    def recalculate_weights(self):
        self.single_loss_calculator.recalculate_weights()


class BalancedAverageMultipleLossCalculator(MultipleLossCalculator):
    def __init__(self, single_loss_calculator: SingleLossCalculator):
        super().__init__()
        self.single_loss_calculators = single_loss_calculator
        self.loss_calculators_by_label: dict[Label, SingleLossCalculator] = {}

    def calculate(self, validations: list[Validation]) -> Loss:
        if len(validations) == 0:
            return Loss(0.0)

        self.loss_calculators_by_label = self._create_loss_calculators_by_label(
            validations
        )
        aggregated_predictions_by_label = self._aggregate_predictions_by_label(
            validations
        )
        aggregated_losses_by_label = self._calculate_losses_per_label(
            self.loss_calculators_by_label, aggregated_predictions_by_label
        )
        average_losses_by_label = self._calculate_average_loss_per_label(
            aggregated_losses_by_label
        )
        average_loss = self._calculate_average_loss(average_losses_by_label)
        return average_loss

    def recalculate_weights(self):
        for loss_calculator in self.loss_calculators_by_label.values():
            loss_calculator.recalculate_weights()

    def _create_loss_calculators_by_label(
        self, validations: list[Validation]
    ) -> dict[Label, SingleLossCalculator]:
        loss_calculators_by_label: dict[Label, SingleLossCalculator] = {}

        for validation in validations:
            if loss_calculators_by_label[validation.label] is None:
                loss_calculators_by_label[
                    validation.label
                ] = self.single_loss_calculators.clone()

        return loss_calculators_by_label

    @staticmethod
    def _aggregate_predictions_by_label(
        validations: list[Validation],
    ) -> dict[Label, list[Prediction]]:
        aggregated_predictions_by_label: dict[Label, list[Prediction]] = {}

        for validation in validations:
            if aggregated_predictions_by_label[validation.label] is None:
                aggregated_predictions_by_label[validation.label] = []

            aggregated_predictions_by_label[validation.label].append(
                validation.prediction
            )

        return aggregated_predictions_by_label

    @staticmethod
    def _calculate_losses_per_label(
        aggregated_loss_calculators: dict[Label, SingleLossCalculator],
        aggregated_predictions_by_label: dict[Label, list[Prediction]],
    ) -> dict[Label, list[Loss]]:
        aggregated_losses_by_label: dict[Label, list[Loss]] = {}

        for label in aggregated_predictions_by_label.keys():
            loss_calculator = aggregated_loss_calculators[label]
            aggregated_losses_by_label[label] = [
                loss_calculator.calculate(Validation(prediction, label))
                for prediction in aggregated_predictions_by_label[label]
            ]

        return aggregated_losses_by_label

    @staticmethod
    def _calculate_average_loss_per_label(
        aggregated_losses_by_label: dict[Label, list[Loss]],
    ) -> dict[Label, Loss]:
        average_losses_by_label: dict[Label, Loss] = {}

        for label in aggregated_losses_by_label.keys():
            assert (
                len(aggregated_losses_by_label[label]) > 0
            ), "Labels should appear only if there's an element with that label"

            average_losses_by_label[label] = Loss(
                sum(aggregated_losses_by_label[label])
                / len(aggregated_losses_by_label[label])
            )

        return average_losses_by_label

    @staticmethod
    def _calculate_average_loss(average_loss_by_label: dict[Label, Loss]) -> Loss:
        assert len(average_loss_by_label) > 0, "We should have at least one label!"
        return Loss(sum(average_loss_by_label) / len(average_loss_by_label))


def _create_single_train_loss_calculator(
    options: TrainingOptionsConfig,
) -> SingleLossCalculator:
    match options.use_clipping:
        case True:
            return ClipBCELossCalculator()
        case False:
            return BCELossCalculator(reduction="none")


def _create_single_test_loss_calculator() -> SingleLossCalculator:
    return BCELossCalculator(reduction="sum")


def _create_multiple_loss_calculator(
    options: TrainingOptionsConfig, single_loss_calculator: SingleLossCalculator
):
    match options.use_class_balancing:
        case True:
            return BalancedAverageMultipleLossCalculator(single_loss_calculator)
        case False:
            return AverageMultipleLossCalculator(single_loss_calculator)


def create_train_loss_calculator(
    options: TrainingOptionsConfig,
) -> MultipleLossCalculator:
    single_loss_calculator = _create_single_train_loss_calculator(options)
    return _create_multiple_loss_calculator(options, single_loss_calculator)


def create_test_loss_calculator(
    options: TrainingOptionsConfig,
) -> MultipleLossCalculator:
    single_loss_calculator = _create_single_test_loss_calculator()
    return _create_multiple_loss_calculator(options, single_loss_calculator)
