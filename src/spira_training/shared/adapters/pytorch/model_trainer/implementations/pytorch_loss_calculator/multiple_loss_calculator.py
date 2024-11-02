from typing import Sequence
from spira_training.shared.adapters.pytorch.models.pytorch_label import PytorchLabel
from spira_training.shared.adapters.pytorch.models.pytorch_tensor import PytorchTensor
from spira_training.shared.core.models.loss import Loss
import torch
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.single_loss_calculator import (
    SingleLossCalculator,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_loss_calculator import (
    PytorchLossCalculator,
)


class AverageMultipleLossCalculator(PytorchLossCalculator):
    def __init__(self, single_loss_calculator: SingleLossCalculator):
        super().__init__()
        self.single_loss_calculator = single_loss_calculator

    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> Loss:
        if len(predictions) == 0 or len(labels) == 0:
            return Loss(value=0.0)

        losses = [
            self.single_loss_calculator.calculate_single_loss_tensor(
                prediction=prediction, label=label
            )
            for prediction, label in zip(predictions, labels)
        ]
        loss_tensor = torch.tensor(sum(losses) / len(losses))
        return Loss(value=loss_tensor.tolist())

    def recalculate_weights(self):
        self.single_loss_calculator.recalculate_weights()


class BalancedAverageMultipleLossCalculator(PytorchLossCalculator):
    def __init__(self, single_loss_calculator: SingleLossCalculator):
        super().__init__()
        self.single_loss_calculators = single_loss_calculator
        self.loss_calculators_by_label: dict[PytorchLabel, SingleLossCalculator] = {}

    def calculate_loss(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> Loss:
        if len(predictions) == 0 or len(labels) == 0:
            return Loss(value=0.0)

        self.loss_calculators_by_label = self._create_loss_calculators_by_label(
            predictions=labels, labels=labels
        )
        aggregated_predictions_by_label = self._aggregate_predictions_by_label(
            predictions=labels, labels=labels
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
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> dict[PytorchLabel, SingleLossCalculator]:
        loss_calculators_by_label: dict[PytorchLabel, SingleLossCalculator] = {}

        for _, label in zip(predictions, labels):
            if loss_calculators_by_label[label] is None:
                loss_calculators_by_label[label] = self.single_loss_calculators.clone()

        return loss_calculators_by_label

    def _aggregate_predictions_by_label(
        self, predictions: Sequence[PytorchLabel], labels: Sequence[PytorchLabel]
    ) -> dict[PytorchLabel, Sequence[PytorchLabel]]:
        aggregated_predictions_by_label = {}

        for prediction, label in zip(predictions, labels):
            if aggregated_predictions_by_label[label] is None:
                aggregated_predictions_by_label[label] = []

            aggregated_predictions_by_label[label].append(prediction)

        return aggregated_predictions_by_label

    def _calculate_losses_per_label(
        self,
        aggregated_loss_calculators: dict[PytorchLabel, SingleLossCalculator],
        aggregated_predictions_by_label: dict[PytorchLabel, Sequence[PytorchLabel]],
    ) -> dict[PytorchLabel, Sequence[PytorchTensor]]:
        aggregated_losses_by_label: dict[PytorchLabel, Sequence[PytorchTensor]] = {}

        for label in aggregated_predictions_by_label.keys():
            loss_calculator = aggregated_loss_calculators[label]

            loss_tensors = [
                loss_calculator.calculate_single_loss_tensor(
                    prediction=prediction, label=label
                )
                for prediction in aggregated_predictions_by_label[label]
            ]
            aggregated_losses_by_label[label] = loss_tensors

        return aggregated_losses_by_label

    def _calculate_average_loss_per_label(
        self,
        aggregated_losses_by_label: dict[PytorchLabel, Sequence[PytorchTensor]],
    ) -> dict[PytorchLabel, PytorchTensor]:
        average_losses_by_label: dict[PytorchLabel, PytorchTensor] = {}

        for label in aggregated_losses_by_label.keys():
            assert (
                len(aggregated_losses_by_label[label]) > 0
            ), "Labels should appear only if there's an element with that label"

            loss_tensor = PytorchTensor(
                torch.tensor(
                    sum(aggregated_losses_by_label[label])
                    / len(aggregated_losses_by_label[label])
                )
            )
            average_losses_by_label[label] = loss_tensor

        return average_losses_by_label

    def _calculate_average_loss(
        self,
        average_loss_by_label: dict[PytorchLabel, PytorchTensor],
    ) -> Loss:
        assert len(average_loss_by_label) > 0, "We should have at least one label!"
        loss_tensor = torch.tensor(
            sum(average_loss_by_label.values()) / len(average_loss_by_label)
        )

        return Loss(value=loss_tensor.tolist())
