from abc import ABC, abstractmethod
from src.spira_training.shared.models.model_evaluation import ModelEvaluation


class ModelEvaluationRepository(ABC):
    @abstractmethod
    def publish_model_evaluation(self, model: ModelEvaluation, path: str) -> None:
        pass

    @abstractmethod
    def get_model_evaluation(self, path: str) -> ModelEvaluation:
        pass
