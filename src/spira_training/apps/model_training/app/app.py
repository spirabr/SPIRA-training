from src.spira_training.apps.model_training.app.model_trainer import ModelTrainer
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)


class App:
    def __init__(
        self,
        audios_repository: AudiosRepository,
        model_trainer: ModelTrainer,
        trained_models_repository: TrainedModelsRepository,
    ):
        self.audios_repository = audios_repository
        self.model_trainer = model_trainer
        self.trained_models_repository = trained_models_repository

    def train_model(
        self,
        train_audios_path: str,
        validation_audios_path: str,
        model_storage_path: str,
    ) -> None:
        train_dataset = self.audios_repository.load_audios(train_audios_path)
        validation_dataset = self.audios_repository.load_audios(validation_audios_path)
        self.model_trainer.train_model(train_dataset, validation_dataset)
        self.trained_models_repository.save_model(
            self.model_trainer.model, model_storage_path
        )
