from pathlib import Path

from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_audio_factory import (
    PytorchTensorFactory,
)

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.shared.core.audio_processor_factory import (
    create_audio_processor,
)
from src.spira_training.shared.core.interfaces.random import Random
from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.file_reader import FileReader
from src.spira_training.shared.ports.path_validator import PathValidator


class FeatureEngineeringService:
    def __init__(
        self,
        config: FeatureEngineeringConfig,
        randomizer: Random,
        dataset_repository: DatasetRepository,
        audios_repository: AudiosRepository,
        file_reader: FileReader,
        path_validator: PathValidator,
        pytorch_audio_factory: PytorchTensorFactory,
    ):
        self.pytorch_audio_factory = pytorch_audio_factory
        self.config = config
        self.randomizer = randomizer
        self.dataset_repository = dataset_repository
        self.audios_repository = audios_repository
        self.file_reader = file_reader
        self.path_validator = path_validator

    async def execute(self, save_dataset_path: Path) -> None:
        patients_inputs, controls_inputs, noises = self._load_data()

        audio_processor = create_audio_processor(
            self.config.audio_processor, self.pytorch_audio_factory
        )

        dataset = self._generate_dataset()

        await self.dataset_repository.save_dataset(dataset, save_dataset_path)  # type: ignore

    def _load_data(self):
        patients_inputs = self._load_audio_data(
            self.config.audio.dataset_paths.patients_csv
        )
        controls_inputs = self._load_audio_data(
            self.config.audio.dataset_paths.controls_csv
        )
        noises = self._load_audio_data(self.config.audio.dataset_paths.noises_csv)

        return patients_inputs, controls_inputs, noises

    def _load_audio_data(self, csv_path: Path):
        validated_csv_path = self.path_validator.validate_path(csv_path)
        file_paths = self.file_reader.read(str(validated_csv_path))
        validated_file_paths = [
            self.path_validator.validate_path(Path(path)) for path in file_paths
        ]
        audio_data = [
            self._load_audio_from_paths(path) for path in validated_file_paths
        ]

        return audio_data

    def _load_audio_from_paths(self, path: ValidPath):
        return self.audios_repository.get_audio(str(path))

    def _generate_dataset(self):
        pass
        # TODO - generate dataset
