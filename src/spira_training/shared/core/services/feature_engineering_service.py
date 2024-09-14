from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import FeatureEngineeringConfig
from src.spira_training.shared.core.interfaces.random import Random
from src.spira_training.shared.ports import dataset_repository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.valid_path_reader import ValidPathReader


class AudioRepository:
    pass


class FeatureEngineeringService:
    def __init__(
            self,
            config: FeatureEngineeringConfig,
            randomizer: Random,
            dataset_repository: DatasetRepository,
            valid_path_reader: ValidPathReader,
            audio_repository: AudioRepository
    ):
        self.config = config
        self.randomizer = randomizer
        self.dataset_repository = dataset_repository
        self.valid_path_reader = valid_path_reader
        self.audio_repository = audio_repository

    def execute(self) -> None:
        patients_inputs, controls_inputs, noises = self._load_data()
        dataset = self._generate_dataset(self.config, self.randomizer, patients_inputs, controls_inputs, noises)
        dataset_repository.save_dataset(dataset, self.config.paths.dataset)

    def _load_data(self):
        patients_inputs = self._load_audio_data(self.config.audio.dataset_paths.patients_csv)
        controls_inputs = self._load_audio_data(self.config.audio.dataset_paths.controls_csv)
        noises = self._load_audio_data(self.config.audio.dataset_paths.noises_csv)

        return patients_inputs, controls_inputs, noises

    def _load_audio_data(self, csv_path):
        valid_paths = ValidPathReader.read_valid_paths(csv_path)
        audio_data = self._load_audio_from_paths(valid_paths)
        return audio_data

    def _load_audio_from_paths(self, paths):
        return self.audio_repository.load(
            paths,
            self.config.audio.hop_length,
            self.config.audio.normalize
        )

    def _generate_dataset(self):
        pass

