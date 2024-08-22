from unittest.mock import patch, MagicMock
import pytest
from src.spira_training.apps.feature_engineering.app.app import App
from src.spira_training.apps.feature_engineering.tests.test_config import TestConfig, TestRandomizer, TestDatasetRepository

@patch('src.spira_training.shared.core.services.data_processing.load_data')
@patch('src.spira_training.shared.core.services.data_processing.generate_dataset')
def test_execute(mock_generate_dataset, mock_load_data):
    config = TestConfig()
    randomizer = TestRandomizer()
    dataset_repository = TestDatasetRepository()

    app = App(config, randomizer, dataset_repository)

    patients_inputs = ['patient1']
    controls_inputs = ['control1']
    noises = ['noise1']
    dataset = "test_dataset"

    mock_load_data.return_value = (patients_inputs, controls_inputs, noises)
    mock_generate_dataset.return_value = dataset

    app.execute()

    assert dataset_repository.saved_datasets == [(dataset, config.paths.dataset)]