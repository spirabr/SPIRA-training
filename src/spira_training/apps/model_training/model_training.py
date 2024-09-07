from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)
from tests.fakes.fake_dataset_repository import FakeDatasetRepository


def main():
    # TODO read env variables and instantiate the dependencies
    dataset_repository = FakeDatasetRepository()
    service = ModelTrainingService(
        dataset_repository=dataset_repository,
        dataset_splitter=None,
        model_trainer=None,
        trained_models_repository=None,
        config=None,
    )
    service.execute()


if __name__ == "__main__":
    main()
