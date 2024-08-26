from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)


def main():
    # TODO read env variables and instantiate the dependencies
    service = ModelTrainingService(
        dataset_repository=None,
        dataset_splitter=None,
        model_trainer=None,
        trained_models_repository=None,
        config=None,
    )
    service.execute()


if __name__ == "__main__":
    main()
